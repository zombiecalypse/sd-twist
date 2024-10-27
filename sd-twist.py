#!/usr/bin/env python3
import argparse
import os
import pprint
import random
import sys
import time
import torch
import torchvision.transforms.functional
import PIL.ImageOps
import collections
import uuid
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, StableDiffusionXLImg2ImgPipeline
from diffusers.callbacks import PipelineCallback
from diffusers.utils import load_image
from functools import partial
from tqdm.auto import tqdm
__doc__ = """
Make optical illusions from two prompts in a given style: The generated
image shows the first prompt, but when you flip it on its head, it looks
like the second prompt.

The quality generally suffers from trying to achieve two goal prompts
simultaneously. Less detailed styles typically do a lot better -- realistic
styles tend to be just two images overlaid and pretty obvious.
""".strip()

EPILOG = """
This script is held together by bubblegum and vague hopes. It's currently
using stabilityai/stable-diffusion-xl-base-1.0 with a lot of hacks to get
the memory use down. It has been tried on a NVIDIA GeForce RTX 3080 and uses
~8GiB of VRAM.
""".strip()

arg_parse = argparse.ArgumentParser(
        prog='PROG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=EPILOG)
arg_parse.add_argument('-o', '--output',
                       default=os.path.expanduser('~/sd_output'),
                       help=('Folder to create the images in. The script will happily '
                             'delete files, so beware.'))
arg_parse.add_argument('-s', '--styles',
                       help=('Semicolon separated list of styles to try.'))
arg_parse.add_argument('-sf', '--styles_file',
                       help=('File where each line is a style prompt to try.'))
arg_parse.add_argument('-p1', '--prompt1',
                       help=('Semicolon separated list of first prompt to try.'))
arg_parse.add_argument('-p2', '--prompt2',
                       help=('Semicolon separated list of second prompt to try.'))
arg_parse.add_argument('-p1f', '--prompt1_file',
                       help=('File where each line is a first prompt to try.'))
arg_parse.add_argument('-p2f', '--prompt2_file',
                       help=('File where each line is a second prompt to try.'))
arg_parse.add_argument('-np', '--negative_prompt',
                       help=('Negative prompt applied to all generated images.'))
arg_parse.add_argument('-npf', '--negative_prompt_file',
                       help=('File with a negative prompt applied to all generated images.'))
arg_parse.add_argument('-n', '--num_images',
                       default=10,
                       type=int,
                       help=('Number of images to generate. If the combination of '
                             'style x prompt1 x prompt2 is larger than this, the script will '
                             'pick a random sample. If it\'s smaller, the script will generate '
                             'multiple images for some options.'))
arg_parse.add_argument('-i', '--inference_steps',
                       default=50,
                       type=int,
                       help=('Number of inference steps. The best value depends on the style, '
                             'but 50 works pretty well in general.'))
arg_parse.add_argument('-t', '--transform',
                       choices=['flip', 'rot90'],
                       default='flip',
                       help=('Transform that reveals the second image. Can be "flip" or "rot90".'))

def get_models():
    """Small-ish model sdxl works on 8GiB VRAM."""
    torch.set_grad_enabled(False)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = True

    model_id = 'stabilityai/stable-diffusion-xl-base-1.0'

    vae = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix',
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant='fp16',
        vae=vae,
    )
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-refiner-1.0',
        text_encoder_2=pipe.text_encoder_2,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant='fp16',
        vae=pipe.vae,
    )

    pipe.enable_model_cpu_offload()
    refiner.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    refiner.enable_xformers_memory_efficient_attention()

    return pipe, refiner

class BaseFlipCallback(PipelineCallback):
    def __init__(self, prompt1, prompt2):
        super().__init__()
        # Order of what comes after this step, so prompt2 is next first.
        self.prompts = (prompt2, prompt1)
        self.nx = self.transforms[0]
        self.updates = 0

    @property
    def balanced(self):
        return self.updates % 2 == 0

    @property
    def tensor_inputs(self):
        return [
                'latents',
                'prompt_embeds',
                'negative_prompt_embeds',
                'add_text_embeds',
                'negative_pooled_prompt_embeds',
        ]

    def callback_fn(self, p, step, timestep, kwargs):
        self.updates += 1
        p = self.prompts[step%2]
        latents = kwargs['latents']
        t = self.transforms[step%2]
        self.nx = self.transforms[(step+1)%2]
        latents = t(latents)
        prompt_embeds = torch.cat([p.negative_prompt_embeds, p.prompt_embeds], dim=0)
        add_text_embeds = torch.cat([p.negative_pooled_prompt_embeds, p.pooled_prompt_embeds], dim=0)

        kwargs.update(**dict(
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=p.negative_prompt_embeds,
                add_text_embeds=add_text_embeds,
        ))
        return kwargs

    def balance(self, latents):
        """Call callback after the run to ensure it's original-side up."""
        if not self.balanced:
            self.updates += 1
            return self.nx(latents)
        else:
            return latents

class FlipCallback(BaseFlipCallback):
    @property
    def transforms(self):
        return (
                # lambda x: torchvision.transforms.functional.rotate(x, -90),
                # lambda x: torchvision.transforms.functional.rotate(x, 90),
                lambda x: torch.flip(x, [2]),
                lambda x: torch.flip(x, [2]),
        )

class TurnCallback(BaseFlipCallback):
    @property
    def transforms(self):
        return (
                lambda x: torchvision.transforms.functional.rotate(x, -90),
                lambda x: torchvision.transforms.functional.rotate(x, 90),
        )

def mkdir(d):
    """mkdir -p: Ignore if directory exists."""
    try:
        os.makedirs(d)
    except: pass

def make_prompt(model, s, n):
    return Prompt(*model.encode_prompt(
        prompt=s,
        prompt_2=s,
        negative_prompt=n,
        negative_prompt_2=n,
    ))

def print_memory_usage():
    max_memory = round(torch.cuda.max_memory_allocated(
        device='cuda') / 1000000000, 2)
    print('Max. memory used:', max_memory, 'GB')

Prompt = collections.namedtuple('Prompt', [
    "prompt_embeds",
    "negative_prompt_embeds",
    "pooled_prompt_embeds",
    "negative_pooled_prompt_embeds",
])
Input = collections.namedtuple('Input', [
    "style_num",
    "style",
    "p1_num",
    "prompt1_obj",
    "p2_num",
    "prompt2_obj",
    "flip",
])
Output = collections.namedtuple('Output', [
    "path",
    "style_num",
    "p1_num",
    "p2_num",
    "flip",
])

def get_sample(styles, prompt1_opts, prompt2_opts, sample_size=100):
    options = []
    for style_num, style in enumerate(styles):
        for p1_num, prompt1_obj in enumerate(prompt1_opts):
            for p2_num, prompt2_obj in enumerate(prompt2_opts):
                for flip in [False, True]:
                    options.append(Input(style_num, style, p1_num, prompt1_obj, p2_num, prompt2_obj, flip))

    random.shuffle(options)
    return sorted(options[:sample_size])

def make_link_tree(output_folder, outputs):
    for o in outputs:
        for dim in [f'style{o.style_num:02}', f'p1_{o.p1_num:02}', f'p2_{o.p2_num:02}']:
            dir_path = os.path.join(output_folder, dim)
            mkdir(dir_path)
            fl = 'flip' if o.flip else 'noflip'
            old_base = os.path.basename(o.path)
            basename = '.'.join([fl, f'style{o.style_num:02}', f'p1_{o.p1_num:02}', f'p2_{o.p2_num:02}', old_base])
            out_path = os.path.join(dir_path, basename)
            if os.path.exists(out_path):
                os.remove(out_path)
            os.symlink(o.path, os.path.join(dir_path, basename))

def make_image(pipe, refiner, inp, negative_prompt, seed, inference_steps, callback_class):
    height = 1024
    width = 1024
    chans = 4
    batch = 1
    refiner_start=0.8
    prompt1 = '; '.join([
        inp.prompt1_obj,
        inp.style,
    ])
    prompt2 = '; '.join([
        inp.prompt2_obj,
        inp.style,
    ])
    if inp.flip:
        prompt1, prompt2 = prompt2, prompt1

    print(f'flip={inp.flip}')
    print(f'prompt1={inp.style_num:03}:{inp.p1_num:03}: {prompt1}')
    print(f'prompt2={inp.style_num:03}:{inp.p2_num:03}: {prompt2}')
    # Reproducible
    torch.manual_seed(hash(seed))

    latents = torch.normal(0., 1., size=(
        batch,
        chans,
        height // pipe.vae_scale_factor,
        width // pipe.vae_scale_factor)).half()
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            make_prompt(pipe, prompt1, negative_prompt)
    )
    cb = callback_class(
            make_prompt(pipe, prompt1, negative_prompt),
            make_prompt(pipe, prompt2, negative_prompt),
    )
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=inference_steps,
        height=height,
        width=width,
        guidance_scale=5.0,
        denoising_end=refiner_start,
        output_type='latent',
        callback_on_step_end=cb,
        latents=latents,
        callback_on_step_end_tensor_inputs=cb.tensor_inputs, 
    ).images
    image = cb.balance(image)
    assert cb.updates % 2 == 0
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            make_prompt(refiner, prompt1, negative_prompt)
    )
    cb = callback_class(
            make_prompt(refiner, prompt1, negative_prompt),
            make_prompt(refiner, prompt2, negative_prompt),
    )
    image = refiner(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=inference_steps,
        denoising_start=refiner_start,
        image=image,
        output_type='pil',
        callback_on_step_end=cb,
        callback_on_step_end_tensor_inputs=cb.tensor_inputs, 
    ).images[0]

    if not cb.balanced:
        image = PIL.ImageOps.flip(image)
    return image

def run(output_folder, styles, prompt1_opts, prompt2_opts, sample_size, inference_steps, callback_class, negative_prompt):
    pipe, refiner = get_models()
    total_done = 0

    options = get_sample(styles, prompt1_opts, prompt2_opts, sample_size=sample_size)
    pprint.pprint(options)
    outputs = []

    for i in tqdm(range(sample_size)):
        inp = options[i%len(options)]
        base = uuid.uuid4()
        image = make_image(pipe, refiner, inp, negative_prompt,
                           seed=str(base), inference_steps=inference_steps,
                           callback_class=callback_class)

        source_dir = os.path.join(output_folder, 'output')
        mkdir(source_dir)
        file_path = os.path.join(source_dir, f'sdxl_{base}.png')
        image.save(file_path)
        outputs.append(Output(file_path, inp.style_num, inp.p1_num, inp.p2_num, inp.flip))
        total_done += 1
    make_link_tree(output_folder, outputs)
    print_memory_usage()

def from_arg_or_file(arg, file_path, default=None):
    if arg:
        return [s.strip() for s in arg.split(';')]
    elif file_path:
        with open(file_path) as f:
            return [s.strip() for s in f.readlines() if s.strip()]
    else:
        return default

if __name__ == '__main__':
    args = arg_parse.parse_args()
    styles = from_arg_or_file(args.styles, args.styles_file, default=[''])
    prompts1 = from_arg_or_file(args.prompt1, args.prompt1_file, default=None)
    prompts2 = from_arg_or_file(args.prompt2, args.prompt2_file, default=None)
    negative_prompt = '; '.join(from_arg_or_file(
        args.negative_prompt, args.negative_prompt_file, default=[]))
    if not prompts1:
        print('either --prompt1 or --prompt1_file is required.', file=sys.stderr)
        sys.exit(1)
    if not prompts2:
        print('either --prompt2 or --prompt2_file is required.', file=sys.stderr)
        sys.exit(1)
    if args.transform == "flip":
        callback_class = FlipCallback
    elif args.transform == "rot90":
        callback_class = TurnCallback
    else:
        sys.exit(1)
    run(
        args.output,
        styles,
        prompts1,
        prompts2,
        args.num_images,
        args.inference_steps,
        callback_class,
        negative_prompt=negative_prompt,
    )
