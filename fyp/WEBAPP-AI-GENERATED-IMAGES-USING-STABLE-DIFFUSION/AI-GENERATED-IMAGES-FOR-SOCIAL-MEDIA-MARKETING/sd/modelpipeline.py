import torch
import numpy as np
from tqdm import tqdm
from ddpmsampler import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        gen = torch.Generator(device=device)
        if seed is None:
            gen.seed()
        else:
            gen.manual_seed(seed)
        clip = models["clip"]
        clip.to(device)       
        if do_cfg:
            cond_toks = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_toks = torch.tensor(cond_toks, dtype=torch.long, device=device)
            cond_ctx = clip(cond_toks)
            uncond_toks = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_toks = torch.tensor(uncond_toks, dtype=torch.long, device=device)
            uncond_ctx = clip(uncond_toks)
            ctx = torch.cat([cond_ctx, uncond_ctx])
        else:
            toks = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            toks = torch.tensor(toks, dtype=torch.long, device=device)
            ctx = clip(toks)
        to_idle(clip)
        if sampler_name == "ddpm":
            sampler = DDPMSampler(gen)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        lts_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            input_img_tensor = input_image.resize((WIDTH, HEIGHT))
            input_img_tensor = np.array(input_img_tensor)
            input_img_tensor = torch.tensor(input_img_tensor, dtype=torch.float32)
            input_img_tensor = rescale(input_img_tensor, (0, 255), (-1, 1))
            input_img_tensor = input_img_tensor.unsqueeze(0)
            input_img_tensor = input_img_tensor.permute(0, 3, 1, 2)
            encoder_noise = torch.randn(lts_shape, generator=gen, device=device)
            lts = encoder(input_img_tensor, encoder_noise)
            sampler.set_strength(strength=strength)
            lts = sampler.add_noise(lts, sampler.steps[0])
            to_idle(encoder)
        else:
            lts = torch.randn(lts_shape, generator=gen, device=device)
        diffusion = models["diffusion"]
        diffusion.to(device)
        steps = tqdm(sampler.steps)
        for i, step in enumerate(steps):
            time_embd = get_time_embedding(step).to(device)
            mdl_inpt = lts
            if do_cfg:
                mdl_inpt = mdl_inpt.repeat(2, 1, 1, 1)
            mdl_out = diffusion(mdl_inpt, ctx, time_embd)
            if do_cfg:
                output_cond, output_uncond = mdl_out.chunk(2)
                mdl_out = cfg_scale * (output_cond - output_uncond) + output_uncond
            lts = sampler.step(step, lts, mdl_out)
        to_idle(diffusion)
        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(lts)
        to_idle(decoder)
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_rng, new_rng, clamp=False):
    old_min, old_max = old_rng
    new_min, new_max = new_rng
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(step):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([step], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
