import torch
import numpy as np

class DDPMSampler:
    def __init__(self, gen: torch.Generator, num_train_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.gen = gen
        self.num_train_steps = num_train_steps
        self.steps = torch.from_numpy(np.arange(0, num_train_steps)[::-1].copy())

    def set_inference_steps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_steps // self.num_inference_steps
        steps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.steps = torch.from_numpy(steps)

    def _get_prev_step(self, step: int) -> int:
        prev_step = step - self.num_train_steps // self.num_inference_steps
        return prev_step
    
    def _get_variance(self, step: int) -> torch.Tensor:
        prev_step = self._get_prev_step(step)
        alpha_prod_step = self.alphas_cumprod[step]
        alpha_prod_prev_step = self.alphas_cumprod[prev_step] if prev_step >= 0 else self.one
        current_beta_step = 1 - alpha_prod_step / alpha_prod_prev_step
        variance = (1 - alpha_prod_prev_step) / (1 - alpha_prod_step) * current_beta_step
        variance = torch.clamp(variance, min=1e-20)
        return variance
    
    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.steps = self.steps[start_step:]
        self.start_step = start_step

    def step(self, step: int, lts: torch.Tensor, mdl_out: torch.Tensor):
        t = step
        prev_t = self._get_prev_step(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev_t = 1 - alpha_prod_prev_t
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t
        current_beta_t = 1 - current_alpha_t
        pred_orig_sample = (lts - beta_prod_t ** (0.5) * mdl_out) / alpha_prod_t ** (0.5)
        pred_orig_sample_coeff = (alpha_prod_prev_t ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_prev_t / beta_prod_t
        pred_prev_sample = pred_orig_sample_coeff * pred_orig_sample + current_sample_coeff * lts
        variance = 0
        if t > 0:
            device = mdl_out.device
            noise = torch.randn(mdl_out.shape, generator=self.gen, device=device, dtype=mdl_out.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
    
    def add_noise(
        self,
        orig_samples: torch.FloatTensor,
        steps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(device=orig_samples.device, dtype=orig_samples.dtype)
        steps = steps.to(orig_samples.device)
        sqrt_alpha_prod = alphas_cumprod[steps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(orig_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[steps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(orig_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise = torch.randn(orig_samples.shape, generator=self.gen, device=orig_samples.device, dtype=orig_samples.dtype)
        noisy_samples = sqrt_alpha_prod * orig_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

