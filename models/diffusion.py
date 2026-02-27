import torch
from tqdm import tqdm


class Diffusion:
    def __init__(self, diffusion_steps, img_size, device):
        self.diffusion_steps = diffusion_steps
        self.img_size = img_size
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, diffusion_steps).to(device)
        self.alpha = (1. - self.beta).to(device)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).to(device)

    def noise_images(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        a_hat = self.alpha_hat[t][:, None, None, None]
        return torch.sqrt(a_hat) * x + torch.sqrt(1 - a_hat) * noise, noise

    @torch.no_grad()
    def sample(self, model, n):
        model.eval()
        x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)

        for i in tqdm(range(self.diffusion_steps - 1, -1, -1), desc='Generating'):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            pred_noise = model(x, t)

            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * pred_noise) + torch.sqrt(beta) * noise

        return x
