import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from typing import Callable, Tuple, Literal
from random import randint

class ForwardProcess:
    """
    Implementation by: Sven Elflein
    https://selflein.github.io/diffusion_practical_guide
    """
    def __init__(self, betas: torch.Tensor):
        self.beta = betas

        self.alphas = 1. - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=-1)


    def get_x_t(self, x_0: torch.Tensor, t: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process given the unperturbed sample x_0.
        
        Args:
            x_0: Original, unperturbed samples.
            t: Target timestamp of the diffusion process of each sample.
        
        Returns:
            Noise added to original sample and perturbed sample.
        """
        eps_0 = torch.randn_like(x_0).to(x_0)
        alpha_bar = self.alpha_bar[t, None]
        mean = (alpha_bar ** 0.5) * x_0
        std = ((1. - alpha_bar) ** 0.5)

        return (eps_0, mean + std * eps_0)
class ReverseProcess(ForwardProcess):
    """
    Implementation by: Sven Elflein
    https://selflein.github.io/diffusion_practical_guide
    """
    def __init__(self, betas: torch.Tensor, model: nn.Module):
        super().__init__(betas)
        self.model = model
        self.T = len(betas) - 1

        self.sigma = (
            (1 - self.alphas)
            * (1 - torch.roll(self.alpha_bar, 1)) / (1 - self.alpha_bar)
        ) ** 0.5
        self.sigma[1] = 0.

    @classmethod
    def sample_timesteps(batch_size: int, T: int) -> torch.LongTensor:
        return torch.randint(0, T, (batch_size,), dtype=torch.long)

    @classmethod
    def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, T)

    
    def get_x_t_minus_one(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
            t_vector = torch.full(size=(len(x_t),), fill_value=t, dtype=torch.long)
            eps = self.model(x_t, t=t_vector)
        
        eps *= (1 - self.alphas[t]) / ((1 - self.alpha_bar[t]) ** 0.5)
        mean =  1 / (self.alphas[t] ** 0.5) * (x_t - eps)
        return mean + self.sigma[t] * torch.randn_like(x_t)

    def sample(self, n_samples=1, full_trajectory=False):
        # Initialize with X_T ~ N(0, I)
        x_t = torch.randn(n_samples, 2)
        trajectory = [x_t.clone()]
        
        for t in range(self.T, 0, -1):
            x_t = self.get_x_t_minus_one(x_t, t=t)
            
            if full_trajectory:
                trajectory.append(x_t.clone())
        return torch.stack(trajectory, dim=0) if full_trajectory else x_t

def fit(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: Literal['cpu', 'cuda'],
    num_epochs: int = 25,
    T: int = 200,
    **kwargs
) -> nn.Module:
    for epoch in range(num_epochs):
        model.train()
        # t = randint(1, T)
        pbar = tqdm(dataloader)
        for batch_idx, (sequence, _) in enumerate(pbar):
            model.train()
            model = model.to(device)

            t = ReverseProcess.sample_timesteps(sequence.shape[0], T)

            synth = None ### TODO: Reverse Process

            loss = loss_fn(synth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_description(f"Epoch {epoch+1}, Running Loss: {running_loss / (batch_idx + 1):.4f}")