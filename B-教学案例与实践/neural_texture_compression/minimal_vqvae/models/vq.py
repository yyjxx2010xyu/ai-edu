import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=2048, embed_dim=64, beta=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e):
        # z_e: (B, D, H, W)
        B, D, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous()  # (B, H, W, D)
        flat_z = z.view(-1, D)
        # Compute distances to embeddings
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z.e
        e = self.embedding.weight  # (K, D)
        z_sq = (flat_z ** 2).sum(dim=1, keepdim=True)  # (N,1)
        e_sq = (e ** 2).sum(dim=1)  # (K)
        ze = flat_z @ e.t()  # (N,K)
        distances = z_sq + e_sq - 2 * ze
        indices = torch.argmin(distances, dim=1)  # (N)
        z_q = self.embedding(indices).view(B, H, W, D)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

        # VQ-VAE losses
        # Codebook loss: ||sg[z_e] - e||^2
        # Commitment loss: beta * ||z_e - sg[e]||^2
        z_e_stopped = z_e.detach()
        z_q_stopped = z_q.detach()
        codebook_loss = F.mse_loss(z_q, z_e_stopped)
        commitment_loss = F.mse_loss(z_e, z_q_stopped)
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        indices = indices.view(B, H, W)
        return z_q, indices, vq_loss

    def get_codebook(self):
        return self.embedding.weight.detach().cpu()
