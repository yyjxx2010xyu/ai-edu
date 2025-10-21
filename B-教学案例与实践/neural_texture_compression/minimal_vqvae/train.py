import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.encoder import Encoder
from models.decoder import Decoder
from models.vq import VectorQuantizer
from data.dataset import TextureTilesDataset
from utils.metrics import psnr, ssim


def build_model(args, device):
    enc = Encoder(in_channels=3, hidden_channels=args.hidden, embed_dim=args.embed_dim, downsample=args.downsample).to(device)
    dec = Decoder(out_channels=3, hidden_channels=args.hidden, embed_dim=args.embed_dim, downsample=args.downsample).to(device)
    vq = VectorQuantizer(codebook_size=args.codebook_size, embed_dim=args.embed_dim, beta=args.beta).to(device)
    return enc, vq, dec


def train_one_epoch(dataloader, enc, vq, dec, optimizer, device, use_lpips=False):
    enc.train(); vq.train(); dec.train()
    lpips_loss = None
    if use_lpips:
        try:
            import lpips
            lpips_loss = lpips.LPIPS(net='vgg').to(device)
        except Exception:
            lpips_loss = None
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0
    for tiles in dataloader:
        tiles = tiles.to(device)
        optimizer.zero_grad()
        z_e = enc(tiles)
        z_q, indices, vq_loss = vq(z_e)
        x_hat = dec(z_q)
        rec_loss = F.mse_loss(x_hat, tiles)
        loss = rec_loss + vq_loss
        if lpips_loss is not None:
            l = lpips_loss(x_hat, tiles).mean()
            loss = loss + 0.1 * l
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item()
            total_psnr += psnr(x_hat.clamp(0,1), tiles.clamp(0,1))
            try:
                total_ssim += ssim(x_hat.clamp(0,1), tiles.clamp(0,1))
            except Exception:
                pass
            n += 1
    return total_loss / max(1, n), total_psnr / max(1, n), total_ssim / max(1, n)


def save_outputs(enc, vq, dec, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Infer architecture parameters for later reconstruction
    try:
        embed_dim = enc.net[-1].out_channels
        hidden = enc.net[0].out_channels
    except Exception:
        embed_dim = None
        hidden = None
    state = {
        'encoder': enc.state_dict(),
        'decoder': dec.state_dict(),
        'vq': vq.state_dict(),
        'arch': {
            'embed_dim': embed_dim,
            'hidden': hidden,
            # Downsample can be inferred by counting Upsample/Conv stride 2 layers is non-trivial; store explicitly if present on modules
            # For minimal prototype, decoder/encoder were constructed with a specific downsample; store via attribute if exists
        }
    }
    # Attach a hint for downsample if layers include Upsample operations
    # This does not guarantee correctness across edits; acceptable for minimal prototype
    # Try to deduce downsample by comparing encoder input/output spatial ratio for a 64x64 dummy
    try:
        import torch
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            z = enc(x)
        ds = 64 // z.shape[-1]
        state['arch']['downsample'] = int(ds)
    except Exception:
        pass

    model_path = os.path.join(out_dir, 'minimal_vqvae.pt')
    torch.save(state, model_path)

    codebook = vq.get_codebook().numpy().astype(np.float32)
    np.savez_compressed(os.path.join(out_dir, 'codebook.npz'), embeddings=codebook)
    return model_path, os.path.join(out_dir, 'codebook.npz')


def main():
    parser = argparse.ArgumentParser(description='Train minimal VQ-VAE for texture compression')
    parser.add_argument('--data', type=str, required=True, help='Path to folder with PNG/JPG images')
    parser.add_argument('--tile-size', type=int, default=64)
    parser.add_argument('--downsample', type=int, default=8, choices=[8,16])
    parser.add_argument('--codebook-size', type=int, default=2048)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--out', type=str, default='./checkpoints')
    parser.add_argument('--lpips', action='store_true', help='Use LPIPS in loss if available')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TextureTilesDataset(args.data, tile_size=args.tile_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    enc, vq, dec = build_model(args, device)
    params = list(enc.parameters()) + list(vq.parameters()) + list(dec.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    logs = []
    for epoch in range(args.epochs):
        loss, p, s = train_one_epoch(loader, enc, vq, dec, optimizer, device, use_lpips=args.lpips)
        print(f"Epoch {epoch+1}/{args.epochs} - loss={loss:.4f} PSNR={p:.2f} SSIM={s:.3f}")
        logs.append({'epoch': epoch+1, 'loss': loss, 'psnr': p, 'ssim': s})

    model_path, codebook_path = save_outputs(enc, vq, dec, args.out)

    with open(os.path.join(args.out, 'train_log.json'), 'w') as f:
        json.dump({'logs': logs, 'model_path': model_path, 'codebook_path': codebook_path, 'args': vars(args)}, f, indent=2)

    print(f"Saved model to {model_path} and codebook to {codebook_path}")


if __name__ == '__main__':
    main()
