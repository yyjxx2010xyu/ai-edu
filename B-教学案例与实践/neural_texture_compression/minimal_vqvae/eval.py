import os
import glob
import argparse
import numpy as np
import torch

from models.encoder import Encoder
from models.decoder import Decoder
from models.vq import VectorQuantizer
from utils.tiles import load_image, split_into_tiles, merge_tiles
from utils.metrics import psnr, ssim


def list_images(path):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    if os.path.isdir(path):
        for ext in exts:
            files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
    else:
        files = [path]
    return sorted(files)


def load_checkpoint(model_path: str, device):
    state = torch.load(model_path, map_location=device)
    arch = state.get('arch', {})
    embed_dim = arch.get('embed_dim', 64)
    hidden = arch.get('hidden', 128)
    downsample = arch.get('downsample', 8)
    enc = Encoder(3, hidden, embed_dim, downsample).to(device)
    dec = Decoder(3, hidden, embed_dim, downsample).to(device)
    vq = VectorQuantizer(codebook_size=2048, embed_dim=embed_dim, beta=0.25).to(device)
    enc.load_state_dict(state['encoder'])
    dec.load_state_dict(state['decoder'])
    vq.load_state_dict(state['vq'])
    return enc, vq, dec, arch


def main():
    parser = argparse.ArgumentParser(description='Evaluate PSNR/SSIM over image(s) without writing .ntex')
    parser.add_argument('--input', type=str, required=True, help='Image file or folder')
    parser.add_argument('--model', type=str, default='./checkpoints/minimal_vqvae.pt')
    parser.add_argument('--codebook', type=str, default='./checkpoints/codebook.npz')
    parser.add_argument('--tile-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc, vq, dec, arch = load_checkpoint(args.model, device)
    enc.eval(); vq.eval(); dec.eval()
    codebook = np.load(args.codebook)['embeddings'].astype(np.float32)

    files = list_images(args.input)
    assert len(files) > 0, 'No images found'

    downsample = arch.get('downsample', 8)
    lh = args.tile_size // downsample
    lw = args.tile_size // downsample

    psnrs = []
    ssims = []

    with torch.no_grad():
        for fp in files:
            img_t = load_image(fp)
            tiles, gh, gw, orig = split_into_tiles(img_t, args.tile_size)
            recons = []
            for tile in tiles:
                tile = tile.unsqueeze(0).to(device)
                z_e = enc(tile)
                z_q, indices, _ = vq(z_e)
                inds = indices.view(-1).detach().cpu().numpy().astype(np.int64)
                e = torch.from_numpy(codebook[inds]).to(device)
                e = e.view(lh, lw, -1).permute(2,0,1).unsqueeze(0).contiguous()
                xh = dec(e).squeeze(0).cpu()
                recons.append(xh)
            recon = merge_tiles(recons, gh, gw, args.tile_size, orig)
            p = psnr(recon.unsqueeze(0), img_t.unsqueeze(0))
            try:
                s = ssim(recon.unsqueeze(0), img_t.unsqueeze(0))
            except Exception:
                s = None
            psnrs.append(p)
            if s is not None:
                ssims.append(s)
            print(f"{os.path.basename(fp)}: PSNR={p:.2f}{' SSIM='+str(round(s,4)) if s is not None else ''}")

    print(f"Average PSNR: {np.mean(psnrs):.2f}")
    if len(ssims) > 0:
        print(f"Average SSIM: {np.mean(ssims):.4f}")


if __name__ == '__main__':
    main()
