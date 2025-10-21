import os
import io
import json
import argparse
import hashlib

import numpy as np
import torch

from models.encoder import Encoder
from models.decoder import Decoder
from models.vq import VectorQuantizer
from utils.tiles import load_image, split_into_tiles
from utils.metrics import psnr, ssim
from utils.io import write_ntex
from codecs.entropy import compress_bytes


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
    # vq state includes codebook weights; we'll still embed codebook separately for self-sufficiency
    vq.load_state_dict(state['vq'])
    return enc, vq, dec, arch


def default_paths():
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    return os.path.join(ckpt_dir, 'minimal_vqvae.pt'), os.path.join(ckpt_dir, 'codebook.npz')


def main():
    parser = argparse.ArgumentParser(description='Compress image(s) using minimal VQ-VAE texture codec')
    parser.add_argument('--input', type=str, required=True, help='Input image path (PNG/JPG)')
    parser.add_argument('--output', type=str, required=True, help='Output .ntex path')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model .pt (if omitted, tries ./checkpoints/minimal_vqvae.pt)')
    parser.add_argument('--codebook', type=str, default=None, help='Path to codebook .npz (if omitted, tries ./checkpoints/codebook.npz)')
    parser.add_argument('--no-embed', action='store_true', help='Do not embed model+codebook into .ntex')
    parser.add_argument('--tile-size', type=int, default=64)
    parser.add_argument('--eval', action='store_true', help='Evaluate PSNR/SSIM by reconstructing after quantization before saving')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = args.model
    codebook_path = args.codebook
    if model_path is None or codebook_path is None:
        dflt_m, dflt_c = default_paths()
        model_path = model_path or dflt_m
        codebook_path = codebook_path or dflt_c
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.isfile(codebook_path):
        raise FileNotFoundError(f"Codebook not found: {codebook_path}")

    enc, vq, dec, arch = load_checkpoint(model_path, device)
    enc.eval(); vq.eval(); dec.eval()

    codebook = np.load(codebook_path)['embeddings'].astype(np.float32)
    codebook_hash = hashlib.sha256(codebook.tobytes()).hexdigest()[:16]

    # Load and tile input image
    img_t = load_image(args.input)  # (C,H,W) [0,1]
    tiles, grid_h, grid_w, orig_size = split_into_tiles(img_t, args.tile_size)

    # Encode tiles to indices
    with torch.no_grad():
        all_indices = []
        for tile in tiles:
            tile = tile.unsqueeze(0).to(device)
            z_e = enc(tile)
            z_q, indices, vq_loss = vq(z_e)
            indices = indices.squeeze(0).contiguous().view(-1).detach().cpu().numpy()
            all_indices.append(indices)
    indices_concat = np.concatenate(all_indices, axis=0)

    # Pack indices into bytes (uint16 is sufficient for codebook_size<=65535)
    dtype = np.uint16 if codebook.shape[0] <= 65535 else np.uint32
    indices_arr = indices_concat.astype(dtype)
    raw_bytes = indices_arr.tobytes(order='C')
    comp_bytes, codec_name = compress_bytes(raw_bytes)

    # Prepare meta
    # latent size per tile: tile_size/downsample
    downsample = arch.get('downsample', 8)
    lh = args.tile_size // downsample
    lw = args.tile_size // downsample
    meta = {
        'tile_size': args.tile_size,
        'downsample': downsample,
        'image_width': int(img_t.shape[2]),
        'image_height': int(img_t.shape[1]),
        'grid_w': int(grid_w),
        'grid_h': int(grid_h),
        'latent_w_per_tile': int(lw),
        'latent_h_per_tile': int(lh),
        'codebook_size': int(codebook.shape[0]),
        'embedding_dim': int(codebook.shape[1]),
        'index_dtype_bits': 16 if dtype == np.uint16 else 32,
        'codebook_id': codebook_hash,
        'entropy_codec': codec_name,
        'format_version': 1,
    }

    # Optionally reconstruct for eval
    if args.eval:
        with torch.no_grad():
            # Use quantized z_q from loop above reconstruct tile by tile
            recons = []
            it = iter(all_indices)
            i = 0
            for tile in tiles:
                # Build z_q from indices
                start = i * (lh * lw)
                end = start + (lh * lw)
                inds = indices_arr[start:end].astype(np.int64)
                i += 1
                e = torch.from_numpy(codebook[inds]).to(device)  # (lh*lw, D)
                e = e.view(lh, lw, -1).permute(2,0,1).unsqueeze(0).contiguous()  # (1,D,lh,lw)
                xh = dec(e).squeeze(0).cpu()
                recons.append(xh)
            from utils.tiles import merge_tiles
            recon_img = merge_tiles(recons, grid_h, grid_w, args.tile_size, orig_size)
            p = psnr(recon_img.unsqueeze(0), img_t.unsqueeze(0))
            try:
                s = ssim(recon_img.unsqueeze(0), img_t.unsqueeze(0))
            except Exception:
                s = None
            print(f"Eval: PSNR={p:.2f} SSIM={s}")
            meta['psnr'] = float(p)
            if s is not None:
                meta['ssim'] = float(s)

    # Embedding model and codebook if requested
    model_state = None
    codebook_arr = None
    if not args.no_embed:
        # Bundle decoder+encoder+vq state dicts
        state = torch.load(model_path, map_location='cpu')
        model_state = state
        codebook_arr = codebook

    # Write container
    write_ntex(args.output, meta, comp_bytes, model_state_dict=model_state, codebook_array=codebook_arr, codec_name=codec_name)
    print(f"Wrote {args.output} ({len(comp_bytes)} bytes compressed indices). Codec={codec_name}")


if __name__ == '__main__':
    main()
