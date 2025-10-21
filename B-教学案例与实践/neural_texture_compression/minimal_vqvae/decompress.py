import os
import argparse
import numpy as np
import torch

from models.encoder import Encoder
from models.decoder import Decoder
from models.vq import VectorQuantizer
from utils.tiles import save_image, merge_tiles
from utils.metrics import psnr, ssim
from utils.io import read_ntex
from codecs.entropy import decompress_bytes


def build_from_arch(arch, device):
    embed_dim = arch.get('embed_dim', 64)
    hidden = arch.get('hidden', 128)
    downsample = arch.get('downsample', 8)
    enc = Encoder(3, hidden, embed_dim, downsample).to(device)
    dec = Decoder(3, hidden, embed_dim, downsample).to(device)
    vq = VectorQuantizer(codebook_size=arch.get('codebook_size', 2048), embed_dim=embed_dim, beta=0.25).to(device)
    return enc, vq, dec


def main():
    parser = argparse.ArgumentParser(description='Decompress .ntex to PNG')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, default=None, help='Optional: model .pt path if not embedded')
    parser.add_argument('--codebook', type=str, default=None, help='Optional: codebook .npz path if not embedded')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    meta, codec_name, data_bytes, model_state, codebook_array = read_ntex(args.input)

    if model_state is None or codebook_array is None:
        if args.model is None or args.codebook is None:
            raise RuntimeError('Model and codebook not embedded. Please provide --model and --codebook paths.')
        import torch
        model_state = torch.load(args.model, map_location='cpu')
        codebook_array = np.load(args.codebook)['embeddings'].astype(np.float32)

    arch = model_state.get('arch', {})
    enc, vq, dec = build_from_arch(arch, device)
    enc.load_state_dict(model_state['encoder'])
    dec.load_state_dict(model_state['decoder'])
    vq.load_state_dict(model_state['vq'])
    enc.eval(); dec.eval(); vq.eval()

    # Decode indices
    raw = decompress_bytes(data_bytes, meta.get('entropy_codec', codec_name))
    dtype = np.uint16 if meta.get('index_dtype_bits', 16) == 16 else np.uint32
    inds = np.frombuffer(raw, dtype=dtype)

    tile_size = int(meta['tile_size'])
    grid_w = int(meta['grid_w'])
    grid_h = int(meta['grid_h'])
    lw = int(meta['latent_w_per_tile'])
    lh = int(meta['latent_h_per_tile'])
    H = int(meta['image_height'])
    W = int(meta['image_width'])

    # Reconstruct tiles
    tiles = []
    D = int(meta['embedding_dim'])
    num_tiles = grid_w * grid_h
    expected = num_tiles * (lh * lw)
    if inds.size != expected:
        raise RuntimeError(f"Mismatched indices length: got {inds.size}, expected {expected}")

    with torch.no_grad():
        codebook = codebook_array
        for i in range(num_tiles):
            start = i * (lh * lw)
            end = start + (lh * lw)
            idx = inds[start:end].astype(np.int64)
            e = torch.from_numpy(codebook[idx]).to(device)  # (lh*lw, D)
            e = e.view(lh, lw, -1).permute(2,0,1).unsqueeze(0).contiguous()  # (1,D,lh,lw)
            xh = dec(e).squeeze(0).cpu()
            tiles.append(xh)

    recon = merge_tiles(tiles, grid_h, grid_w, tile_size, (H, W))
    save_image(args.output, recon)
    print(f"Wrote {args.output}")

    # Optional metrics if original image embedded not available; we only have final image
    # Metrics can be computed by the compressor at creation time and stored in meta
    if 'psnr' in meta:
        print(f"(Stored) PSNR={meta['psnr']:.2f}")
    if 'ssim' in meta:
        print(f"(Stored) SSIM={meta['ssim']:.4f}")


if __name__ == '__main__':
    main()
