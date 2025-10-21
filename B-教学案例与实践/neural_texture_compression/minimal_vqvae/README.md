# Minimal VQ-VAE Texture Compression Prototype

This is a minimal, runnable prototype of VQ‑VAE based texture (image) compression with three pipelines:
- Train (learn codebook + decoder)
- Compress (encode to indices + entropy coding)
- Decompress (entropy decode + decode indices to image)

It aims to be simple, CPU/GPU runnable, and parameterizable for later extension.

Key features
- Tile size: 64×64
- Downsample factor: 8 (configurable 8/16)
- Codebook size: default 2048 (configurable 1024–8192)
- Loss: MSE (+ optional LPIPS if installed)
- Straight‑Through Estimator for quantization
- Entropy codec: zlib by default; zstd used if the `zstandard` package is available
- Container: `.ntex` with metadata and compressed index stream; by default includes model+codebook so that decompression only needs the `.ntex` file

Directory structure
- models/{encoder.py, decoder.py, vq.py}
- data/{dataset.py, download_cc0_textures.py}
- utils/{tiles.py, metrics.py, io.py}
- codecs/entropy.py
- train.py, compress.py, decompress.py, eval.py
- configs/minimal.yaml

Requirements
- Python 3.9+
- PyTorch >= 2.0
- Pillow, numpy
- Optional: zstandard (for zstd entropy coding), lpips (for perceptual loss), torchvision (if you want to use its transforms)

Quickstart
1) (Optional) Download a small CC0 texture demo set (<50 images):
   python data/download_cc0_textures.py --out ./data/textures

2) Train for 1 epoch to obtain a usable codebook/weights (demo):
   python train.py --data ./data/textures --tile-size 64 --downsample 8 --codebook-size 2048 --epochs 1
   Outputs to ./checkpoints by default:
   - minimal_vqvae.pt (model weights)
   - codebook.npz (codebook embeddings)
   - train_log.json (metrics)

3) Compress an image (embeds model+codebook into .ntex by default):
   python compress.py --input demo.png --output demo.ntex

4) Decompress (only needs the .ntex if it embeds model+codebook):
   python decompress.py --input demo.ntex --output demo_recon.png

Notes
- If you prefer slimmer containers, you can disable model embedding via:
  python compress.py --input demo.png --output demo.ntex --no-embed
  In that case, decompression will require external weights/codebook:
  python decompress.py --input demo.ntex --output demo_recon.png --model ./checkpoints/minimal_vqvae.pt --codebook ./checkpoints/codebook.npz

- PSNR is always reported; SSIM is reported when dependencies are available.

License
- Demo texture downloader pulls CC0 licensed assets (see script for attributions). Use strictly for demo/testing.
