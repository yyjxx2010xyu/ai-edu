import math
import numpy as np
import torch
from PIL import Image


def pad_to_tile(img_t: torch.Tensor, tile_size: int):
    # img_t: (C,H,W)
    C, H, W = img_t.shape
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if pad_h == 0 and pad_w == 0:
        return img_t, (H, W)
    img_p = torch.nn.functional.pad(img_t.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
    return img_p, (H, W)


def split_into_tiles(img_t: torch.Tensor, tile_size: int):
    # Returns tiles tensor list, grid_h, grid_w, original_size
    img_p, orig_size = pad_to_tile(img_t, tile_size)
    C, H, W = img_p.shape
    tiles = []
    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tiles.append(img_p[:, y:y+tile_size, x:x+tile_size])
    grid_h = H // tile_size
    grid_w = W // tile_size
    return tiles, grid_h, grid_w, orig_size


def merge_tiles(tiles, grid_h, grid_w, tile_size, orig_size):
    # tiles: list of (C,ts,ts)
    C = tiles[0].shape[0]
    Ht = grid_h * tile_size
    Wt = grid_w * tile_size
    out = torch.zeros((C, Ht, Wt), dtype=tiles[0].dtype, device=tiles[0].device)
    i = 0
    for gy in range(grid_h):
        for gx in range(grid_w):
            out[:, gy*tile_size:(gy+1)*tile_size, gx*tile_size:(gx+1)*tile_size] = tiles[i]
            i += 1
    H, W = orig_size
    return out[:, :H, :W]


def load_image(path):
    im = Image.open(path).convert('RGB')
    arr = np.array(im, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2,0,1).contiguous()
    return t


def save_image(path, tensor):
    # tensor: (C,H,W), [0,1]
    t = tensor.detach().cpu().clamp(0,1)
    arr = (t.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)
