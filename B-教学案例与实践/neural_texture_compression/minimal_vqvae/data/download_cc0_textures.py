import argparse
import os
import sys
import time
from urllib.parse import urlparse

try:
    import requests
except Exception:
    requests = None

CC0_URLS = [
    # A tiny curated list of CC0 preview images from Poly Haven and ambientCG
    # Poly Haven (CC0): https://polyhaven.com/
    "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/BricksPaintedBeige/BricksPaintedBeige_col_1k.jpg",
    "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/ConcretePlaster015/ConcretePlaster015_col_1k.jpg",
    "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/WoodFloor049/WoodFloor049_col_1k.jpg",
    "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/MetalPlates004/MetalPlates004_col_1k.jpg",
    "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/RockGranite010/RockGranite010_col_1k.jpg",
    # ambientCG (CC0): https://ambientcg.com/
    "https://ambientcg.com/get?file=Wood049_1K-JPG_Color.jpg",
    "https://ambientcg.com/get?file=Fabric058_1K-JPG_Color.jpg",
    "https://ambientcg.com/get?file=Asphalt006_1K-JPG_Color.jpg",
    "https://ambientcg.com/get?file=BrickWall022_1K-JPG_Color.jpg",
    "https://ambientcg.com/get?file=Concrete035_1K-JPG_Color.jpg",
]


def download(url, out_dir):
    if requests is None:
        print("The 'requests' package is required for downloading. Please install it: pip install requests")
        return False
    os.makedirs(out_dir, exist_ok=True)
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        name = os.path.basename(urlparse(url).path)
        if not name:
            name = f"tex_{int(time.time()*1000)}.jpg"
        out_path = os.path.join(out_dir, name)
        with open(out_path, 'wb') as f:
            f.write(r.content)
        print(f"Downloaded: {out_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download a small CC0 texture demo set (<= 50 images)")
    parser.add_argument("--out", type=str, default="./data/textures", help="Output directory")
    args = parser.parse_args()
    count = 0
    for url in CC0_URLS:
        ok = download(url, args.out)
        if ok:
            count += 1
    print(f"Done. {count} files downloaded to {args.out}.")


if __name__ == "__main__":
    main()
