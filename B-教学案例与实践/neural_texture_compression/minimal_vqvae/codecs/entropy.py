import zlib

try:
    import zstandard as zstd
    HAS_ZSTD = True
except Exception:
    zstd = None
    HAS_ZSTD = False


def compress_bytes(data: bytes, level: int = 3):
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data), 'zstd'
    else:
        return zlib.compress(data, level), 'zlib'


def decompress_bytes(data: bytes, codec: str):
    if codec == 'zstd':
        if not HAS_ZSTD:
            raise RuntimeError("zstd codec was used but 'zstandard' package is not available")
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    elif codec == 'zlib':
        return zlib.decompress(data)
    else:
        raise ValueError(f"Unknown codec: {codec}")
