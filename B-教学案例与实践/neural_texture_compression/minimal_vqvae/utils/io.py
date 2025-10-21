import io
import json
import struct
import base64
from typing import Optional, Tuple

import numpy as np
import torch

MAGIC = b"NTEX1"


def _state_dict_to_bytes(state_dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def _bytes_to_state_dict(data: bytes):
    buf = io.BytesIO(data)
    state = torch.load(buf, map_location='cpu')
    return state


def _array_to_npz_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, embeddings=arr)
    return buf.getvalue()


def _npz_bytes_to_array(data: bytes) -> np.ndarray:
    buf = io.BytesIO(data)
    z = np.load(buf)
    return z['embeddings']


def write_ntex(path: str, meta: dict, indices_bytes: bytes,
               model_state_dict: Optional[dict] = None,
               codebook_array: Optional[np.ndarray] = None,
               codec_name: str = 'zlib'):
    header = {
        'meta': meta,
        'codec': codec_name,
        'attachments': {}
    }
    if model_state_dict is not None:
        m_bytes = _state_dict_to_bytes(model_state_dict)
        header['attachments']['model_state_b64'] = base64.b64encode(m_bytes).decode('ascii')
    if codebook_array is not None:
        c_bytes = _array_to_npz_bytes(codebook_array)
        header['attachments']['codebook_b64'] = base64.b64encode(c_bytes).decode('ascii')

    header_bytes = json.dumps(header).encode('utf-8')
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<Q', len(header_bytes)))
        f.write(header_bytes)
        f.write(indices_bytes)


def read_ntex(path: str):
    with open(path, 'rb') as f:
        magic = f.read(len(MAGIC))
        if magic != MAGIC:
            raise RuntimeError("Invalid NTEX file magic")
        (hlen,) = struct.unpack('<Q', f.read(8))
        hbytes = f.read(hlen)
        header = json.loads(hbytes.decode('utf-8'))
        data = f.read()

    attachments = header.get('attachments', {})
    model_state = None
    codebook_array = None
    if 'model_state_b64' in attachments:
        m_bytes = base64.b64decode(attachments['model_state_b64'])
        model_state = _bytes_to_state_dict(m_bytes)
    if 'codebook_b64' in attachments:
        c_bytes = base64.b64decode(attachments['codebook_b64'])
        codebook_array = _npz_bytes_to_array(c_bytes)

    return header.get('meta', {}), header.get('codec', 'zlib'), data, model_state, codebook_array
