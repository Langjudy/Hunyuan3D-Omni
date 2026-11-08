"""把 float32 的 pytorch_model.bin 转成 float16 的 safetensors（推理结果与运行时转换逐位一致）。

用法: ./venv/python.exe tools/convert_fp16.py weights/Hunyuan3D-Omni/model/pytorch_model.bin
输出: 同目录下的 model_fp16.safetensors；原文件保留不动。
"""
import os
import sys
import time

import torch
from safetensors.torch import save_file

src = sys.argv[1]
dst = os.path.join(os.path.dirname(src), 'model_fp16.safetensors')

t = time.time()
# mmap：按需从硬盘读取，不会把 12GB 一次性塞进内存
sd = torch.load(src, map_location='cpu', mmap=True, weights_only=True)
out = {}
for k, v in sd.items():
    v = v.to(torch.float16) if v.is_floating_point() else v
    out[k] = v.contiguous().clone()  # clone：断开共享存储，safetensors 要求
print(f'loaded+cast {len(out)} tensors in {time.time() - t:.1f}s', flush=True)

tmp = dst + '.tmp'
save_file(out, tmp, metadata={'format': 'pt', 'source': os.path.basename(src), 'dtype': 'float16'})
os.replace(tmp, dst)
print(f'saved {dst} ({os.path.getsize(dst) / 1e9:.2f} GB) total {time.time() - t:.1f}s', flush=True)
