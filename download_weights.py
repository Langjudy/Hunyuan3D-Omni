"""把 Hunyuan3D-Omni 需要的全部权重下载到 ./weights（可断点续传，失败自动重试）。

整合包里已经带好了权重，一般不需要运行本脚本；权重缺失或损坏时再用。

用法:
    ./venv/python.exe download_weights.py          # 主模型 + DINOv2 + 抠图模型
    ./venv/python.exe download_weights.py --ema    # 额外下载 EMA 权重（下载 12 GB，转换后 6 GB）
    ./venv/python.exe download_weights.py --mirror # 使用 hf-mirror.com 国内镜像
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(ROOT, 'weights')
if '--mirror' in sys.argv:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['U2NET_HOME'] = os.path.join(WEIGHTS, 'rembg')

from huggingface_hub import snapshot_download  # noqa: E402

OMNI = os.path.join(WEIGHTS, 'Hunyuan3D-Omni')
# 主模型以 float16 safetensors 形式保存（6.1GB，加载快、省内存）；官方 fp32 .bin 下载后自动转换并删除
MAIN = [('model/pytorch_model.bin', 'model/model_fp16.safetensors')]
if '--ema' in sys.argv:
    MAIN.append(('model/pytorch_model_ema.bin', 'model/model_ema_fp16.safetensors'))


def convert_to_fp16(src, dst):
    import torch
    from safetensors.torch import save_file
    sd = torch.load(src, map_location='cpu', mmap=True, weights_only=True)
    out = {k: (v.to(torch.float16) if v.is_floating_point() else v).contiguous().clone() for k, v in sd.items()}
    del sd
    save_file(out, dst + '.tmp', metadata={'format': 'pt', 'dtype': 'float16'})
    os.replace(dst + '.tmp', dst)
    os.remove(src)


def download_all():
    patterns = ['*.json', 'cond_encoder/*', 'vae/*']
    patterns += [b for b, st in MAIN if not os.path.isfile(os.path.join(OMNI, st))]
    snapshot_download('tencent/Hunyuan3D-Omni', local_dir=OMNI, allow_patterns=patterns)
    for b, st in MAIN:
        if os.path.isfile(os.path.join(OMNI, b)) and not os.path.isfile(os.path.join(OMNI, st)):
            print(f'converting {b} -> {st} (float16) ...', flush=True)
            convert_to_fp16(os.path.join(OMNI, b), os.path.join(OMNI, st))
    print('[1/3] Hunyuan3D-Omni done', flush=True)
    snapshot_download('facebook/dinov2-large', local_dir=os.path.join(WEIGHTS, 'dinov2-large'),
                      allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])
    print('[2/3] DINOv2 done', flush=True)
    from rembg import new_session
    new_session('isnet-general-use', providers=['CPUExecutionProvider'])
    print('[3/3] rembg isnet-general-use done', flush=True)


for attempt in range(1, 31):
    try:
        download_all()
        break
    except Exception as e:  # 网络中断等，等待后续传
        print(f'attempt {attempt} failed: {e!r}', flush=True)
        time.sleep(5)
else:
    sys.exit(1)
