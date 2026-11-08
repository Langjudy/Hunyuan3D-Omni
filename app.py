"""Hunyuan3D-Omni 图形界面（Gradio）。

一键启动：双击 “启动.bat”，或运行  ./venv/python.exe app.py
所有模型均从 ./weights 读取，不需要联网。
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
WEIGHTS = os.path.join(ROOT, 'weights')

# ---- 离线 / 本地权重环境（必须在导入 transformers、rembg 之前设置）----
os.environ.setdefault('HY3D_DINO_PATH', os.path.join(WEIGHTS, 'dinov2-large'))
os.environ.setdefault('U2NET_HOME', os.path.join(WEIGHTS, 'rembg'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('GRADIO_TEMP_DIR', os.path.join(ROOT, 'outputs', '.gradio_tmp'))

import argparse
import glob
import random
import shutil
import threading
import time
import traceback
import warnings

warnings.filterwarnings('ignore')

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
import trimesh
from PIL import Image

from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
from hy3dshape.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover

MODEL_DIR = os.path.join(WEIGHTS, 'Hunyuan3D-Omni')
OUTPUT_DIR = os.path.join(ROOT, 'outputs')
MESH_EXTS = ['.ply', '.obj', '.glb', '.gltf', '.stl', '.off']
MAX_SEED = 2 ** 31 - 1

POSE_PRESETS = {
    'A 字站姿 (A-pose)': 'demos/pose/a_pose_bone.txt',
    '举手 (Hand-up)': 'demos/pose/handup_pose_bone.txt',
    '仰望天空 (Sky)': 'demos/pose/sky_pose_bone.txt',
}
MODE_KEYS = ['point', 'voxel', 'pose', 'bbox']
MODE_NAMES = {'point': '点云控制', 'voxel': '体素控制', 'pose': '骨骼姿态', 'bbox': '包围盒'}
UP_Y = 'Y 轴向上（Hunyuan / glTF 默认）'
UP_Z = 'Z 轴向上（Blender / MILo / 3DGS 常见）'


# ============================================================================
# 模型加载
# ============================================================================
class Engine:
    def __init__(self):
        self.pipeline = None
        self.rembg_session = None
        self.lock = threading.Lock()

    def load(self):
        if self.pipeline is None:
            model_sub = os.path.join(MODEL_DIR, 'model')
            if not any(os.path.isfile(os.path.join(model_sub, n))
                       for n in ('model_fp16.safetensors', 'pytorch_model.bin')):
                raise gr.Error('未找到主模型权重（weights/Hunyuan3D-Omni/model/model_fp16.safetensors），'
                               '请检查整合包是否完整。')
            print('[Hunyuan3D-Omni] 正在加载模型 ...')
            t = time.time()
            self.pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(MODEL_DIR)
            print(f'[Hunyuan3D-Omni] 模型加载完成，用时 {time.time() - t:.1f}s')
        return self.pipeline

    def remove_bg(self, image: Image.Image) -> Image.Image:
        if self.rembg_session is None:
            from rembg import new_session
            self.rembg_session = new_session('isnet-general-use', providers=['CPUExecutionProvider'])
        from rembg import remove
        return remove(image.convert('RGB'), session=self.rembg_session)


ENGINE = Engine()


# ============================================================================
# 输入处理
# ============================================================================
def has_real_alpha(img: Image.Image) -> bool:
    if img.mode != 'RGBA':
        return False
    a = np.asarray(img.getchannel('A'))
    return (a < 250).mean() > 0.01


def prepare_image(img, auto_rembg: bool) -> Image.Image:
    if img is None:
        raise gr.Error('请先上传一张物体图片。')
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if has_real_alpha(img):
        return img.convert('RGBA')
    if auto_rembg:
        return ENGINE.remove_bg(img)
    return img.convert('RGBA')


def normalize_points(pts: np.ndarray, scale: float) -> np.ndarray:
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    extent = (hi - lo).max()
    return (pts - center) / extent * 2 * scale


def load_geometry(path: str):
    geo = trimesh.load(path, force=None, process=False)
    if isinstance(geo, trimesh.Scene):
        geos = list(geo.dump()) if hasattr(geo, 'dump') else list(geo.geometry.values())
        meshes = [g for g in geos if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if meshes:
            geo = trimesh.util.concatenate(meshes)
        else:
            pts = np.concatenate([np.asarray(g.vertices) for g in geos], 0)
            geo = trimesh.PointCloud(pts)
    return geo


def geometry_points(path: str, up_axis: str, n_points: int, scale: float) -> np.ndarray:
    """读入网格 / 点云，统一到 Y 轴向上并归一化到 [-scale, scale]，采样 n_points 个点。"""
    if path is None:
        raise gr.Error('请上传控制用的网格或点云文件（.ply / .obj / .glb / .stl）。')
    geo = load_geometry(path)
    if isinstance(geo, trimesh.Trimesh) and len(geo.faces) > 0:
        pts, _ = trimesh.sample.sample_surface(geo, n_points)
    else:
        pts = np.asarray(geo.vertices)
        if len(pts) == 0:
            raise gr.Error('文件里没有读到任何点。')
        idx = np.random.choice(len(pts), n_points, replace=len(pts) < n_points)
        pts = pts[idx]
    pts = np.asarray(pts, dtype=np.float64)
    if up_axis == UP_Z:  # Z-up -> Y-up：绕 X 轴旋转 -90°
        pts = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)
    return normalize_points(pts, scale)


def to_tensor(pts: np.ndarray, pipe):
    return torch.as_tensor(np.asarray(pts), dtype=pipe.dtype, device=pipe.device).unsqueeze(0)


# ============================================================================
# 可视化
# ============================================================================
def control_figure(points: np.ndarray, mode: str, title: str = '控制信号预览'):
    fig = go.Figure()
    if points is not None and len(points):
        p = np.asarray(points, dtype=np.float64)
        if mode == 'pose':
            bones = p.reshape(-1, 2, 3)
            xs, ys, zs = [], [], []
            for a, b in bones:
                xs += [a[0], b[0], None]; ys += [a[2], b[2], None]; zs += [a[1], b[1], None]
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines+markers',
                                       line=dict(color='#8b5cf6', width=6),
                                       marker=dict(size=3, color='#22d3ee')))
        elif mode == 'bbox':
            c = p
            edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
            xs, ys, zs = [], [], []
            for i, j in edges:
                xs += [c[i, 0], c[j, 0], None]; ys += [c[i, 2], c[j, 2], None]; zs += [c[i, 1], c[j, 1], None]
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color='#8b5cf6', width=6)))
        else:
            if len(p) > 20000:
                p = p[np.random.choice(len(p), 20000, replace=False)]
            # 传普通 list：plotly 7 会把 numpy 数组编码成二进制，gradio 4 自带的 plotly.js 解不开
            fig.add_trace(go.Scatter3d(
                x=p[:, 0].tolist(), y=p[:, 2].tolist(), z=p[:, 1].tolist(), mode='markers',
                marker=dict(size=2, color=p[:, 1].tolist(), colorscale=[[0, '#6366f1'], [0.5, '#a855f7'], [1, '#22d3ee']],
                            opacity=0.9)))
    axis = dict(range=[-1.05, 1.05], showbackground=False, showgrid=True, gridcolor='rgba(148,163,184,0.18)',
                zeroline=False, showticklabels=False, title='')
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color='#94a3b8'), x=0.02),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode='cube',
                   camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9))),
        margin=dict(l=0, r=0, t=32, b=0), height=360, showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def bbox_corners(dims):
    l, h, w = [d / 2 for d in dims]
    return np.array([[x, y, z] for x in (-l, l) for y in (-h, h) for z in (-w, w)])


# ============================================================================
# 预览（不跑模型，只看控制信号）
# ============================================================================
def preview_control(mode, point_file, point_up, point_n, voxel_file, voxel_up,
                    pose_preset, pose_file, bbox_x, bbox_y, bbox_z):
    try:
        if mode == 'point':
            pts = geometry_points(point_file, point_up, int(point_n), 0.98)
        elif mode == 'voxel':
            pts = geometry_points(voxel_file, voxel_up, 81920, 0.9999)
            vox = np.clip(np.floor((pts + 1) / 2 * 16), 0, 15)
            pts = np.unique(vox, axis=0) / 16 * 2 - 1 + 1 / 16
        elif mode == 'pose':
            pts = np.loadtxt(pose_file if pose_file else POSE_PRESETS[pose_preset]).reshape(-1, 3)
        else:
            pts = bbox_corners([bbox_x, bbox_y, bbox_z])
        return control_figure(pts, mode, f'{MODE_NAMES[mode]} · 输入预览')
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f'读取控制文件失败：{e}')


# ============================================================================
# 生成
# ============================================================================
def generate(image, auto_rembg, mode,
             point_file, point_up, point_n,
             voxel_file, voxel_up,
             pose_preset, pose_file,
             bbox_x, bbox_y, bbox_z,
             steps, guidance, octree_res, seed, randomize_seed, flashvdm,
             remove_floater, max_faces,
             progress=gr.Progress()):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    progress(0.02, desc='准备图片')
    img = prepare_image(image, auto_rembg)

    progress(0.05, desc='加载模型（首次约 1 分钟）')
    with ENGINE.lock:
        pipe = ENGINE.load()

        progress(0.12, desc='解析控制信号')
        kwargs = {}
        if mode == 'point':
            kwargs['point'] = to_tensor(geometry_points(point_file, point_up, int(point_n), 0.98), pipe)
        elif mode == 'voxel':
            kwargs['voxel'] = to_tensor(geometry_points(voxel_file, voxel_up, 81920, 0.9999), pipe)
        elif mode == 'pose':
            bone = np.loadtxt(pose_file if pose_file else POSE_PRESETS[pose_preset])
            kwargs['pose'] = to_tensor(bone, pipe)
        else:
            kwargs['bbox'] = to_tensor(np.array([[bbox_x, bbox_y, bbox_z]]), pipe)

        progress(0.18, desc=f'扩散生成中（{int(steps)} 步）')
        t0 = time.time()
        try:
            result = pipe(
                image=[img], num_inference_steps=int(steps), octree_resolution=int(octree_res),
                mc_level=0, guidance_scale=float(guidance), fast_decode=bool(flashvdm),
                generator=torch.Generator('cuda').manual_seed(seed), **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error('显存不足：请把“网格分辨率”调低到 384 或 256，或开启 FlashVDM。')
        gen_time = time.time() - t0
        mesh = result['shapes'][0][0]
        sampled_point = result['sampled_point'][0].float().cpu().numpy()

    if mesh is None:
        raise gr.Error('没有生成出有效网格，请换一个随机种子或调整参数再试。')

    progress(0.9, desc='网格后处理')
    if remove_floater:
        mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    if max_faces and int(max_faces) > 0 and len(mesh.faces) > int(max_faces):
        mesh = FaceReducer()(mesh, max_facenum=int(max_faces))

    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(OUTPUT_DIR, f'{stamp}_{mode}_{seed}')
    os.makedirs(out_dir, exist_ok=True)
    glb = os.path.join(out_dir, 'mesh.glb')
    obj = os.path.join(out_dir, 'mesh.obj')
    mesh.export(glb)
    mesh.export(obj)
    # 网页预览用减面版本（完整精度的文件照常提供下载），避免浏览器卡顿
    preview = glb
    if len(mesh.faces) > 300000:
        try:
            preview = os.path.join(out_dir, 'preview_300k.glb')
            FaceReducer()(mesh.copy(), max_facenum=300000).export(preview)
        except Exception:
            traceback.print_exc()
            preview = glb
    img.save(os.path.join(out_dir, 'input.png'))
    trimesh.PointCloud(sampled_point.reshape(-1, 3)).export(os.path.join(out_dir, 'control_points.ply'))
    torch.cuda.empty_cache()

    info = (f"✅ 生成完成 · {MODE_NAMES[mode]} · 用时 {gen_time:.1f}s · 种子 {seed}\n"
            f"顶点 {len(mesh.vertices):,} · 面 {len(mesh.faces):,} · 保存在 outputs/{os.path.basename(out_dir)}")
    if mode == 'pose':
        fig_pts = bone.reshape(-1, 3)
    elif mode == 'bbox':
        fig_pts = bbox_corners([bbox_x, bbox_y, bbox_z])
    else:
        fig_pts = sampled_point.reshape(-1, 3)
    fig = control_figure(fig_pts, mode, f'{MODE_NAMES[mode]} · 模型实际使用的控制信号')
    return preview, [glb, obj], img, fig, info, seed


# ============================================================================
# 界面
# ============================================================================
CSS = """
:root { --radius-xl: 16px; }
.gradio-container { max-width: 1480px !important; margin: 0 auto !important; }
#hero { padding: 26px 30px 22px; border-radius: 20px; margin-bottom: 6px;
  background: radial-gradient(1200px 240px at 0% 0%, rgba(99,102,241,.35), transparent 60%),
              radial-gradient(900px 260px at 100% 0%, rgba(34,211,238,.25), transparent 60%),
              linear-gradient(135deg, #0f172a 0%, #1e1b4b 55%, #0f172a 100%);
  border: 1px solid rgba(148,163,184,.18); box-shadow: 0 10px 40px -12px rgba(79,70,229,.45); }
#hero h1 { margin: 0; font-size: 30px; font-weight: 800; letter-spacing: .3px;
  background: linear-gradient(90deg, #e0e7ff, #c4b5fd 40%, #67e8f9); -webkit-background-clip: text;
  background-clip: text; color: transparent; }
#hero p { margin: 8px 0 0; color: #cbd5e1; font-size: 14.5px; }
#hero .chips { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
#hero .chip { font-size: 12.5px; padding: 4px 11px; border-radius: 999px; color: #e0e7ff;
  background: rgba(99,102,241,.18); border: 1px solid rgba(165,180,252,.28); }
.card { border-radius: 16px !important; border: 1px solid var(--border-color-primary) !important;
  box-shadow: 0 4px 24px -14px rgba(15,23,42,.35) !important; padding: 14px !important; }
.section-title { font-weight: 700; font-size: 15px; margin: 2px 0 6px; opacity: .92; }
.hint { font-size: 12.5px; opacity: .72; line-height: 1.55; }
#gen-btn { height: 52px; font-size: 17px !important; font-weight: 700 !important; border-radius: 14px !important;
  background: linear-gradient(90deg, #6366f1, #8b5cf6 55%, #06b6d4) !important; border: none !important;
  color: white !important; box-shadow: 0 10px 28px -10px rgba(99,102,241,.75) !important; }
#gen-btn:hover { filter: brightness(1.08); transform: translateY(-1px); }
#viewer { border-radius: 14px; overflow: hidden; }
#info textarea { font-size: 13.5px !important; }
footer { display: none !important; }
"""

HERO = """
<div id="hero">
  <h1>Hunyuan3D-Omni · 可控 3D 生成工作台</h1>
  <p>上传一张物体图片，再选择一种控制方式（点云 / 体素 / 骨骼 / 包围盒），即可生成干净、封闭的 3D 网格。
     适合用来修复 MILo / 3DGS 等重建得到的粗糙模型。</p>
  <div class="chips">
    <span class="chip">🧊 点云控制 · 修复重建模型</span><span class="chip">🧱 体素控制 · 粗形状引导</span>
    <span class="chip">🦴 骨骼姿态 · 人物摆姿</span><span class="chip">📦 包围盒 · 比例约束</span>
    <span class="chip">⚡ FlashVDM 加速</span><span class="chip">🔒 全离线运行</span>
  </div>
</div>
"""

TIPS = """
**使用小贴士**
- 图片：物体完整、清晰、尽量正面或斜 45°；没有透明背景会自动抠图。
- 修复 MILo 模型：先在 Blender 里删掉地面/底座等多余部分，再用 **点云控制** 上传；Blender 导出的模型请选 **Z 轴向上**。
- 图片视角最好与模型朝向一致（模型正面朝 -Z / 朝向相机）。
- 生成结果是**白模**（只有几何），可在 Blender 中把原模型颜色烘焙到新网格上。
- 显存不够时把“网格分辨率”调到 384 或 256。
"""


def build_ui():
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo, secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate, radius_size=gr.themes.sizes.radius_lg,
        # 只用系统字体：整合包需要完全离线（GoogleFont 在国内会卡住页面加载）
        font=['Segoe UI', 'Microsoft YaHei', 'PingFang SC', 'sans-serif'],
        font_mono=['Consolas', 'monospace'],
    ).set(block_title_text_weight='600', button_large_radius='14px')

    with gr.Blocks(title='Hunyuan3D-Omni 可控 3D 生成', theme=theme, css=CSS) as demo:
        gr.HTML(HERO)
        mode_state = gr.Textbox('point', visible=False)

        with gr.Row(equal_height=False):
            # ---------------- 左侧：输入 ----------------
            with gr.Column(scale=5, min_width=380):
                with gr.Group(elem_classes='card'):
                    gr.HTML('<div class="section-title">① 物体图片</div>')
                    image = gr.Image(label='参考图片', type='pil', image_mode='RGBA', height=300,
                                     sources=['upload', 'clipboard'])
                    auto_rembg = gr.Checkbox(value=True, label='自动去除背景（图片已是透明 PNG 时自动跳过）')

                with gr.Group(elem_classes='card'):
                    gr.HTML('<div class="section-title">② 控制方式</div>')
                    with gr.Tabs() as tabs:
                        with gr.Tab('🧊 点云', id='point') as tab_point:
                            point_file = gr.File(label='网格 / 点云文件', file_types=MESH_EXTS, height=110)
                            point_up = gr.Radio([UP_Y, UP_Z], value=UP_Z, label='模型的“上方向”')
                            point_n = gr.Slider(512, 8192, value=2048, step=256, label='控制点数量',
                                                info='官方训练使用 2048；点越多越贴合原形状，也越吃显存')
                            gr.Markdown('<div class="hint">最适合修复 MILo / 3DGS / 扫描得到的粗糙模型：'
                                        '保留整体比例，重新生成干净表面。</div>')
                        with gr.Tab('🧱 体素', id='voxel') as tab_voxel:
                            voxel_file = gr.File(label='网格 / 点云文件', file_types=MESH_EXTS, height=110)
                            voxel_up = gr.Radio([UP_Y, UP_Z], value=UP_Z, label='模型的“上方向”')
                            gr.Markdown('<div class="hint">会被量化成 16×16×16 的粗体素，只约束大致形状，'
                                        '细节更多由图片决定。原模型很糙时可以试试。</div>')
                        with gr.Tab('🦴 骨骼', id='pose') as tab_pose:
                            pose_preset = gr.Dropdown(list(POSE_PRESETS), value=list(POSE_PRESETS)[0], label='预设姿态')
                            pose_file = gr.File(label='自定义骨骼 .txt（可选，覆盖预设）', file_types=['.txt'], height=90)
                            gr.Markdown('<div class="hint">用于人物/角色。.txt 每行一根骨骼：x1 y1 z1 x2 y2 z2，'
                                        '格式见 demos/pose/README.md。</div>')
                        with gr.Tab('📦 包围盒', id='bbox') as tab_bbox:
                            bbox_x = gr.Slider(0.05, 1.0, value=0.8, step=0.01, label='长 (X)')
                            bbox_y = gr.Slider(0.05, 1.0, value=1.0, step=0.01, label='高 (Y)')
                            bbox_z = gr.Slider(0.05, 1.0, value=0.6, step=0.01, label='宽 (Z)')
                            gr.Markdown('<div class="hint">按比例填写即可，最长边建议设为 1。</div>')
                    preview_btn = gr.Button('👁 预览控制信号', size='sm', variant='secondary')

                with gr.Accordion('⚙️ 高级参数', open=False, elem_classes='card'):
                    with gr.Row():
                        steps = gr.Slider(10, 100, value=50, step=1, label='推理步数')
                        guidance = gr.Slider(1.0, 10.0, value=4.5, step=0.1, label='引导强度 (CFG)')
                    with gr.Row():
                        octree_res = gr.Dropdown([256, 384, 512], value=512, label='网格分辨率',
                                                 info='越高越精细，越吃显存')
                        max_faces = gr.Number(value=0, precision=0, label='最大面数（0 = 不减面）')
                    with gr.Row():
                        seed = gr.Number(value=1234, precision=0, label='随机种子')
                        randomize_seed = gr.Checkbox(value=False, label='每次随机种子')
                    with gr.Row():
                        flashvdm = gr.Checkbox(value=True, label='FlashVDM 加速解码')
                        remove_floater = gr.Checkbox(value=True, label='移除漂浮碎片')

                gen_btn = gr.Button('✨ 开始生成 3D 模型', variant='primary', elem_id='gen-btn')

            # ---------------- 右侧：输出 ----------------
            with gr.Column(scale=7, min_width=480):
                with gr.Group(elem_classes='card'):
                    gr.HTML('<div class="section-title">③ 生成结果</div>')
                    viewer = gr.Model3D(label='3D 预览（可拖拽旋转 / 滚轮缩放）', height=520, elem_id='viewer',
                                        clear_color=[0.06, 0.07, 0.12, 1.0])
                    info = gr.Textbox(label='状态', lines=2, interactive=False, elem_id='info',
                                      value='等待生成 · 首次生成需要加载模型，约 1 分钟')
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=220):
                        processed = gr.Image(label='实际输入图片（去背景后）', height=360, interactive=False)
                    with gr.Column(scale=2, min_width=260):
                        ctrl_plot = gr.Plot(label='控制信号', value=control_figure(None, 'point'))
                files = gr.File(label='下载模型（GLB / OBJ）', file_count='multiple', interactive=False)
                with gr.Accordion('💡 使用小贴士', open=False):
                    gr.Markdown(TIPS)

        # ---------------- 示例 ----------------
        with gr.Group(elem_classes='card'):
            gr.HTML('<div class="section-title">📚 示例（点击即可填入）</div>')
            with gr.Tabs():
                with gr.Tab('点云'):
                    gr.Examples([['demos/point/imgs/008.png', 'demos/point/plys/008.ply', UP_Y],
                                 ['demos/point/imgs/012.png', 'demos/point/plys/012.ply', UP_Y]],
                                inputs=[image, point_file, point_up], examples_per_page=6)
                with gr.Tab('体素'):
                    vox = sorted(glob.glob('demos/voxel/imgs/*.png'))
                    gr.Examples([[p, p.replace('imgs', 'plys').replace('.png', '.ply'), UP_Z] for p in vox],
                                inputs=[image, voxel_file, voxel_up], examples_per_page=6)
                with gr.Tab('骨骼'):
                    gr.Examples([[p, k] for p in sorted(glob.glob('demos/pose/*.png')) for k in list(POSE_PRESETS)[:2]],
                                inputs=[image, pose_preset], examples_per_page=6)
                with gr.Tab('包围盒'):
                    gr.Examples([['demos/bbox/furniture0.png', 0.8, 0.64, 1.0],
                                 ['demos/bbox/jianbihua0.png', 0.96, 1.0, 0.356],
                                 ['demos/bbox/jianying0.png', 0.378, 1.0, 0.389]],
                                inputs=[image, bbox_x, bbox_y, bbox_z], examples_per_page=6)

        gr.HTML('<div class="hint" style="text-align:center;margin-top:8px">'
                'Powered by Tencent Hunyuan3D-Omni · 仅限遵守 Tencent Hunyuan 3D Omni Community License 使用</div>')

        # ---------------- 事件 ----------------
        for tab, key in zip([tab_point, tab_voxel, tab_pose, tab_bbox], MODE_KEYS):
            tab.select(lambda k=key: k, outputs=mode_state)

        ctrl_inputs = [mode_state, point_file, point_up, point_n, voxel_file, voxel_up,
                       pose_preset, pose_file, bbox_x, bbox_y, bbox_z]
        preview_btn.click(preview_control, ctrl_inputs, ctrl_plot, api_name='preview')

        gen_btn.click(
            generate,
            inputs=[image, auto_rembg, mode_state, point_file, point_up, point_n, voxel_file, voxel_up,
                    pose_preset, pose_file, bbox_x, bbox_y, bbox_z,
                    steps, guidance, octree_res, seed, randomize_seed, flashvdm, remove_floater, max_faces],
            outputs=[viewer, files, processed, ctrl_plot, info, seed],
            concurrency_limit=1, api_name='generate',
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description='Hunyuan3D-Omni WebUI')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true', help='生成公网分享链接')
    parser.add_argument('--preload', action='store_true', help='启动时立即加载模型')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('[错误] 未检测到可用的 NVIDIA 显卡 / CUDA，本程序需要 NVIDIA 显卡（建议 12GB 以上显存）。')
        sys.exit(1)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.preload:
        ENGINE.load()

    demo = build_ui()
    demo.queue(max_size=20).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        inbrowser=not args.no_browser, show_api=False,
        allowed_paths=[OUTPUT_DIR, os.path.join(ROOT, 'demos')])


if __name__ == '__main__':
    main()
