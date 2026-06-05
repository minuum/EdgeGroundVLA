import gradio as gr
import os
import sys
import h5py
import numpy as np
import cv2
import re
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
# --- Local Model Setup ---
_project_root = Path(__file__).resolve().parent.parent
VLA_ROOT = str(_project_root)
if VLA_ROOT not in sys.path:
    sys.path.insert(0, VLA_ROOT)
for root_name in ['RoboVLMs', 'third_party/RoboVLMs']:
    p = os.path.join(VLA_ROOT, root_name)
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Dummy/Fallback inference model when not loaded
class DummyModel:
    def predict(self, img_str, instruction):
        return np.array([0.0, 0.0]), 0.0, np.zeros((1, 2))

local_model_instance = None
LOCAL_MODEL_AVAILABLE = False
try:
    from robovlm_nav.serve.inference_server import MobileVLAInference
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    pass

def scan_h5_files(dataset_dir=None):
    if dataset_dir is None:
        dataset_dir = os.path.join(VLA_ROOT, "ROS_action/mobile_vla_dataset_v3")
    if not os.path.exists(dataset_dir):
        return []
    h5_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.h5')]
    return sorted(h5_files)

def scan_checkpoints():
    root = os.path.join(VLA_ROOT, "runs")
    ckpts = []
    if os.path.exists(root):
        for r, d, f in os.walk(root):
            for file in f:
                if file.endswith(('.ckpt', '.pth')):
                    full_p = os.path.join(r, file)
                    parts = full_p.split(os.sep)
                    display_name = f"{parts[-2]}/{file}" if len(parts) >= 2 else file
                    ckpts.append((display_name, full_p))
    return sorted(ckpts)

def scan_configs():
    root = os.path.join(VLA_ROOT, "configs")
    confs = []
    if os.path.exists(root):
        for f in os.listdir(root):
            if f.endswith('.json'):
                confs.append((f, os.path.join(root, f)))
    return sorted(confs)

def load_h5_episode(h5_path):
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'images' in f: images = f['images'][:]
            elif 'observations/images/camera' in f: images = f['observations/images/camera'][:]
            else: return None, None, None

            if 'actions' in f: actions = f['actions'][:, :2]
            else: actions = np.zeros((len(images), 2))
            
            instr = "Navigate to the gray basket"
            if 'language_instruction' in f:
                val = f['language_instruction'][0]
                instr = val.decode('utf-8') if isinstance(val, bytes) else str(val)
                
            return images, actions, instr
    except Exception as e:
        print(f"H5 Error: {e}")
        return None, None, None

def draw_bounding_box_from_text(img_pil, text, grid_size=32):
    pattern = r"<patch_index_(\d{4})>\s*<patch_index_(\d{4})>"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return img_pil
        
    img_cv = np.array(img_pil.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    patch_w, patch_h = w / grid_size, h / grid_size

    for match in matches:
        start_idx = int(match.group(1))
        end_idx = int(match.group(2))
        y1, x1 = (start_idx // grid_size) * patch_h, (start_idx % grid_size) * patch_w
        y2, x2 = ((end_idx // grid_size) + 1) * patch_h, ((end_idx % grid_size) + 1) * patch_w
        cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(img_cv, "Target", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# --- Gradio Callback Logic ---
def load_model(ckpt_path, conf_path):
    global local_model_instance
    if not LOCAL_MODEL_AVAILABLE:
         return "❌ Model inference code not found."
    try:
        local_model_instance = MobileVLAInference(ckpt_path, conf_path, use_quant=False)
        return f"✅ Model Loaded: {os.path.basename(ckpt_path)} (FP16)"
    except Exception as e:
        return f"❌ Load Failed: {str(e)}"

def analyze_frame(h5_path, frame_idx, instr_override):
    images, gt_actions, _ = load_h5_episode(h5_path)
    if images is None or frame_idx >= len(images):
        return None, "Error loading H5", "N/A", "N/A"
    
    img_arr = images[frame_idx]
    img_pil = Image.fromarray(img_arr)
    
    gt_act = gt_actions[frame_idx]
    gt_str = f"GT: [{gt_act[0]:.2f}, {gt_act[1]:.2f}]"
    
    if local_model_instance is not None:
        import base64, io
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        act, lat, chunk = local_model_instance.predict(b64, instr_override)
        pred_str = f"Pred: [{act[0]:.2f}, {act[1]:.2f}] | {lat:.1f}ms"
        
        # Visualize bounding box if model generated patch tokens
        try:
             # Depending on internal model var, look for generated text
             # Fallback context: model_instance.last_text_output isn't directly exposed in V4 standard but can be grabbed if patched.
             text_out = getattr(local_model_instance, 'last_text_output', "")
             if text_out:
                  img_pil = draw_bounding_box_from_text(img_pil, text_out)
             elif hasattr(local_model_instance.model, 'last_text_output'):
                  img_pil = draw_bounding_box_from_text(img_pil, local_model_instance.model.last_text_output)
        except Exception:
             pass
    else:
        pred_str = "Model not loaded"
        
    return img_pil, f"Frame: {frame_idx}/{len(images)}", gt_str, pred_str

def update_max_frames(h5_path):
    images, _, instr = load_h5_episode(h5_path)
    if images is None:
        return gr.update(maximum=0, value=0), ""
    return gr.update(maximum=len(images)-1, value=0), instr

# --- Gradio UI ---
with gr.Blocks(title="MoNaVLA Offline HTML Analyzer") as demo:
    gr.Markdown("# 📊 Offline Dataset (V3) & Model Grounding Analyzer")
    gr.Markdown("이미 수집된 H5 데이터셋을 불러와 해당 이미지의 Target을 모델이 시각적(Bounding Box)으로 잘 추적하고 올바른 제어값을 출력하는지 오프라인으로 테스트하는 웹 페이지입니다.")
    
    with gr.Row():
        with gr.Column(scale=1):
            h5_files = scan_h5_files()
            h5_dropdown = gr.Dropdown(choices=h5_files, label="📂 Select H5 Dataset Episode", value=h5_files[0] if h5_files else None)
            
            ckpts = scan_checkpoints()
            ckpt_dropdown = gr.Dropdown(choices=[c[1] for c in ckpts], label="🧠 Select Model Checkpoint", value=ckpts[0][1] if ckpts else None)
            
            confs = scan_configs()
            conf_dropdown = gr.Dropdown(choices=[c[1] for c in confs], label="⚙️ Select Config (.json)", value=confs[0][1] if confs else None)

            load_btn = gr.Button("Load Model", variant="primary")
            load_status = gr.Textbox(label="Model Status", value="Not Loaded", interactive=False)
            
            instr_box = gr.Textbox(label="Instruction Override", value="Navigate to the gray basket")
            
            frame_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="🎞️ Frame Index", interactive=True)
            
        with gr.Column(scale=2):
            image_out = gr.Image(label="Visual Grounding Result (Bounding Box)", type="pil")
            with gr.Row():
                frame_info = gr.Textbox(label="Frame Info")
                gt_act_out = gr.Textbox(label="Ground Truth Action")
                pred_act_out = gr.Textbox(label="Model Prediction")

    # Events
    load_btn.click(fn=load_model, inputs=[ckpt_dropdown, conf_dropdown], outputs=[load_status])
    h5_dropdown.change(fn=update_max_frames, inputs=[h5_dropdown], outputs=[frame_slider, instr_box])
    
    # Auto-update when slider moves
    frame_slider.change(
        fn=analyze_frame,
        inputs=[h5_dropdown, frame_slider, instr_box],
        outputs=[image_out, frame_info, gt_act_out, pred_act_out]
    )
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866, share=True)
