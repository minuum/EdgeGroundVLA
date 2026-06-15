#!/usr/bin/env python3
"""
Exp66 Masking Ablation — base PaliGemma2 grounding (학습 데이터 생성과 동일)

Stage2 v2 학습 시 bbox_cx 생성에 쓴 모델:
  base PaliGemma2-3B-mix-224, 프롬프트 "detect gray basket", <loc> 토큰 파싱

절차:
  1. v5 에피소드 프레임 → PG2 "detect gray basket" → bbox
  2. has_bbox=True + 적절 area 프레임만 선별
  3. Stage2 v2 ActionMLP (bbox_history=zeros, image-only)
  4. basket 영역 gray 마스킹
  5. action flip 측정

Usage:
  .venv/bin/python3 scripts/exp66_masking_ablation_pg2.py
"""
import io, re, sys, warnings, random
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────
PG2_PATH    = Path.home() / ".cache/huggingface/hub" \
              / "models--google--paligemma2-3b-mix-224" \
              / "snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
STAGE1_CKPT = ROOT / "runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt"
STAGE2_CKPT = ROOT / "runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp_base_pg2_aug.pt"
KOS_PATH    = ROOT / ".vlms/kosmos-2-patch14-224"
DATA_DIR    = ROOT / "ROS_action/mobile_vla_dataset_v5"

OUT_DIR     = ROOT / "docs/v5/exp66_masking_viz"
OUT_PNG_EXP = ROOT / "docs/v5/exp54_viz/masking_comparison.png"
OUT_PNG_PORT= ROOT / "docs/v5/portfolio/masking_comparison.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────────
LOC_RE       = re.compile(r"<loc(\d{4})>")
MIN_AREA     = 0.015   # 너무 작으면 마스킹 효과 없음 (PG2는 감도 높음)
MAX_AREA     = 0.65    # 도착 직전 프레임 제외
MASK_SCALE   = 1.5
MASK_COLOR   = (128, 128, 128)

VIS_DIM  = 1024
PROJ_DIM = 256
BBOX_DIM = 32
N_CLS    = 8
ACTION_NAMES = ["STOP","FWD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
ACTION_COLORS_HEX = [
    "#ef4444","#4ade80","#60a5fa","#f59e0b",
    "#a78bfa","#fb923c","#38bdf8","#818cf8",
]

PATH_TYPES = [
    "center_straight","center_left","center_right",
    "left_straight","left_right",
    "right_straight","right_left",
]
FRAMES_PER_EP = 5
EPS_PER_TYPE  = 3
FRAME_PHASE   = 0.65   # 에피소드 앞 65% 프레임만 (도착 직전 제외)


# ── PG2 grounding ────────────────────────────────────────────────────────
def load_pg2(device):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    dtype = torch.bfloat16
    print("[PG2] base PaliGemma2-3B-mix-224 로딩 ...")
    proc  = PaliGemmaProcessor.from_pretrained(str(PG2_PATH))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2_PATH), torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    return proc, model, dtype


@torch.no_grad()
def run_pg2_grounding(pg_model, pg_proc, img_pil: Image.Image,
                      device, dtype) -> Optional[dict]:
    """PG2 detect gray basket → bbox dict or None."""
    inp = pg_proc(text="detect gray basket", images=img_pil,
                  return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    gen = pg_model.generate(**inp, max_new_tokens=48, do_sample=False)
    raw = pg_proc.batch_decode(
        gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False
    )[0]
    locs = [int(v) / 1023.0 for v in LOC_RE.findall(raw)]
    if len(locs) < 4:
        return None
    y1, x1, y2, x2 = locs[:4]
    # 이미지 좌표 바운드 체크
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(1,x2), min(1,y2)
    area = (x2-x1)*(y2-y1)
    if area <= 0:
        return None
    return {"cx":(x1+x2)/2,"cy":(y1+y2)/2,"area":area,
            "x1":x1,"y1":y1,"x2":x2,"y2":y2,"raw":raw}


# ── Kosmos-2 encoder + Stage2 ActionMLP ──────────────────────────────────
def load_action_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("[Kosmos-2] vision encoder + image_proj ...")
    s1 = torch.load(str(STAGE1_CKPT), map_location=device, weights_only=False)
    processor = AutoProcessor.from_pretrained(str(KOS_PATH))
    base = AutoModelForVision2Seq.from_pretrained(str(KOS_PATH), torch_dtype=torch.float16)
    vm = base.vision_model.to(device).eval()

    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    image_proj.load_state_dict(s1["image_proj"])
    image_proj.eval()

    print("[Stage2] ActionMLP ...")
    s2 = torch.load(str(STAGE2_CKPT), map_location=device, weights_only=False)

    class ActionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(PROJ_DIM+BBOX_DIM,256),nn.ReLU(),nn.Dropout(0.25),
                nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),
                nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1),
                nn.Linear(64,N_CLS),
            )
        def forward(self,x): return self.net(x)

    mlp = ActionMLP().to(device)
    mlp.load_state_dict(s2["mlp"])
    mlp.eval()
    print(f"  val_acc={s2['val_acc']:.4f}")
    return processor, vm, image_proj, mlp


@torch.no_grad()
def predict(vm, image_proj, mlp, kos_proc, img_pil: Image.Image, device):
    inp = kos_proc(images=[img_pil], return_tensors="pt")
    pv  = inp["pixel_values"].to(device, dtype=torch.float16)
    vis = vm(pixel_values=pv).last_hidden_state.mean(dim=1).float()
    proj= F.normalize(image_proj(vis), dim=-1)
    feat= torch.cat([proj, torch.zeros(1, BBOX_DIM, device=device)], dim=-1)
    logits = mlp(feat)
    probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return int(probs.argmax()), probs


def mask_basket(img_pil: Image.Image, x1, y1, x2, y2) -> Image.Image:
    W, H = img_pil.size
    cx, cy = (x1+x2)/2, (y1+y2)/2
    hw = (x2-x1)*MASK_SCALE/2;  hh = (y2-y1)*MASK_SCALE/2
    px0=max(0,int((cx-hw)*W)); py0=max(0,int((cy-hh)*H))
    px1=min(W,int((cx+hw)*W)); py1=min(H,int((cy+hh)*H))
    m = img_pil.copy()
    ImageDraw.Draw(m).rectangle([px0,py0,px1,py1], fill=MASK_COLOR)
    return m


def load_frame(ep_path, idx):
    with h5py.File(str(ep_path),"r") as f:
        raw = f["observations"]["images"][idx]
    if isinstance(raw,(bytes,np.bytes_)):
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    arr = np.array(raw)
    if arr.dtype!=np.uint8: arr=(arr*255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def get_pt(name):
    for pt in PATH_TYPES:
        if pt in name: return pt
    return "other"


# ── Visualization ─────────────────────────────────────────────────────────
BG=(15,23,42); FG=(226,232,240)
RED=(239,68,68); GREEN=(74,222,128); YELLOW=(251,191,36)
CELL_W, CELL_H = 300, 190

def hex2rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))

def make_grid(results):
    cols=3
    rows=(len(results)+cols-1)//cols
    STAT_H=82
    W=cols*(CELL_W*2+32)+20
    H=STAT_H+rows*(CELL_H+52)+10
    canvas=Image.new("RGB",(W,H),BG)
    draw=ImageDraw.Draw(canvas)
    try:
        f14=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
        f11=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",11)
        f9 =ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",9)
    except:
        f14=f11=f9=ImageFont.load_default()

    flips=sum(1 for r in results if r["changed"]); total=len(results)
    flip_pc=100*flips/max(total,1)

    draw.text((12,10),"Exp66 Masking Ablation — base PaliGemma2 Grounding (학습 데이터 생성 동일 모델)",fill=FG,font=f11)
    draw.text((12,26),"prompt: 'detect gray basket'  |  bbox_history=zeros  |  Stage2 v2 (val_acc 93.5%)",fill=(100,116,139),font=f9)
    col=GREEN if flip_pc>=60 else YELLOW
    draw.text((12,44),f"Action Flip: {flips}/{total}  ({flip_pc:.0f}%)",fill=col,font=f14)
    draw.text((300,48),"Stage2 v2 CL 96.6% · FPE 0.102m · SOTA",fill=(100,116,139),font=f9)

    for idx,r in enumerate(results):
        ci=idx%cols; ri=idx//cols
        ox=10+ci*(CELL_W*2+32)
        oy=STAT_H+ri*(CELL_H+52)

        # original + grounding bbox 표시
        orig_r=r["orig_img"].resize((CELL_W,CELL_H))
        canvas.paste(orig_r,(ox,oy))
        W_,H_=r["orig_img"].size
        bx0=int(r["x1"]*CELL_W); by0=int(r["y1"]*CELL_H)
        bx1=int(r["x2"]*CELL_W); by1=int(r["y2"]*CELL_H)
        draw.rectangle([ox+bx0,oy+by0,ox+bx1,oy+by1],outline=(255,80,80),width=2)
        draw.text((ox+bx0+2,oy+by0+2),"PG2",fill=(255,80,80),font=f9)

        # arrow
        ax=ox+CELL_W+1
        draw.rectangle([ax,oy,ax+28,oy+CELL_H],fill=BG)
        arrow_col=RED if r["changed"] else GREEN
        draw.text((ax+4,oy+CELL_H//2-10),"→",fill=arrow_col,font=f14)

        # masked
        mx=ax+29
        mask_r=r["mask_img"].resize((CELL_W,CELL_H))
        canvas.paste(mask_r,(mx,oy))
        border=RED if r["changed"] else (30,41,59)
        draw.rectangle([mx,oy,mx+CELL_W,oy+CELL_H],outline=border,width=2)
        if r["changed"]:
            draw.rectangle([mx,oy,mx+46,oy+18],fill=RED)
            draw.text((mx+4,oy+3),"FLIP",fill=(255,255,255),font=f9)

        # labels
        ly=oy+CELL_H+3
        po=r["pred_orig"]; pm=r["pred_mask"]
        draw.text((ox,ly),f"{r['path_type']}  area={r['area']:.2f}",fill=(100,116,139),font=f9)
        aco=hex2rgb(ACTION_COLORS_HEX[po]); acm=hex2rgb(ACTION_COLORS_HEX[pm])
        draw.rectangle([ox,ly+12,ox+55,ly+24],fill=aco)
        draw.text((ox+3,ly+13),ACTION_NAMES[po],fill=BG,font=f9)
        draw.text((ox+59,ly+13),"→",fill=(100,116,139),font=f9)
        draw.rectangle([ox+72,ly+12,ox+127,ly+24],fill=acm)
        draw.text((ox+75,ly+13),ACTION_NAMES[pm],fill=BG,font=f9)

    return canvas


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pg_proc, pg_model, pg_dtype = load_pg2(device)
    kos_proc, vm, image_proj, mlp = load_action_model(device)

    all_eps=sorted(DATA_DIR.glob("*.h5"))
    by_type={pt:[] for pt in PATH_TYPES}
    for ep in all_eps:
        pt=get_pt(ep.name)
        if pt in by_type: by_type[pt].append(ep)

    random.seed(42)
    results=[]; tested=0; found=0

    for pt, eps in by_type.items():
        random.shuffle(eps)
        ep_used=0
        for ep_path in eps:
            if ep_used>=EPS_PER_TYPE: break
            try:
                with h5py.File(str(ep_path),"r") as f:
                    n=len(f["observations"]["images"])
            except: continue

            pool=list(range(max(1,int(n*FRAME_PHASE))))
            random.shuffle(pool)
            frame_found=0

            for fidx in pool:
                if frame_found>=FRAMES_PER_EP: break
                tested+=1
                try: img=load_frame(ep_path,fidx)
                except: continue

                bbox=run_pg2_grounding(pg_model,pg_proc,img,device,pg_dtype)
                if bbox is None:
                    print(f"  [{pt}] f{fidx:2d} NO BBOX")
                    continue
                if not (MIN_AREA<=bbox["area"]<=MAX_AREA):
                    print(f"  [{pt}] f{fidx:2d} area={bbox['area']:.3f} 범위 밖  cx={bbox['cx']:.2f}")
                    continue

                pred_o,_=predict(vm,image_proj,mlp,kos_proc,img,device)
                masked=mask_basket(img,bbox["x1"],bbox["y1"],bbox["x2"],bbox["y2"])
                pred_m,_=predict(vm,image_proj,mlp,kos_proc,masked,device)

                changed=pred_o!=pred_m
                found+=1; frame_found+=1
                status="FLIP ✓" if changed else "same "
                print(f"  [{pt}] f{fidx:2d} area={bbox['area']:.2f} cx={bbox['cx']:.2f}"
                      f"  {ACTION_NAMES[pred_o]:8s}→{ACTION_NAMES[pred_m]:8s}  {status}")

                results.append({
                    "orig_img":img,"mask_img":masked,
                    "pred_orig":pred_o,"pred_mask":pred_m,
                    "changed":changed,"path_type":pt,
                    "area":bbox["area"],
                    "cx":bbox["cx"],"cy":bbox["cy"],
                    "x1":bbox["x1"],"y1":bbox["y1"],
                    "x2":bbox["x2"],"y2":bbox["y2"],
                })

            if frame_found>0: ep_used+=1

    print(f"\n── Summary ──────────────────────────────")
    print(f"  Tested        : {tested}")
    print(f"  has_bbox=True : {found}")
    flips=sum(1 for r in results if r["changed"])
    print(f"  Action flip   : {flips}/{found}  ({100*flips/max(found,1):.1f}%)")

    if not results:
        print("결과 없음."); return

    grid=make_grid(results)
    grid.save(str(OUT_DIR/"masking_comparison_pg2.png"))
    grid.save(str(OUT_PNG_EXP))
    grid.save(str(OUT_PNG_PORT))
    print(f"\nSaved → {OUT_PNG_EXP}")
    print(f"         {OUT_PNG_PORT}")


if __name__=="__main__":
    main()
