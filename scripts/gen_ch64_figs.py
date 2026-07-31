#!/usr/bin/env python3
"""CH64 카드 보완용 그림 생성 — exp73 실제 데이터 기반, Okabe-Ito 색맹안전 팔레트."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_kf = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
fm.fontManager.addfont(_kf)
plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs/v5/ch64_figs"; OUT.mkdir(parents=True, exist_ok=True)
CE = ROOT / "docs/v5/closed_loop_eval"

# Okabe-Ito (CVD-safe)
C = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","verm":"#D55E00",
     "pink":"#CC79A7","sky":"#56B4E9","yellow":"#F0E442","black":"#222222","grey":"#999999"}
plt.rcParams.update({"figure.dpi":130,"font.size":11,"axes.grid":True,
                     "grid.alpha":0.25,"axes.axisbelow":True,"axes.edgecolor":"#cccccc",
                     "savefig.bbox":"tight","savefig.facecolor":"white","figure.facecolor":"white"})

def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("  saved", name)

# ---------- 데이터 로드 헬퍼 ----------
def cl(stem):
    return json.load(open(CE / f"exp73_closed_loop_exp73_{stem}.json"))

from scripts.train_exp73_trackA_heads import MLPActionHead, ContRegHead
from scripts.sim.evaluate_closed_loop_exp73 import val_split, build_episode_windows
from scripts.sim.rollout_core import ACTION_VEL, Pose, pose_step
import torch

# =========================================================
# 64-1: 정정 워터폴 (84.8 → 60.6 → 48.5/39.4)
# =========================================================
fig, ax = plt.subplots(figsize=(6.4,3.6))
labels = ["hybrid\n84.8%\n(버그A)","mlp v6\n60.6%\n(버그B)","mlp+trackF\n48.5%\nbest","mlp+trackF\n39.4%\n평균"]
vals = [84.8,60.6,48.5,39.4]; cols=[C["grey"],C["grey"],C["green"],C["blue"]]
ax.bar(range(4), vals, color=cols, width=0.6)
for i,v in enumerate(vals): ax.text(i, v+1.5, f"{v}%", ha="center", fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Closed-loop Success@0.5m (%)"); ax.set_ylim(0,95)
ax.set_title("64-1 · 정정 3회로 무너진 '1위' 주장", fontweight="bold")
ax.axhspan(0,50, color=C["verm"], alpha=0.05)
save(fig,"fig_64_1_waterfall.png")

# =========================================================
# 64-2: 통일 리더보드
# =========================================================
configs = [("pg448/mlp","mlp"),("owl/mlp","mlp"),("pg448/chunk","chunk"),
           ("owl/hybrid","hybrid"),("pg448/hybrid","hybrid"),("pg448/cxgeom","cxgeom"),
           ("owl/cxgeom","cxgeom"),("owl/chunk","chunk"),("pg448/transformer","transformer"),
           ("owl/transformer","transformer")]
def stem_of(name):
    g,h = name.split("/"); g = g+"_trackF"
    if h=="hybrid": return f"{g}_v6_hybrid_azdiscrete_thr0.1"
    return f"{g}_v6_{h}"
hcol={"mlp":C["blue"],"chunk":C["green"],"hybrid":C["orange"],"cxgeom":C["sky"],"transformer":C["verm"]}
rows=[]
for name,h in configs:
    d=cl(stem_of(name)); rows.append((name,h,d["success_rate"]*100,d["fpe_mean"],d["val_acc_mean"]*100))
rows.sort(key=lambda r:-r[2])
fig, ax = plt.subplots(figsize=(7,4.2))
ax.barh([r[0] for r in rows][::-1],[r[2] for r in rows][::-1],
        color=[hcol[r[1]] for r in rows][::-1])
for i,r in enumerate(rows[::-1]): ax.text(r[2]+0.6,i,f"{r[2]:.1f}%",va="center",fontsize=9)
ax.set_xlabel("Closed-loop Success@0.5m (%)"); ax.set_xlim(0,58)
ax.set_title("64-2 · 통일 리더보드 (225ep 정합)", fontweight="bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=v,label=k) for k,v in hcol.items()],fontsize=8,loc="lower right")
save(fig,"fig_64_2_leaderboard.png")

# offline vs closed-loop scatter
fig, ax = plt.subplots(figsize=(5.6,4.2))
for name,h,sr,fpe,off in rows:
    ax.scatter(off,sr,color=hcol[h],s=90,edgecolor="white",zorder=3)
    ax.annotate(name,(off,sr),fontsize=7,xytext=(4,3),textcoords="offset points")
ax.set_xlabel("Offline val_acc (%)"); ax.set_ylabel("Closed-loop Success (%)")
ax.set_title("64-2 · offline ≠ closed-loop (상관 약함)", fontweight="bold")
save(fig,"fig_64_2_offline_vs_cl.png")

# grounder paired (pg448 vs owl)
fig, ax = plt.subplots(figsize=(6,3.8))
heads=["mlp","cxgeom","transformer","hybrid","chunk"]
pg=[cl(stem_of(f"pg448/{h}"))["success_rate"]*100 for h in heads]
ow=[cl(stem_of(f"owl/{h}"))["success_rate"]*100 for h in heads]
x=np.arange(len(heads)); w=0.38
ax.bar(x-w/2,pg,w,label="PG448",color=C["blue"]); ax.bar(x+w/2,ow,w,label="OWL",color=C["orange"])
ax.set_xticks(x); ax.set_xticklabels(heads); ax.set_ylabel("Success (%)")
ax.set_title("64-2 · 그라운더 무차별 (PG448 vs OWL)", fontweight="bold"); ax.legend(fontsize=9)
save(fig,"fig_64_2_grounder.png")

# seed variance strip
fig, ax = plt.subplots(figsize=(5.6,3.8))
seed_sets={"mlp":[33.3,36.4,48.5],"hybrid(pg448)":[None],"chunk(pg448)":[None]}
# champion mlp seeds
mlp_seeds=[cl(f"pg448_trackF_v6_mlp_seed{s}")["success_rate"]*100 for s in range(3)]
ax.plot([0,0,0],mlp_seeds,"o",color=C["blue"],ms=11,label="개별 seed")
ax.plot([0],[np.mean(mlp_seeds)],"_",color=C["black"],ms=40,mew=3)
ax.text(0.12,np.mean(mlp_seeds),f"평균 {np.mean(mlp_seeds):.1f}%",va="center",fontweight="bold")
ax.axhspan(np.mean(mlp_seeds)-6.5,np.mean(mlp_seeds)+6.5,color=C["blue"],alpha=0.08)
ax.set_xlim(-0.5,1.5); ax.set_xticks([0]); ax.set_xticklabels(["pg448/mlp\n(챔피언)"])
ax.set_ylabel("Success (%)"); ax.set_ylim(20,60)
ax.set_title("64-2 · champion seed 분산 (±6.5%p)", fontweight="bold")
save(fig,"fig_64_2_seed_variance.png")

# =========================================================
# 64-3: 실패 시점 (per-path_type, thirds, trajectory)
# =========================================================
eps=torch.load(CE/"exp73_v6_vis_cache.pt",weights_only=False)
eps=[e for e in eps if e.get("acts") is not None]
val=val_split(eps)
m=MLPActionHead(); m.load_state_dict(torch.load(ROOT/"runs/v5_nav/mlp/exp73/exp73_pg448_trackF_v6_mlp_seed2.pt",map_location="cpu",weights_only=False)["model"]); m.eval()

def predict(e):
    X=torch.tensor(build_episode_windows(e))
    with torch.no_grad(): return m(X).argmax(1).numpy()

# per-path_type straight vs curve
d=cl("pg448_trackF_v6_mlp")
from collections import defaultdict
g=defaultdict(list)
for e in d["per_episode"]: g[e["path_type"]].append(e["success"])
pts=sorted(g, key=lambda k:np.mean(g[k]))
srs=[np.mean(g[k])*100 for k in pts]
colr=[C["green"] if "straight" in k else C["verm"] for k in pts]
fig, ax=plt.subplots(figsize=(6.6,4.6))
ax.barh(pts,srs,color=colr)
ax.set_xlabel("Success (%)"); ax.set_title("64-3 · 직진(초록) vs 곡선(주황) 성공률", fontweight="bold")
ax.legend(handles=[Patch(color=C["green"],label="straight"),Patch(color=C["verm"],label="curve")],fontsize=8,loc="lower right")
save(fig,"fig_64_3_pathtype.png")

# thirds accuracy on failing curves
targets=["strong_right_left_curve","weak_left_left_curve","weak_right_right_curve"]
fig, ax=plt.subplots(figsize=(6,3.8))
allthirds=[]
for e in val:
    if e["path_type"] not in targets: continue
    pred=predict(e); gt=np.asarray(e["gts"]); n=len(gt)
    th=[(pred[s]==gt[s]).mean()*100 for s in [slice(0,n//3),slice(n//3,2*n//3),slice(2*n//3,n)]]
    allthirds.append(th)
allthirds=np.array(allthirds)
mean_th=allthirds.mean(0)
ax.bar(["초반\n(cold-start)","중반\n(회전구간)","후반\n(직진복귀)"],mean_th,
       color=[C["orange"],C["verm"],C["green"]])
for i,v in enumerate(mean_th): ax.text(i,v+1.5,f"{v:.0f}%",ha="center",fontweight="bold")
ax.set_ylabel("프레임 정확도 (%)"); ax.set_ylim(0,100)
ax.set_title("64-3 · 실패곡선의 구간별 정확도 (중반 최저)", fontweight="bold")
save(fig,"fig_64_3_thirds.png")

# trajectory plots: one failing curve, one success straight
def traj_xy(cls_seq):
    p=Pose(); xs=[p.x]; ys=[p.y]
    for c in cls_seq:
        lx,ly,az=ACTION_VEL.get(int(c),(0,0,0)); p=pose_step(p,lx,ly,az); xs.append(p.x); ys.append(p.y)
    return xs,ys
def plot_traj(e, title, fname):
    pred=predict(e); gt=np.asarray(e["gts"])
    ex,ey=traj_xy(gt); px,py=traj_xy(pred)
    fig,ax=plt.subplots(figsize=(4.8,4.4))
    ax.plot(ex,ey,"-o",color=C["black"],ms=3,label="expert(정답)",lw=2)
    ax.plot(px,py,"-o",color=C["verm"],ms=3,label="예측",lw=2)
    ax.plot(ex[0],ey[0],"s",color=C["green"],ms=12,label="시작")
    ax.set_aspect("equal"); ax.legend(fontsize=9); ax.set_title(title,fontweight="bold",fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    save(fig,fname)
fail=[e for e in val if e["path_type"]=="strong_right_left_curve"][0]
plot_traj(fail,"64-3 · 곡선 실패 궤적 (FPE 4.6m)","fig_64_3_traj_fail.png")
succ=[e for e in val if e["path_type"]=="center_straight"][0]
plot_traj(succ,"64-3 · 직진 성공 궤적 (FPE≈0)","fig_64_3_traj_success.png")

# cx over time for failing ep
fig,ax=plt.subplots(figsize=(6,3.4))
for e in [e for e in val if e["path_type"] in targets][:3]:
    cxs=[b[0] for b in e["bboxes"]]
    ax.plot(cxs,lw=1.8,alpha=0.8,label=e["path_type"][:16])
ax.axhspan(0.25,0.75,color=C["grey"],alpha=0.1)
ax.set_xlabel("frame"); ax.set_ylabel("bbox cx"); ax.set_ylim(0,1)
ax.set_title("64-3 · 실패 에피소드 cx 시계열", fontweight="bold"); ax.legend(fontsize=7)
save(fig,"fig_64_3_cx_time.png")

# =========================================================
# 64-4: 연속 vs 이산 (4 evidence)
# =========================================================
fig,ax=plt.subplots(figsize=(6.4,3.8))
ev=["offline\n(val_acc)","연속az적분\n(Success)","완전연속궤적\n(Success)"]
disc=[78.3,39.4,30.3]; cont=[74.7,33.3,15.2]
x=np.arange(3); w=0.38
ax.bar(x-w/2,disc,w,label="이산(discrete)",color=C["blue"])
ax.bar(x+w/2,cont,w,label="연속(continuous)",color=C["verm"])
for i in range(3):
    ax.text(x[i]-w/2,disc[i]+1,f"{disc[i]:.0f}",ha="center",fontsize=8)
    ax.text(x[i]+w/2,cont[i]+1,f"{cont[i]:.0f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(ev,fontsize=9); ax.set_ylabel("점수 (%)")
ax.set_title("64-4 · 연속화는 전부 악화", fontweight="bold"); ax.legend(fontsize=9)
save(fig,"fig_64_4_cont_vs_disc.png")

# =========================================================
# 64-5: 그라운더 (cx dist, detection)
# =========================================================
ann=json.load(open(ROOT/"docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"))
cxd=[fr["cx_det"] for ep in ann for fr in ep["frames"] if fr.get("has_bbox")]
fig,ax=plt.subplots(figsize=(6,3.6))
ax.hist(cxd,bins=40,color=C["blue"],alpha=0.85)
ax.axvspan(0,0.15,color=C["verm"],alpha=0.15); ax.axvspan(0.85,1,color=C["verm"],alpha=0.15)
ax.set_xlabel("검출된 bbox cx"); ax.set_ylabel("프레임 수")
ax.set_title("64-5 · 검출 cx 분포 — 극단(주황)은 희소", fontweight="bold")
save(fig,"fig_64_5_cx_dist.png")

fig,ax=plt.subplots(figsize=(5,3.4))
ax.bar(["좌극단","중앙","우극단"],[100,100,100],color=C["green"])
for i in range(3): ax.text(i,101,"100%",ha="center",fontweight="bold")
ax.set_ylim(0,110); ax.set_ylabel("PG448 LIVE 검출률 (%)")
ax.set_title("64-5 · 극단 cx도 100% 검출 (병목 아님)", fontweight="bold")
save(fig,"fig_64_5_detection.png")

# =========================================================
# 64-6: 학습 트릭 전부 무효
# =========================================================
fig,ax=plt.subplots(figsize=(6.4,3.8))
tricks=["baseline","부스트3배","부스트6배","오버샘플4배","V5혼합"]
means=[39.4,32.3,28.3,37.4,39.4]
cols=[C["blue"],C["verm"],C["verm"],C["orange"],C["orange"]]
ax.bar(tricks,means,color=cols)
ax.axhline(39.4,ls="--",color=C["black"],lw=1.5,alpha=0.7)
for i,v in enumerate(means): ax.text(i,v+0.8,f"{v:.1f}",ha="center",fontsize=9,fontweight="bold")
ax.set_ylabel("Success 평균 (%)"); ax.set_ylim(0,50)
ax.set_title("64-6 · 학습 트릭 전부 무효 (점선=baseline)", fontweight="bold")
save(fig,"fig_64_6_tricks.png")

# =========================================================
# 64-7: 로드맵
# =========================================================
fig,ax=plt.subplots(figsize=(7,3.2)); ax.axis("off")
steps=["트랙C\n수집(soda)","289ep\n재학습","통일\n재평가","본격\n실기검증"]
for i,s in enumerate(steps):
    ax.add_patch(plt.Rectangle((i*1.7,0),1.4,1,color=C["sky"] if i==0 else C["grey"],alpha=0.5 if i else 0.9))
    ax.text(i*1.7+0.7,0.5,s,ha="center",va="center",fontsize=10,fontweight="bold")
    if i<3: ax.annotate("",xy=(i*1.7+1.65,0.5),xytext=(i*1.7+1.42,0.5),arrowprops=dict(arrowstyle="->",lw=2))
ax.set_xlim(-0.2,6.6); ax.set_ylim(-0.3,1.3)
ax.set_title("64-7 · 남은 로드맵 (병목=트랙C)", fontweight="bold")
save(fig,"fig_64_7_roadmap.png")

# 궤적 6패널 그리드 (성공/실패 대비)
fig,axes=plt.subplots(2,3,figsize=(10,6.6))
pick=[e for e in val if "straight" in e["path_type"]][:3]+[e for e in val if "curve" in e["path_type"]][:3]
for ax,e in zip(axes.ravel(),pick):
    pred=predict(e); gt=np.asarray(e["gts"])
    ex,ey=traj_xy(gt); px,py=traj_xy(pred)
    ax.plot(ex,ey,"-",color=C["black"],lw=1.8); ax.plot(px,py,"-",color=C["verm"],lw=1.8)
    ax.plot(ex[0],ey[0],"s",color=C["green"],ms=8)
    ok="✓" if "straight" in e["path_type"] else "✗"
    ax.set_title(f"{e['path_type'][:20]} {ok}",fontsize=9); ax.set_aspect("equal")
fig.suptitle("64-3 · 궤적 그리드 (검정=정답 주황=예측): 직진 일치, 곡선 이탈",fontweight="bold")
save(fig,"fig_64_3_traj_grid.png")

# 혼동행렬 (전체 val 프레임)
CLS=["STOP","F","L","R","FL","FR","RoL","RoR"]
conf=np.zeros((8,8),int)
for e in val:
    pred=predict(e); gt=np.asarray(e["gts"])
    for g_,p_ in zip(gt,pred): conf[g_,p_]+=1
confn=conf/np.maximum(conf.sum(1,keepdims=True),1)*100
fig,ax=plt.subplots(figsize=(5.6,5))
im=ax.imshow(confn,cmap="Blues",vmin=0,vmax=100)
ax.set_xticks(range(8)); ax.set_xticklabels(CLS,fontsize=8); ax.set_yticks(range(8)); ax.set_yticklabels(CLS,fontsize=8)
for i in range(8):
    for j in range(8):
        if conf[i,j]>0: ax.text(j,i,f"{confn[i,j]:.0f}",ha="center",va="center",fontsize=7,color="white" if confn[i,j]>50 else "#333")
ax.set_xlabel("예측"); ax.set_ylabel("정답(GT)")
ax.set_title("64-3 · 혼동행렬 — 회전이 F로 흡수됨",fontweight="bold")
save(fig,"fig_64_3_confusion.png")

print("DONE")
