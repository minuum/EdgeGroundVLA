# MoNaVLA Project Memory Snapshot (2026-04-21)

## 1. Project Context

- Project: `MoNaVLA`
- Current workspace on this server: `/home/soda/MoNaVLA`
- Historical Billy path seen in older notes: `/home/billy/25-1kp/MoNaVLA`
- Primary long-term memory file in this workspace: `.menemory/core/master_memory.md`
- This snapshot is a transfer-oriented companion file for handing context to another AI agent or server.

## 2. Key Findings

### BBox limitation is already an evidence-backed conclusion

- The team has already treated BBox-only control as a limiting approach, not an open question.
- Observed failure modes:
  - When the target is too close or too far, detections become unstable or ghost-like.
  - Grid/token quantization such as `32 x 32` discretization reduces steering precision.
  - Predicted text-space coordinates do not map reliably to physical control actions.

### Current direction

- Do not default back to "just improve BBox prediction a bit more" without strong new evidence.
- Preferred next-paradigm candidates:
  - Direct action regression
  - Keypoint-based control
- Important evaluation axes:
  - Driving success rate
  - Trajectory consistency
  - PMDM-style batch analysis over episodes

## 3. Experimental History To Preserve

- Exp10 through Exp14 established the practical ceiling of BBox-based control for this project setting.
- The conclusion was based on repeated rollout behavior and episode-batch analysis, not a single anecdotal failure.
- The value of this memory is to skip repeated dead-end experimentation and continue from the post-BBox decision point.

## 4. Agent Handoff Prompt

Use this as the first instruction when handing the project to another AI agent:

> You are continuing the MoNaVLA project from an existing long-running research context. Read `.menemory/master_snapshot.md` and `.menemory/core/master_memory.md` first, then treat the BBox-only approach as an experimentally bounded baseline rather than the default direction. Preserve past findings, avoid re-running the same dead-end reasoning, and continue from the current transition toward direct action or keypoint-based control.

## 5. One-Line Compressed Handoff

> MoNaVLA has already validated the practical limits of BBox-based driving due to detection instability, quantization error, and weak action alignment, so continue from the current transition toward direct action or keypoint-based control rather than revisiting BBox as the main path.

## 6. Operational Notes

- If another server uses `codex-w` or a similar wrapper, point it at this repository and have it read `.menemory/master_snapshot.md` plus `.menemory/core/master_memory.md` at session start.
- Keep secrets out of this file. API keys and server credentials should be registered through environment variables or local secret files that remain gitignored.
