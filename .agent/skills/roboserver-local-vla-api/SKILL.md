---
name: roboserver-local-vla-api
description: Use when wiring, documenting, or validating the local Robo server API consumed by vla-inference-gradio or ros2_client/vla_api_client. Defines the required localhost /predict and /reset contract, auth header, and compatibility constraints such as keeping action as 2D [linear_x, linear_y].
---

# RoboServer Local VLA API Skill

## When to Use
- When the user wants to run inference only on the robot server locally
- When `vla-inference-gradio` or `ros2_client/vla_api_client.py` must talk to a localhost API
- When documenting or reviewing `/predict` and `/reset` compatibility

## Source of Truth
- Consumer defaults live in `scripts/gradio_inference_dashboard.py`
- Client request format lives in `ros2_client/vla_api_client.py`
- Exact payload and constraints are in [references/api_contract.md](references/api_contract.md)

Read the reference file before changing a local inference server or handing API requirements to another team.

## Workflow
1. Confirm the target is the robot-local server, not `robovlm_nav/serve/proxy_inference_server.py`.
2. Keep the contract compatible with existing clients unless the user explicitly wants a breaking change.
3. Validate `/predict` request keys, auth header, and response field names against the reference file.
4. If extra metadata is needed, add new fields without changing the shape of `action`.

## Hard Constraints
- `POST /predict` must accept JSON with `image` and `instruction`
- `X-API-Key` header is required
- `action` must remain a 2-element array: `[linear_x, linear_y]`
- `latency_ms` must be present in the response
- Rotation or 3D control must go into a separate field such as `action_3d`

## Notes
- Default local base URL is `http://localhost:8000`
- `/reset` should clear history or episode state and return a simple success payload
