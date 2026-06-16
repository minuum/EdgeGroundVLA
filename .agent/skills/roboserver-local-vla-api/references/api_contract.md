# Local RoboServer API Contract

Use this contract when `vla-inference-gradio` or `ros2_client/vla_api_client.py`
needs to call a robot-local inference server.

## Endpoints

- `POST http://localhost:8000/predict`
- `POST http://localhost:8000/reset`

## Headers

```http
Content-Type: application/json
X-API-Key: <VLA_API_KEY>
```

## `/predict` Request

```json
{
  "image": "<base64 encoded image>",
  "instruction": "Navigate toward the gray basket until it gets closer"
}
```

Required fields:

- `image`
- `instruction`

## `/predict` Response

Minimum compatible response:

```json
{
  "action": [1.15, 0.0],
  "latency_ms": 87.3
}
```

Recommended full response:

```json
{
  "action": [1.15, 0.0],
  "latency_ms": 87.3,
  "model_name": "exp19_proxy_local",
  "strategy": "proxy_mlp",
  "source": "inferred",
  "buffer_status": {}
}
```

Compatibility rules:

- `action` must be exactly 2 numbers: `[linear_x, linear_y]`
- Do not overload `action` with rotation or 3D control values
- Additional metadata fields are safe if existing keys remain unchanged
- If 3D output is needed, use a separate field such as `action_3d`

## `/reset` Response

```json
{
  "status": "success",
  "message": "history reset"
}
```

## Practical Guidance

- Keep the API local-first. The expected deployment is the robot server itself
  running the inference API on localhost.
- If the dashboard or client is already configured for another host or port,
  update that setting explicitly. Do not assume discovery.
- If a server change would break this contract, update the client and dashboard in
  the same task or call out the compatibility break clearly.
