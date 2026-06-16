#!/usr/bin/env python3
import argparse
import json
import os
import socket
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.json"
DEFAULT_PROFILE = os.getenv("VLA_PROFILE", "end_to_end_default")


def load_registry() -> dict:
    with REGISTRY_PATH.open("r") as f:
        return json.load(f)


def expand_path(raw: str) -> str:
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def path_exists(raw: str) -> bool:
    if not raw:
        return False
    return Path(expand_path(raw)).exists()


def model_candidates(registry: dict, profile_name: str) -> list[str]:
    profiles = registry["profiles"]
    if profile_name not in profiles:
        raise KeyError(f"Unknown profile: {profile_name}")
    profile = profiles[profile_name]
    return [profile["default_model"], *profile.get("fallback_models", [])]


def resolve_profile(registry: dict, profile_name: str) -> dict:
    profiles = registry["profiles"]
    models = registry["models"]
    profile = profiles[profile_name]
    reasons = []

    for model_id in model_candidates(registry, profile_name):
        model = models[model_id]
        checkpoint_path = expand_path(model.get("checkpoint", ""))
        config_path = expand_path(model.get("config", ""))
        checkpoint_ok = bool(model.get("checkpoint")) and Path(checkpoint_path).exists()
        config_ok = bool(model.get("config")) and Path(config_path).exists()
        runtime = model.get("runtime", "")
        launchable = runtime in profile.get("allow_runtimes", [])

        reasons.append(
            {
                "model_id": model_id,
                "runtime": runtime,
                "checkpoint_exists": checkpoint_ok,
                "config_exists": config_ok,
                "launchable": launchable,
            }
        )

        if runtime == "mlp_step2":
            return {
                "profile": profile_name,
                "profile_label": profile["label"],
                "model_id": model_id,
                "model": model,
                "checkpoint_path": checkpoint_path,
                "config_path": config_path,
                "runtime": runtime,
                "status": model.get("status", "unknown"),
                "fallback_trace": reasons,
                "launchable": False,
            }

        # proxy_server: MLP 체크포인트만 있으면 실행 가능, config 불필요
        config_required = runtime != "proxy_server"
        if checkpoint_ok and (config_ok or not config_required) and launchable:
            return {
                "profile": profile_name,
                "profile_label": profile["label"],
                "model_id": model_id,
                "model": model,
                "checkpoint_path": checkpoint_path,
                "config_path": config_path,
                "runtime": runtime,
                "status": model.get("status", "unknown"),
                "fallback_trace": reasons,
                "launchable": True,
            }

    last_model_id = model_candidates(registry, profile_name)[-1]
    last_model = models[last_model_id]
    return {
        "profile": profile_name,
        "profile_label": profile["label"],
        "model_id": last_model_id,
        "model": last_model,
        "checkpoint_path": expand_path(last_model.get("checkpoint", "")),
        "config_path": expand_path(last_model.get("config", "")),
        "runtime": last_model.get("runtime", ""),
        "status": last_model.get("status", "unknown"),
        "fallback_trace": reasons,
        "launchable": False,
    }


def shell_exports(resolved: dict) -> str:
    model = resolved["model"]
    lines = [
        f'export VLA_PROFILE="{resolved["profile"]}"',
        f'export VLA_PROFILE_LABEL="{resolved["profile_label"]}"',
        f'export VLA_MODEL_ID="{resolved["model_id"]}"',
        f'export VLA_MODEL_LABEL="{model.get("label", resolved["model_id"])}"',
        f'export VLA_MODEL_KIND="{model.get("kind", "")}"',
        f'export VLA_MODEL_RUNTIME="{resolved["runtime"]}"',
        f'export VLA_MODEL_STATUS="{resolved["status"]}"',
    ]
    if resolved["checkpoint_path"]:
        lines.append(f'export VLA_CHECKPOINT_PATH="{resolved["checkpoint_path"]}"')
    if resolved["config_path"]:
        lines.append(f'export VLA_CONFIG_PATH="{resolved["config_path"]}"')
    if resolved["runtime"] == "proxy_server" and resolved["checkpoint_path"]:
        lines.append(f'export VLA_PROXY_WEIGHTS_PATH="{resolved["checkpoint_path"]}"')
        lines.append(f'export VLA_SERVER_SCRIPT="robovlm_nav/serve/proxy_inference_server.py"')
    return "\n".join(lines)


def print_profile_summary(resolved: dict) -> None:
    model = resolved["model"]
    print(f'Profile: {resolved["profile"]} ({resolved["profile_label"]})')
    print(f'Model:   {resolved["model_id"]} - {model.get("label", "")}')
    print(f'Kind:    {model.get("kind", "")}')
    print(f'Runtime: {resolved["runtime"]}')
    print(f'Status:  {resolved["status"]}')
    print(f'Launch:  {"yes" if resolved["launchable"] else "no"}')
    print(f'CKPT:    {resolved["checkpoint_path"] or "N/A"}')
    print(f'Config:  {resolved["config_path"] or "N/A"}')
    evidence = ", ".join(model.get("evidence", []))
    print(f'Evidence:{(" " + evidence) if evidence else " N/A"}')


def doctor(registry: dict, profile_name: str) -> int:
    resolved = resolve_profile(registry, profile_name)
    print_profile_summary(resolved)
    print("")
    print("Checks:")
    print(f'- Registry: {"OK" if REGISTRY_PATH.exists() else "MISSING"} ({REGISTRY_PATH})')
    print(f'- Checkpoint exists: {"OK" if path_exists(resolved["model"].get("checkpoint", "")) else "MISSING"}')
    print(f'- Config exists: {"OK" if path_exists(resolved["model"].get("config", "")) else "MISSING"}')
    ros_ws = Path(os.getenv("VLA_ROS_WS", str(PROJECT_ROOT / "ROS_action")))
    print(f'- ROS workspace: {"OK" if ros_ws.exists() else "MISSING"} ({ros_ws})')
    camera_pkg = ros_ws / "install" / "camera_interfaces"
    print(f'- camera_interfaces install: {"OK" if camera_pkg.exists() else "MISSING"}')
    print(f'- API URL: {os.getenv("VLA_API_SERVER", "http://localhost:8000")}')
    try:
        socket.gethostbyname("localhost")
        print("- Local hostname resolution: OK")
    except OSError:
        print("- Local hostname resolution: FAIL")
    if resolved["runtime"] == "mlp_step2":
        print("- Dedicated decomposition runtime: NOT IMPLEMENTED IN LAUNCHER")
    return 0


def cmd_models(registry: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps(registry, indent=2))
        return 0
    for model_id, model in registry["models"].items():
        ckpt = expand_path(model.get("checkpoint", ""))
        cfg = expand_path(model.get("config", ""))
        print(
            f"{model_id:28} | {model.get('kind',''):13} | {model.get('runtime',''):10} | "
            f"{model.get('status',''):16} | ckpt={'Y' if Path(ckpt).exists() else 'N'} | "
            f"cfg={'Y' if Path(cfg).exists() else 'N'}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MoNaVLA profile resolver")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_profile = sub.add_parser("profile")
    p_profile.add_argument("--profile", default=DEFAULT_PROFILE)
    p_profile.add_argument("--json", action="store_true")
    p_profile.add_argument("--shell", action="store_true")

    p_models = sub.add_parser("models")
    p_models.add_argument("--json", action="store_true")

    p_doctor = sub.add_parser("doctor")
    p_doctor.add_argument("--profile", default=DEFAULT_PROFILE)

    p_env = sub.add_parser("env")
    p_env.add_argument("--profile", default=DEFAULT_PROFILE)

    args = parser.parse_args()
    registry = load_registry()

    if args.cmd == "models":
        return cmd_models(registry, args.json)

    if args.cmd in {"profile", "env", "doctor"}:
        resolved = resolve_profile(registry, args.profile)

    if args.cmd == "env":
        print(shell_exports(resolved))
        return 0
    if args.cmd == "doctor":
        return doctor(registry, args.profile)
    if args.cmd == "profile":
        if args.shell:
            print(shell_exports(resolved))
        elif args.json:
            payload = dict(resolved)
            payload["model"] = {
                **payload["model"],
                "checkpoint": payload["checkpoint_path"],
                "config": payload["config_path"],
            }
            print(json.dumps(payload, indent=2))
        else:
            print_profile_summary(resolved)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
