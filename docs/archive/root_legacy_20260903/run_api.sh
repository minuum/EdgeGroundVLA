#!/bin/bash
# Load environment settings (checkpoint path, config path, API key, etc.)
if [ -f "/home/soda/vla/.vla_env_settings" ]; then
    source /home/soda/vla/.vla_env_settings
fi

export CUDA_VISIBLE_DEVICES=0
python3 -m robovlm_nav.serve.api_server_fix > server.log 2>&1 &
echo $! > api_server.pid
