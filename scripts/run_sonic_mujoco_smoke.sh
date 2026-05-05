#!/usr/bin/env bash
set -euo pipefail

echo "Run this from the official GR00T-WholeBodyControl repo root."
echo "Terminal 1:"
echo "  source .venv_sim/bin/activate"
echo "  python gear_sonic/scripts/run_sim_loop.py"
echo
echo "Terminal 2:"
echo "  cd gear_sonic_deploy"
echo "  bash deploy.sh sim"

