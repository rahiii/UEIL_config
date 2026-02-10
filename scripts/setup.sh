#!/usr/bin/env bash
set -euo pipefail

CFG_PATH="${1:-}"
if [[ -z "${CFG_PATH}" ]]; then
  echo "Usage: scripts/setup.sh <config_path>"
  exit 1
fi

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "Config not found: ${CFG_PATH}"
  exit 1
fi

# Resolve repo root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Read model name + setup info from YAML config (simple parsing without yaml module)
# Extract model name (first line with "model:")
MODEL_NAME=$(grep -E "^model:" "${CFG_PATH}" | head -1 | sed -E 's/^model:[[:space:]]*([^[:space:]]+).*/\1/' | tr -d '"' | tr -d "'")
if [[ -z "${MODEL_NAME}" ]]; then
    echo "Error: Could not find 'model:' in config file ${CFG_PATH}"
    exit 1
fi

# Extract python version (look for "python:" under "setup:")
PY_VER=$(grep -A 10 "^setup:" "${CFG_PATH}" | grep -E "^[[:space:]]*python:" | head -1 | sed -E 's/^[[:space:]]*python:[[:space:]]*"([^"]+)".*/\1/' | sed -E "s/^[[:space:]]*python:[[:space:]]*'([^']+)'.*/\1/")
if [[ -z "${PY_VER}" ]]; then
    PY_VER="3.10"  # default
fi

MODEL_DIR="${ROOT_DIR}/models/${MODEL_NAME}"
ENV_NAME="${MODEL_NAME}"

echo "Repo root: ${ROOT_DIR}"
echo "Model from config: ${MODEL_NAME}"
echo "Model dir: ${MODEL_DIR}"
echo "Conda env: ${ENV_NAME}"
echo "Python: ${PY_VER}"

# Enable conda activate in non-interactive shells
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH."
  echo "Run this once in your terminal, then retry:"
  echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env if missing (or activate if present)
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env exists: ${ENV_NAME}"
else
  echo "Creating conda env: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi

conda activate "${ENV_NAME}"

# Prefer model-provided env files if present
ENV_YML="${MODEL_DIR}/environment.yml"
REQ_TXT="${MODEL_DIR}/requirements.txt"
SETUP_PY="${MODEL_DIR}/setup.py"

if [[ -f "${ENV_YML}" ]]; then
  echo "Using model environment.yml: ${ENV_YML}"
  conda env update -n "${ENV_NAME}" -f "${ENV_YML}"
  echo "Setup complete."
  echo "Next run:"
  echo "  conda activate ${ENV_NAME}"
  echo "  python train.py --config ${CFG_PATH}"
  exit 0
fi

if [[ -f "${REQ_TXT}" ]]; then
  echo "Using model requirements.txt: ${REQ_TXT}"
  conda run -n "${ENV_NAME}" pip install -r "${REQ_TXT}"
  echo "Setup complete."
  echo "Next run:"
  echo "  conda activate ${ENV_NAME}"
  echo "  python train.py --config ${CFG_PATH}"
  exit 0
fi

# Install optional conda deps from config (needed before setup.py or pip installs)
# Extract conda dependencies using simple text parsing
# Get lines between "conda:" and "pip:"
awk '
/^[[:space:]]*conda:/ { in_conda=1; next }
/^[[:space:]]*pip:/ { in_conda=0; next }
in_conda && /^[[:space:]]+-/ {
    # Extract package name from: - "package" or - 'package'
    gsub(/^[[:space:]]+-[[:space:]]*/, "")  # Remove leading spaces and dash
    gsub(/^["'\'']/, "")  # Remove leading quote
    gsub(/["'\''].*$/, "")  # Remove trailing quote and everything after
    gsub(/[[:space:]]*#.*$/, "")  # Remove comments
    if(length($0) > 0) print
}
' "${CFG_PATH}" > /tmp/conda_deps.txt

if [[ -s /tmp/conda_deps.txt ]]; then
  echo "Installing conda deps from config:"
  cat /tmp/conda_deps.txt
  # Read packages into array for safe installation
  conda_packages=()
  while IFS= read -r pkg; do
    if [[ -n "$pkg" ]]; then
      conda_packages+=("$pkg")
    fi
  done < /tmp/conda_deps.txt
  # Install all packages at once if we have any
  if [[ ${#conda_packages[@]} -gt 0 ]]; then
    echo "Installing packages: ${conda_packages[*]}"
    # Use pip for all packages (more reliable than conda, especially for numpy/scipy/matplotlib)
    # These packages work perfectly fine from pip
    if ! conda run -n "${ENV_NAME}" pip install "${conda_packages[@]}"; then
      echo "ERROR: Failed to install packages via pip"
      exit 1
    fi
    echo "Successfully installed packages via pip"
  fi
fi

if [[ -f "${SETUP_PY}" ]]; then
  echo "Using model setup.py: ${SETUP_PY}"
  echo "Installing model package in editable mode..."
  # Install in editable mode so changes to model code are reflected immediately
  # setup.py's install_requires will be installed automatically
  if ! conda run -n "${ENV_NAME}" pip install -e "${MODEL_DIR}"; then
    echo "ERROR: Failed to install model package from setup.py"
    exit 1
  fi
  echo "Setup complete."
  echo "Next run:"
  echo "  conda activate ${ENV_NAME}"
  echo "  python train.py --config ${CFG_PATH}"
  exit 0
fi

# Otherwise: use config-driven pip deps (fallback if no setup.py)
echo "No environment.yml / requirements.txt / setup.py found for model. Using cfg.setup.* dependencies."

# Install pip deps from config
# Extract pip dependencies using simple text parsing
# Get lines after "pip:" until next top-level key
awk '
/^[[:space:]]*pip:/ { in_pip=1; next }
/^[^[:space:]]/ && !/^[[:space:]]*#/ && !/^setup:/ { if(in_pip) in_pip=0; next }
in_pip && /^[[:space:]]+-/ {
    # Extract package name from: - "package" or - 'package'
    gsub(/^[[:space:]]+-[[:space:]]*/, "")  # Remove leading spaces and dash
    gsub(/^["'\'']/, "")  # Remove leading quote
    gsub(/["'\''].*$/, "")  # Remove trailing quote and everything after
    gsub(/[[:space:]]*#.*$/, "")  # Remove comments
    if(length($0) > 0) print
}
' "${CFG_PATH}" > /tmp/pip_deps.txt

if [[ -s /tmp/pip_deps.txt ]]; then
  echo "Installing pip deps from config:"
  cat /tmp/pip_deps.txt
  # Use conda run to ensure we're in the right environment
  echo "Running: conda run -n ${ENV_NAME} pip install -r /tmp/pip_deps.txt"
  if ! conda run -n "${ENV_NAME}" pip install -r /tmp/pip_deps.txt; then
    echo "ERROR: Failed to install pip packages"
    exit 1
  fi
  echo "Successfully installed pip packages"
else
  echo "No cfg.setup.pip deps provided; skipping pip installs."
fi

echo "Setup complete."
echo "Next run:"
echo "  conda activate ${ENV_NAME}"
echo "  python train.py --config ${CFG_PATH}"
