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

# Extract model name: look for "name:" under "model:" block
MODEL_NAME=$(awk '
/^model:/ { in_model=1; next }
in_model && /^[[:space:]]+name:/ {
    gsub(/^[[:space:]]*name:[[:space:]]*/, "")
    gsub(/[\"'\'']/, "")
    gsub(/[[:space:]]*#.*$/, "")
    print
    exit
}
/^[^[:space:]]/ && !/^model:/ { in_model=0 }
' "${CFG_PATH}")
if [[ -z "${MODEL_NAME}" ]]; then
    # Fallback: try flat "model:" line (old format)
    MODEL_NAME=$(grep -E "^model:" "${CFG_PATH}" | head -1 | sed -E 's/^model:[[:space:]]*([^[:space:]]+).*/\1/' | tr -d '"' | tr -d "'")
fi
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
  if conda run -n "${ENV_NAME}" pip install -r "${REQ_TXT}"; then
    echo "Installed packages from requirements.txt"
  else
    echo ""
    echo "WARNING: Some packages from requirements.txt failed to install."
    echo "         Installing packages one-by-one (skipping failures)..."
    while IFS= read -r line; do
      # skip comments and empty lines
      [[ "$line" =~ ^[[:space:]]*# ]] && continue
      [[ "$line" =~ ^[[:space:]]*$ ]] && continue
      # strip inline comments
      pkg="${line%%#*}"
      pkg="$(echo "$pkg" | xargs)"
      [[ -z "$pkg" ]] && continue
      if conda run -n "${ENV_NAME}" pip install "$pkg" 2>/dev/null; then
        echo "  OK: $pkg"
      else
        echo "  SKIP: $pkg (failed — may require CUDA or platform-specific build)"
      fi
    done < "${REQ_TXT}"
    echo "Finished processing requirements.txt (some packages may have been skipped)"
  fi
  # DON'T exit here — still need to install config-driven deps below
fi

# ── Fallback: parse README.md "## Requirements" section ──────────
# Many repos list deps only as conda/pip install commands
# inside a code block under a "## Requirements" heading in README.md.
README_MD="${MODEL_DIR}/README.md"
if [[ -f "${README_MD}" ]] && [[ ! -f "${SETUP_PY}" ]]; then
  echo "No requirements.txt, environment.yml, or setup.py found. Checking README.md for install commands..."

  # Extract package names from conda/pip install commands inside
  # the first "## Requirements" section's code blocks.
  awk '
  BEGIN { in_req=0; in_code=0 }
  /^#+[[:space:]]+[Rr]equirements[[:space:]]*$/ { in_req=1; next }
  in_req && /^#+[[:space:]]/ { in_req=0 }
  in_req && /^```/ { in_code = !in_code; next }
  in_req && in_code && /(conda|pip)[[:space:]]+install/ {
      # skip env management commands (create, activate, etc.)
      if (/conda[[:space:]]+(create|activate|deactivate|remove|env)/) next
      # strip everything up to and including "install"
      sub(/.*install[[:space:]]*/, "")
      n = split($0, tokens)
      skip_next = 0
      for (i = 1; i <= n; i++) {
          if (skip_next) { skip_next=0; continue }
          # skip flags; those that take a value also skip next token
          if (tokens[i] ~ /^-/) {
              if (tokens[i] ~ /^-(c|n|r|f|e|p)$/ || \
                  tokens[i] ~ /^--(name|channel|file|requirement|prefix|index-url|extra-index-url|editable|target)/)
                  skip_next = 1
              continue
          }
          if (length(tokens[i]) > 0) print tokens[i]
      }
  }
  ' "${README_MD}" | sed -E '
    # drop conda-only meta-packages that have no pip equivalent
    /^cudatoolkit/d
    # map conda names → pip names
    s/^pytorch(=|==|$)/torch\1/
    s/^opencv(=|==|$)/opencv-python\1/
    # convert conda single = version pin to pip ==
    s/([^=])=([^=])/\1==\2/
  ' > /tmp/readme_deps.txt

  if [[ -s /tmp/readme_deps.txt ]]; then
    echo "Found requirements in README.md:"
    cat /tmp/readme_deps.txt

    readme_pkgs=()
    while IFS= read -r pkg; do
      [[ -n "$pkg" ]] && readme_pkgs+=("$pkg")
    done < /tmp/readme_deps.txt

    if [[ ${#readme_pkgs[@]} -gt 0 ]]; then
      echo "Installing packages from README.md: ${readme_pkgs[*]}"
      if ! conda run -n "${ENV_NAME}" pip install "${readme_pkgs[@]}"; then
        echo "WARNING: Version-pinned install failed (versions may be outdated)."
        echo "Retrying without version pins..."
        stripped_pkgs=()
        for p in "${readme_pkgs[@]}"; do
          stripped_pkgs+=("${p%%[=<>!]*}")
        done
        if ! conda run -n "${ENV_NAME}" pip install "${stripped_pkgs[@]}"; then
          echo "ERROR: Failed to install packages parsed from README.md"
          exit 1
        fi
      fi
      echo "Successfully installed packages from README.md"
    fi
  fi
fi

# Install optional conda deps from config (needed before setup.py or pip installs)
# Extract conda dependencies using simple text parsing
# Get lines between "conda:" and "pip:"
awk '
/^[[:space:]]*conda:.*\[/ {
    # Inline array: conda: [pkg1, pkg2, ...]
    sub(/.*\[/, "")
    sub(/\].*/, "")
    n = split($0, items, /[[:space:]]*,[[:space:]]*/)
    for (i = 1; i <= n; i++) {
        gsub(/^[[:space:]]*["'\'']?/, "", items[i])
        gsub(/["'\'']?[[:space:]]*$/, "", items[i])
        if (length(items[i]) > 0) print items[i]
    }
    next
}
/^[[:space:]]*conda:/ { in_conda=1; next }
/^[[:space:]]*pip:/ { in_conda=0; next }
in_conda && /^[[:space:]]+-/ {
    # Multi-line list: - "package"
    gsub(/^[[:space:]]+-[[:space:]]*/, "")
    gsub(/^["'\'']/, "")
    gsub(/["'\''].*$/, "")
    gsub(/[[:space:]]*#.*$/, "")
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

# Also install config-driven pip deps (covers framework deps like pyyaml)
echo "Installing any additional dependencies from config (cfg.setup.*)..."

# Install pip deps from config
# Extract pip dependencies using simple text parsing
# Get lines after "pip:" until next top-level key
awk '
/^[[:space:]]*pip:.*\[/ {
    # Inline array: pip: [pkg1, pkg2, ...]
    sub(/.*\[/, "")
    sub(/\].*/, "")
    n = split($0, items, /[[:space:]]*,[[:space:]]*/)
    for (i = 1; i <= n; i++) {
        gsub(/^[[:space:]]*["'\'']?/, "", items[i])
        gsub(/["'\'']?[[:space:]]*$/, "", items[i])
        if (length(items[i]) > 0) print items[i]
    }
    next
}
/^[[:space:]]*pip:/ { in_pip=1; next }
/^[^[:space:]]/ && !/^[[:space:]]*#/ && !/^setup:/ { if(in_pip) in_pip=0; next }
in_pip && /^[[:space:]]+-/ {
    # Multi-line list: - "package"
    gsub(/^[[:space:]]+-[[:space:]]*/, "")
    gsub(/^["'\'']/, "")
    gsub(/["'\''].*$/, "")
    gsub(/[[:space:]]*#.*$/, "")
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
