# Source this file: source scripts/activate.sh

export ISAAC_SIMLAB_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ISAACLAB_ROOT="${ISAACLAB_ROOT:-/media/data-ssd/software/IsaacLab-v2.0.2}"

if command -v conda >/dev/null 2>&1; then
    ISAAC_SIMLAB_CONDA_BASE="$(conda info --base)"
else
    ISAAC_SIMLAB_CONDA_BASE=""
    for candidate in "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/anaconda3" /opt/conda; do
        if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
            ISAAC_SIMLAB_CONDA_BASE="$candidate"
            break
        fi
    done
    if [ -z "$ISAAC_SIMLAB_CONDA_BASE" ]; then
        echo "[isaac_simlab] Conda was not found. Install and initialize Miniconda or Mambaforge." >&2
        return 1
    fi
fi
. "$ISAAC_SIMLAB_CONDA_BASE/etc/profile.d/conda.sh"

unset CONDA_ENVS_PATH
unset CONDA_PKGS_DIRS

# Never guess a developer's environment name. Use an explicit override, or
# preserve the non-base Conda environment that the developer already activated.
if [ -n "${ISAAC_SIMLAB_ENV:-}" ]; then
    conda activate "$ISAAC_SIMLAB_ENV" || return 1
elif [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    export ISAAC_SIMLAB_ENV="$CONDA_DEFAULT_ENV"
else
    echo "[isaac_simlab] No project Conda environment is selected." >&2
    echo "  Activate your environment first: conda activate <your-env>" >&2
    echo "  Or set it explicitly: export ISAAC_SIMLAB_ENV=<your-env>" >&2
    return 1
fi

export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r 2>/dev/null || true
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

# The project itself remains importable from its checkout. Isaac Lab must be
# linked into this user's Conda environment during the README installation.
export PYTHONPATH="$ISAAC_SIMLAB_PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

export PIP_REQUIRE_VIRTUALENV=false
export PIP_NO_CACHE_DIR=1
export ISAAC_SIMLAB_RUNTIME="$ISAAC_SIMLAB_PROJECT/outputs/runtime"
export XDG_CACHE_HOME="$ISAAC_SIMLAB_RUNTIME/cache"
export XDG_CONFIG_HOME="$ISAAC_SIMLAB_RUNTIME/config"
export XDG_DATA_HOME="$ISAAC_SIMLAB_RUNTIME/data"
export XDG_STATE_HOME="$ISAAC_SIMLAB_RUNTIME/state"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR="$ISAAC_SIMLAB_PROJECT/outputs/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME"

check_install_environment() {
    local expected_tag="v2.0.2"
    local python_path
    local pip_path
    local actual_tag

    python_path="$(command -v python)"
    pip_path="$(command -v pip)"

    if [ -z "${CONDA_PREFIX:-}" ] || [ "${CONDA_DEFAULT_ENV:-}" != "$ISAAC_SIMLAB_ENV" ]; then
        echo "[isaac_simlab] Expected Conda environment '$ISAAC_SIMLAB_ENV', got '${CONDA_DEFAULT_ENV:-none}'." >&2
        return 1
    fi
    if [ "$python_path" != "$CONDA_PREFIX/bin/python" ] || [ "$pip_path" != "$CONDA_PREFIX/bin/pip" ]; then
        echo "[isaac_simlab] python/pip do not come from $CONDA_PREFIX." >&2
        echo "  python: $python_path" >&2
        echo "  pip:    $pip_path" >&2
        return 1
    fi
    if [ ! -d "$ISAACLAB_ROOT/.git" ] || [ ! -d "$ISAACLAB_ROOT/source/isaaclab" ]; then
        echo "[isaac_simlab] Shared Isaac Lab checkout is missing: $ISAACLAB_ROOT" >&2
        return 1
    fi
    actual_tag="$(git -C "$ISAACLAB_ROOT" describe --tags --exact-match 2>/dev/null || true)"
    if [ "$actual_tag" != "$expected_tag" ]; then
        echo "[isaac_simlab] Isaac Lab must be $expected_tag; found '${actual_tag:-untagged}' at $ISAACLAB_ROOT." >&2
        return 1
    fi

    ISAACLAB_ROOT="$ISAACLAB_ROOT" python - <<'PY'
import os
import sys
from pathlib import Path

try:
    import isaaclab
    import isaacsim
    import isaac_simlab
except Exception as exc:
    raise SystemExit(f"[isaac_simlab] Import verification failed: {type(exc).__name__}: {exc}")

root = Path(os.environ["ISAACLAB_ROOT"]).resolve()
module_path = Path(isaaclab.__file__).resolve()
try:
    module_path.relative_to(root)
except ValueError:
    raise SystemExit(
        f"[isaac_simlab] isaaclab resolves outside ISAACLAB_ROOT:\n"
        f"  expected root: {root}\n  actual file:   {module_path}"
    )

print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV')}")
print(f"Python: {sys.executable}")
print(f"Isaac Lab tag: v2.0.2")
print(f"isaaclab: {module_path}")
print(f"isaacsim: {Path(isaacsim.__file__).resolve()}")
print(f"isaac_simlab: {Path(isaac_simlab.__file__).resolve()}")
PY
}
