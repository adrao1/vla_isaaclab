# Source this file: source scripts/activate.sh
export ISAAC_SIMLAB_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. /home/vlakbnn/miniconda3/etc/profile.d/conda.sh
unset CONDA_ENVS_PATH
unset CONDA_PKGS_DIRS
conda activate /home/vlakbnn/miniconda3/envs/jiajunl_isaac || return 1
# ~/.bashrc prepends ~/.local/bin after initializing Conda.  Put this
# environment first again so python and pip always come from the same prefix.
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r 2>/dev/null || true
export PYTHONNOUSERSITE=1
# The Conda environment's editable Isaac Lab install contains absolute paths.
# Point directly at this checkout so renaming or moving this repository does not
# leave imports bound to an obsolete project path.
export PYTHONPATH="$ISAAC_SIMLAB_PROJECT/src:$ISAAC_SIMLAB_PROJECT/IsaacLab/source/isaaclab:$ISAAC_SIMLAB_PROJECT/IsaacLab/source/isaaclab_assets:$ISAAC_SIMLAB_PROJECT/IsaacLab/source/isaaclab_mimic:$ISAAC_SIMLAB_PROJECT/IsaacLab/source/isaaclab_rl:$ISAAC_SIMLAB_PROJECT/IsaacLab/source/isaaclab_tasks"
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
    which python
    which pip
    conda info --envs
    [ "$(command -v python)" = "/home/vlakbnn/miniconda3/envs/jiajunl_isaac/bin/python" ] &&
    [ "$(command -v pip)" = "/home/vlakbnn/miniconda3/envs/jiajunl_isaac/bin/pip" ] &&
    [ "$CONDA_PREFIX" = "/home/vlakbnn/miniconda3/envs/jiajunl_isaac" ]
}
