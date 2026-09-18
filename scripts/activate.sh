# Source this file: source scripts/activate.sh
export YCB_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. /home/vlakbnn/miniconda3/etc/profile.d/conda.sh
unset CONDA_ENVS_PATH
unset CONDA_PKGS_DIRS
conda activate /home/vlakbnn/miniconda3/envs/jiajunl_isaac || return 1
# ~/.bashrc prepends ~/.local/bin after initializing Conda.  Put this
# environment first again so python and pip always come from the same prefix.
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r 2>/dev/null || true
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIP_REQUIRE_VIRTUALENV=false
export PIP_NO_CACHE_DIR=1
export YCB_RUNTIME="$YCB_PROJECT/outputs/runtime"
export XDG_CACHE_HOME="$YCB_RUNTIME/cache"
export XDG_CONFIG_HOME="$YCB_RUNTIME/config"
export XDG_DATA_HOME="$YCB_RUNTIME/data"
export XDG_STATE_HOME="$YCB_RUNTIME/state"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export OMNI_KIT_ACCEPT_EULA=YES
export TMPDIR="$YCB_PROJECT/outputs/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME"

check_install_environment() {
    which python
    which pip
    conda info --envs
    [ "$(command -v python)" = "/home/vlakbnn/miniconda3/envs/jiajunl_isaac/bin/python" ] &&
    [ "$(command -v pip)" = "/home/vlakbnn/miniconda3/envs/jiajunl_isaac/bin/pip" ] &&
    [ "$CONDA_PREFIX" = "/home/vlakbnn/miniconda3/envs/jiajunl_isaac" ]
}
