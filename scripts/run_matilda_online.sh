#!/bin/bash -l
set -eo pipefail

cd /projects/scale-lab/xinyiwang/CODE/diff_rl_safe
if [[ ! -f scripts/run_gpu.sh ]]; then
    echo 'Missing scripts/run_gpu.sh; update the Matilda checkout first.' >&2
    exit 1
fi

module load miniconda3/current
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate
conda activate diffrl
hash -r

# Fail before running commands with a Python outside the activated environment.
python - <<'CHECK_PYTHON'
import os
import sys

expected = os.path.realpath(os.environ["CONDA_PREFIX"])
actual = os.path.realpath(sys.prefix)
if actual != expected:
    raise RuntimeError(
        f"Wrong Python: {sys.executable}; expected environment {expected}, got {actual}"
    )
import jax
print(f"Python: {sys.executable}; JAX: {jax.__version__}", flush=True)
CHECK_PYTHON

# First launch prompts for your W&B API key; later launches reuse the login.
wandb login

sbatch <<'SLURM'
#!/bin/bash -l
#SBATCH --job-name=diffrl-all
#SBATCH --account=scale-lab
#SBATCH --partition=gpu6000
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:6000:1
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/projects/scale-lab/xinyiwang/CODE/diff_rl_safe
#SBATCH --output=/projects/scale-lab/xinyiwang/CODE/diff_rl_safe/slurm-%j.out

set -eo pipefail
module load miniconda3/current
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate
conda activate diffrl
hash -r

# Fail before running commands with a Python outside the activated environment.
python - <<'CHECK_PYTHON'
import os
import sys

expected = os.path.realpath(os.environ["CONDA_PREFIX"])
actual = os.path.realpath(sys.prefix)
if actual != expected:
    raise RuntimeError(
        f"Wrong Python: {sys.executable}; expected environment {expected}, got {actual}"
    )
import jax
print(f"Python: {sys.executable}; JAX: {jax.__version__}", flush=True)
CHECK_PYTHON

export WANDB_MODE=online
export WANDB_ENTITY=xinywa
export WANDB_PROJECT=diff_rl_safe
export RUN_NAME="matilda_${SLURM_JOB_ID}"
export CPU_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/users/xinyiwang/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

# CBF evaluation results already exist; run the remaining RL experiments.
for method in ppo ppo_filter barriernet barriernet_sim; do
    bash scripts/run_gpu.sh "$method" all
done
SLURM
