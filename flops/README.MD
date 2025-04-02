# FLOPS Example

## How to use

Request a GPU node

```bash
srun -t 5:00:00 --partition=gpu --nodes=1 --gpus=1 --pty bash -i
```

Load necessary modules 

```bash
module load rocm
```

Compile 

```bash
make N_ITER=16
```

```bash 
source /lustre/home/fnsc03/shared_cursoU/miniconda3/etc/profile.d/conda.sh
conda activate rocprof-compute
export PATH=/lustre/home/fnsc03/shared_cursoU/rocprof-compute/bin:$PATH
```

```bash
ROCPROF=rocprofv2 rocprof-compute profile -n flops --roof-only -- ./gpu_flops 100
```