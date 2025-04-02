# FLOPS Example

This is a simple example of a kernel that can be modified to use peak GPU memory bandwidth of peak GPU compute.  

## How to use

First, login to the Unison cluster 

```bash
ssh fnsc03@148.225.111.153
```

Then, request a GPU node

```bash
srun -t 5:00:00 --partition=gpu --nodes=1 --gpus=1 --pty bash -i
```

Load necessary modules 

```bash
module load rocm
```

Compile the program and determine the number of iterations.

```bash
make N_ITER=16
```

Add `rocprof-compute` to your path

```bash 
source /lustre/home/fnsc03/shared_cursoU/miniconda3/etc/profile.d/conda.sh
conda activate rocprof-compute
export PATH=/lustre/home/fnsc03/shared_cursoU/rocprof-compute/bin:$PATH
```

Run rocprof-compute in profiling mode

```bash
ROCPROF=rocprofv2 rocprof-compute profile -n flops -- ./gpu_flops 100
```

Run rocprof-compute in analysis mode


```bash
rocprof-compute analyze -p workloads/flops/MI210
```