Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.

This training example is released under the MIT license as listed
in the top-level directory. If this example is separated from the
main directory, include the LICENSE file with it.

Contributions from Suyash Tandon, Noel Chalmers, Nick Curtis,
Justin Chang, and Gina Sitaraman.

# Description document for the GPU-based Jacobi solver

## Contents:
---------
1.	[Application overview](#application-overview)
2.  [Prerequisites](#prerequisites)
3.	[Build instructions](#build-instructions)
4.	[Run instructions](#run-instructions)
---
## Application overview

This is a distributed Jacobi solver, using GPUs to perform the computation and MPI for halo exchanges.
It uses a 2D domain decomposition scheme to allow for a better computation-to-communication ratio than just 1D domain decomposition.

The flow of the application is as follows:
*	The MPI environment is initialized
*	The command-line arguments are parsed, and a MPI grid and mesh are created
*	Resources (including host and device memory blocks, streams etc.) are initialized
*	The Jacobi loop is executed; in every iteration, the local block is updated and then the halo values are exchanged; the algorithm
	converges when the global residue for an iteration falls below a threshold, but it is also limited by a maximum number of
	iterations (irrespective if convergence has been achieved or not)
*	Run measurements are displayed and resources are disposed

The application uses the following command-line arguments:
*	`-g x y`		-	mandatory argument for the process topology, `x` denotes the number of processes on the X direction (i.e. per row) and `y` denotes the number of processes on the Y direction (i.e. per column); the topology size must always match the number of available processes (i.e. the number of launched MPI processes must be equal to x * y)
*	`-m dx dy` 	-	optional argument indicating the size of the local (per-process) domain size; if it is omitted, the size will default to `DEFAULT_DOMAIN_SIZE` as defined in `defines.h`
* `-h | --help`	-	optional argument for printing help information; this overrides all other arguments

## Prerequisites

To build and run the jacobi application on A+A hardware, the following dependencies must be installed first:

* an MPI implementation (openMPI, MPICH, etc.)
* ROCm 2.1 or later.

## Build Instructions

A `Makefile` is included along with the source files that configures and builds multiple objects and then stitches them together to build the binary for the application `Jacobi_hip`. To build, simply run:
```
make
```
An alternative cmake build system is also include
```
mkdir build && cd build
cmake ..
make
```

## Run instructions

To run use:
```
mpirun -np 2 ./Jacobi_hip -g 2 1
```


## Instructions for Unison cluster

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
module use --prepend ${HOME}/curso/modules
module load openmpi/5.0-ucc1.3-ucx1.16
```

Compile the program

```bash
make
```

Run the Jacobi exercise

```bash
mpirun -np 1 ./Jacobi_hip -g 1 1 -m 8192 8192
```

## Kernel Statistics 

Get kernel performance statistics

```bash
mpirun -np 1 rocprofv3 -d rocprof_results -o results --kernel-trace --stats --truncate-kernels -- ./Jacobi_hip -g 1 1 -m 8192 8192
```

Check out the results

```bash
cat rocprof_results/results_kernel_stats.csv
```
You should see something like the following:

```bash
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"JacobiIterationKernel",1000,2484971674,2484971.674000,44.82,2098237,88032708,4995449.197068
"NormKernel1",1001,1853203384,1851352.031968,33.43,1591519,87335904,3514946.954781
"LocalLaplacianKernel",1000,1185656942,1185656.942000,21.39,1048959,87003489,2862313.845168
"HaloLaplacianKernel",1000,16566548,16566.548000,0.2988,15520,18240,406.756963
"NormKernel2",1001,3932479,3928.550450,0.0709,3520,5280,271.779796
"__amd_rocclr_fillBufferAligned",1,7200,7200.000000,1.299e-04,7200,7200,0.00000000e+00
```


## Profile the time line with Omnitrace

```bash
module load omnitrace
```

Generate a default configuration file
```bash
omnitrace-avail -G ~/.omnitrace.cfg
```

Edit your configuration file and set
```bash
vim ~/.omnitrace.cfg
OMNITRACE_SAMPLING_CPUS = none
OMNITRACE_SAMPLING_GPUS = 0
```

Set your omnitrace configuration file
```bash
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
```

Now we generate an instrumented binary

```bash
omnitrace-instrument -o ./Jacobi_hip.inst -- ./Jacobi_hip
```

Finally, we run the instrumented binary and use Omnitrace to trace our application

```bash
mpirun -np 1 omnitrace-run -- ./Jacobi_hip.inst -g 1 1 -m 8192 8192
```

The results are saved to the `omnitrace-Jacobi_hip.inst-output` directory.
Copy the results to your computer


```bash
scp -r curso_49@148.225.111.153:~/training_unison_2025/jacobi/omnitrace-Jacobi_hip.inst-output .
```

Go to [https://ui.perfetto.dev](https://ui.perfetto.dev) and load the file with `.proto` extension

## Using two GPUs

If you want to use more than one GPU, we recommend you submit your job instead of using an interactive session. 

To submit your job you need to write a SLURM script. Here an example  of the content.

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH -J jacobi
#SBATCH --time=0:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH -o job_output.log
#SBATCH -e job_error.log

echo "Starting job. $(date)"

module load rocm
module use --prepend ${HOME}/curso/modules
module load openmpi/5.0-ucc1.3-ucx1.16
module load omnitrace

export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg

mpirun -np 2 omnitrace-run -- ./Jacobi_hip.inst  -g 2 1 -m 8192 8192

echo "Finished job. $(date)"
```


Copy the content into a file, for example `submit_job.slurm`
and then submit the Job to the SLURM queues by running:

```bash
sbatch submit_job.slurm
```


Now you wile see two output files:

```bash
 perfetto-trace-0.proto  perfetto-trace-1.proto
```

You can merged them into a single one by running:

```bash
cat perfetto-trace-*.proto > merged.proto
```
You can load the `merged.proto` file into [https://ui.perfetto.dev](https://ui.perfetto.dev) to see the two timelines together