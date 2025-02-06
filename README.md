# Layout-Agnostic MPI Abstraction for Modern C++

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

This repository contains the proof of concept implementation of the paper "Layout-Agnostic MPI Abstraction for Modern C++".

## About

Message Passing Interface (MPI) has been a well-established technology in the domain of distributed high-performance computing for several decades. However, one of its greatest drawbacks is a rather ancient pure-C interface. It lacks many useful features of modern languages (namely C++), like basic type-checking or support for generic code design. We propose a novel abstraction for MPI, which we implemented as an extension of the C++ Noarr library. It follows Noarr paradigms (first-class types and traverser abstraction) and offers layout-agnostic design of MPI applications. We also implemented a layout agnostic distributed GEMM kernel as a proof of concept that our abstraction is viable.

## How to build

To build the project, you need to have CMake and a C++ compiler installed. The project assumes support for C++20 or later and requires MPI to be installed on your system (and the mpi.h header file must be available). The other dependency, the Noarr library, is retrieved automatically by CMake.

To build the project, run the following commands:

```bash
# Get the source code from GitHub
git clone https://github.com/jiriklepl/noarr-mpi
cd noarr-mpi

# Configure the project
./configure.sh

# Build the project
./build.sh
```

The script `configure.sh` creates a build directory and runs CMake to configure the project. The script `build.sh` builds three executables:

- `gemm` - the baseline GEMM kernel implemented in the Noarr library

  the code for this variant is located in [examples/gemm/gemm.cpp](examples/gemm/gemm.cpp)

- `gemm-mpi` - the distributed GEMM kernel using the proposed Noarr MPI abstraction; slight modification of `gemm`

  the code for this variant is located in [examples/gemm/gemm-mpi.cpp](examples/gemm/gemm-mpi.cpp)

- `gemm-mpi-tilea-transpose` - the same distributed GEMM kernel, differing solely in the layout of the distributed sub-matrices of the input matrix `A` (no other code is changed, the code for matrix initialization and the GEMM kernel is the same as in `gemm-mpi`)

  the code for this variant is located in [examples/gemm/gemm-mpi.cpp](examples/gemm/gemm-mpi.cpp) as well. It is compiled with an additional preprocessor definition `A_TILE_K_MAJOR` that changes the layout of sub-matrix tiles of `A`.

The two MPI-distributed GEMM kernels are proof of concept implementations of the proposed Noarr MPI abstraction that showcase the proposed layout-agnostic design. They test two different layout configurations to demonstrate that the abstraction is indeed layout-agnostic.

## How to run

To run the script that automatically runs the each of the GEMM variants using `mpirun`, run the following command:

```bash
./compare.sh
```

The script performs $5$ warm-up runs and $10$ measurement runs of each GEMM variant. All outputs (the resulting $C$ matrix) are discarded for all runs except the last one for each variant. The execution times are printed in seconds in a CSV format. [data/compare-example.csv](data/compare-example.csv) shows a possible output of the script.

After all measurements are done, the script then compares the results of each variant and, if they differ, prints an error message.

You can control the number of mpi processes using the `NUM_TASKS` variable (default is 4) and the number of nodes using the `NUM_NODES` variable (default is 1). The mpi-related variables do not apply to the baseline GEMM kernel, which is always executed in a single process without MPI.

### Slurm

To run the same experiment on a Slurm cluster (such as <https://gitlab.mff.cuni.cz/mff/hpc/clusters>), modify the command as follows (replace `YOUR_ACCOUNT` and `YOUR_PARTITION` with your Slurm account and partition):

```bash
USE_SLURM=1 NUM_NODES=4 NUM_TASKS=64 ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION ./compare.sh
```

[data/compare-example-slurm.csv](data/compare-example-slurm.csv) shows a possible output of the script when run on a Slurm cluster with the specified parameters.

### Single run

To run a single run of a particular GEMM variant, use the following command:

```bash
./run.sh gemm-mpi
```

Again, you can use the `USE_SLURM` variable to run the experiment on a Slurm cluster or the `NUM_NODES` and `NUM_TASKS` variables to control the number of nodes and MPI processes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
