# Layout-Agnostic MPI Abstraction for Modern C++

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE) [![DOI](https://img.shields.io/badge/License-DOI-yellow.svg)]()

This repository contains the proof of concept implementation of the paper *Layout-Agnostic MPI Abstraction for Modern C++* and the evaluation of the proposed abstraction using a distributed GEMM kernel.

## About

Message Passing Interface (MPI) has been a well-established technology in the domain of distributed high-performance computing for several decades. However, one of its greatest drawbacks is a rather ancient pure-C interface. It lacks many useful features of modern languages (namely C++), like basic type-checking or support for generic code design. In this paper, we propose a novel abstraction for MPI, which we implemented as an extension of the C++ Noarr library. It follows Noarr paradigms (first-class layout and traversal abstraction) and offers layout-agnostic design of MPI applications. We also implemented a layout-agnostic distributed GEMM kernel as a case study to demonstrate the usability and syntax of the proposed abstraction. We show that the abstraction achieves performance comparable to the state-of-the-art MPI C++ bindings while allowing for a more flexible design of distributed applications.

## Library

The project contains a header-only extension of the Noarr library that provides a layout-agnostic C++ abstraction for MPI. The library can be included using the following include directive (assuming an MPI implementation is available on the system and the [include](include) directory is in the include path):

```cpp
#include <noarr/mpi.hpp>
```

The library provides a set of abstractions for MPI operations, including:

- `mpi_transform` ([include/noarr/mpi/transform.hpp](include/noarr/mpi/transform.hpp)) - a function that transforms Noarr structures to MPI data types.
- `mpi_traverser_t` ([include/noarr/mpi/traverser.hpp](include/noarr/mpi/traverser.hpp)) - a class that associates a Noarr traverser with an MPI communicator.
- `scatter`, `gather`, `broadcast` ([include/noarr/mpi/algorithms.hpp](include/noarr/mpi/algorithms.hpp)) - functions that implement collective operations for Noarr structures.

## GEMM kernel implementations

To showcase the proposed abstraction, we implemented a distributed GEMM kernel using the proposed Noarr MPI abstraction and compared it with other libraries. All these implementations are included in the [examples](examples) directory.

- [examples/noarr/gemm.cpp](examples/noarr/gemm.cpp) - implementation of the distributed GEMM kernel using the Noarr library and the proposed Noarr MPI abstraction.
- [examples/boost/gemm.cpp](examples/boost/gemm.cpp) - implementation of the distributed GEMM kernel using the Boost.MPI library and serialization. This implementation ensures the layout-agnostic design of the GEMM kernel via the `mdspan` abstraction that is part of the C++ standard.
- [examples/boostP2P/gemm.cpp](examples/boostP2P/gemm.cpp) - implementation of the distributed GEMM kernel using the Boost.MPI library and point-to-point communication of matrices serialized into input/output archives. This implementation ensures the layout-agnostic design of the GEMM kernel via the `mdspan` abstraction that is part of the C++ standard.
- [examples/kokkosComm/gemm.cpp](examples/kokkosComm/gemm.cpp) - implementation of the distributed GEMM kernel using the Kokkos library and the KokkosComm abstraction that enables communication over MPI.
- [examples/mpi/gemm.cpp](examples/mpi/gemm.cpp) - implementation of the distributed GEMM kernel using the MPI interface directly. This implementation ensures the layout-agnostic design of the GEMM kernel via the `mdspan` abstraction that is part of the C++ standard.

All implementations are based on the GEMM kernel implementation from the PolyBench/C 4.2.1 benchmark suite by Louis-Noël Pouchet et al. The original code is available at <https://sourceforge.net/projects/polybench/files>.

## How to build

To build the project, you need to have CMake and a C++ compiler installed. The project assumes support for C++20 or later and requires MPI to be installed on your system (and the `mpi.h` header file must be available). The other dependency, the Noarr library, is retrieved automatically by CMake.

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

The script `configure.sh` creates a build directory and runs CMake to configure the project. The script `build.sh` builds the executables for all GEMM variants and configurations differing in the major dimensions of the privatized sub-matrices used in the distributed GEMM kernel and the dataset size. For each dataset size, there are eight configurations of the GEMM kernel in total, each named `gemm-<framework>-<dataset-size>-<C-tile-major-dim>-<A-tile-major-dim>-<B-tile-major-dim>`, where `<framework>` is the name of the framework used (e.g., `noarr`, `boost`, `boostP2P`, `kokkosComm`, `mpi`), `<dataset-size>` is the size of the dataset (`MINI`, `MEDIUM`, `EXTRALARGE`). The dataset sizes are defined in [examples/include/gemm.hpp](examples/include/gemm.hpp).

## How to run

To run the script that automatically runs each of the GEMM variants using `mpirun`, run the following command:

```bash
./compare.sh
```

Each execution performs an unmeasured warm-up run followed by 20 measurement runs of a given GEMM kernel implementation, dataset size, and configuration. The execution then reports the average execution time and the standard deviation of the measurements in seconds. The output of the script is in CSV format with the following columns:

```csv
algorithm,framework,dataset,datatype,c_tile,a_tile,b_tile,i_tiles,mean_time,sd_time,valid
```

- `algorithm` - the name of the algorithm (always `gemm`).
- `framework` - the name of the framework used (e.g., `noarr`, `boost`, `boostP2P`, `kokkosComm`, `mpi`).
- `dataset` - the size of the dataset (`MINI`, `MEDIUM`, `EXTRALARGE`).
- `datatype` - the type of the data used in the GEMM kernel (for the default configuration of the project, it is always `FLOAT`).
- `c_tile`, `a_tile`, `b_tile` - the major dimensions of the privatized sub-matrices used in the distributed GEMM kernel.
- `i_tiles` - the number of tiles in the `i` dimension of the `C` matrix; the number of tiles in the `j` dimension is determined by the number of MPI processes.
- `mean_time` - the average execution time of the GEMM kernel in seconds.
- `sd_time` - the standard deviation of the execution time in seconds.
- `valid` - a flag indicating whether the result of the GEMM kernel is valid (1) or not (0).

### Slurm

To run the same experiment on a Slurm cluster (such as <https://gitlab.mff.cuni.cz/mff/hpc/clusters>), modify the command as follows (replace `YOUR_ACCOUNT` and `YOUR_PARTITION` with your Slurm account and partition):

```bash
USE_SLURM=1 NUM_TASKS=8 NUM_NODES=8 I_TILES=2 ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION ./compare.sh
```

[data/compare-example-slurm.csv](data/compare-example-slurm.csv) shows a possible output of the script when run on a Slurm cluster with the specified parameters.

## Visualization

Generate virtual Python environment and enter it:

```bash
python3 -m venv .venv
. .venv/bin/activate
```

Upgrade pip and install the requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To visualize the results of the GEMM kernel execution, run the following command:

```bash
python3 gen_plots.py <path_to_csv_file>
```

## Reproducing the results reported in the paper

To reproduce the results reported in the paper, run the following sequence of commands:

```bash
./configure.sh
./build.sh

mkdir -p data

# May run for over an hour; add USE_SLURM=1 to run on a Slurm cluster
NUM_TASKS=8 NUM_NODES=8 I_TILES=2 ./compare.sh > data/compare.csv 2> data/compare.err

# Further experiments for more stable results
NUM_TASKS=8 NUM_NODES=8 I_TILES=2 ./compare.sh > data/compare2.csv 2> data/compare2.err
NUM_TASKS=8 NUM_NODES=8 I_TILES=2 ./compare.sh > data/compare3.csv 2> data/compare3.err

# Generate plots (only one experiment)
./gen_plots.py --show-sdev --no-validation --no-boostP2P data/compare.csv

# Generate plots (all experiments aggregated)
./gen_plots.py --show-sdev --no-validation --no-boostP2P data/compare.csv data/compare2.csv data/compare3.csv
```

## Testing

To run the tests that verify the type safety of the Noarr MPI abstraction (requires `ctest`), run the following command:

```bash
./test.sh
```

The result of the command is a list of tests and their outcomes. All tests should pass.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
