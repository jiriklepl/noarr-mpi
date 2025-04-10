#!/bin/bash

set -euo pipefail

tmpdir=$(mktemp -d)
trap 'rm -rf "${tmpdir}"' EXIT

BUILD_DIR=${BUILD_DIR:-build}
USE_SLURM=${USE_SLURM:-0}
NUM_TASKS=${NUM_TASKS:-4}
NUM_NODES=${NUM_NODES:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}

WARMUP_RUNS=${WARMUP_RUNS:-3}
NUM_RUNS=${NUM_RUNS:-5}

# Slurm settings (defaults specific to <https://gitlab.mff.cuni.cz/mff/hpc/clusters>)
SLURM_PARTITION=${SLURM_PARTITION:-mpi-homo-short}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-kdss}

echo "algorithm,framework,dataset,datatype,c_tile,a_tile,b_tile,time"
find "$BUILD_DIR/examples/" -mindepth 2 -maxdepth 2 -type f -executable -name "gemm-*-*-*-*-*-*" |
	while read -r file; do
		IFS="-" read algorithm framework dataset datatype c_tile a_tile b_tile <<< "${file}"

		echo "Running benchmark for ${algorithm} with ${framework} on ${dataset}..." >&2

		if output_time=$(bash ./run.sh "${file}" | tail -n1); then
			echo "${algorithm},${framework},${dataset},${datatype},${c_tile},${a_tile},${b_tile},${output_time}"
		else
			echo "${algorithm},${framework},${dataset},${datatype},${c_tile},${a_tile},${b_tile},ERROR"
		fi
	done
