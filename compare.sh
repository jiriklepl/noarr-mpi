#!/bin/bash

set -e

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

algorithms=(gemm gemm-mpi gemm-mpi-tileb-transpose)

echo "algorithm,seconds"
for algorithm in "${algorithms[@]}"; do
	for _ in $(seq "${WARMUP_RUNS}"); do
		bash ./run.sh "${algorithm}" &>/dev/null
	done
	for i in $(seq "${NUM_RUNS}"); do
		printf "%s" "${algorithm},"
		if [ "${i}" -eq "${NUM_RUNS}" ]; then
			{ bash ./run.sh "${algorithm}" >"${tmpdir}/${algorithm}.out"; } 2>&1
		else
			{ bash ./run.sh "${algorithm}" >/dev/null; } 2>&1
		fi
	done
done

for algorithm in "${algorithms[@]}"; do
	if [ "${algorithm}" != "gemm" ]; then
		cmp "${tmpdir}/${algorithm}.out" "${tmpdir}/gemm.out"
	fi
done
