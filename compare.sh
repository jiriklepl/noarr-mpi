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
I_TILES=${I_TILES:-2}

# Slurm settings (defaults specific to <https://gitlab.mff.cuni.cz/mff/hpc/clusters>)
SLURM_PARTITION=${SLURM_PARTITION:-mpi-homo-short}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-kdss}

echo "algorithm,framework,dataset,datatype,c_tile,a_tile,b_tile,i_tiles,time,valid"
files=$(find "$BUILD_DIR/examples/" -mindepth 2 -maxdepth 2 -type f -executable -name "gemm-*-*-*-*-*-*")
for file in ${files}; do
	IFS="-" read -r algorithmRaw framework dataset datatype c_tile a_tile b_tile <<< "${file}"
	IFS="/" read -r buildDir examplesDir frameworkDir algorithm <<< "${algorithmRaw}"

	testFile="${buildDir}/${algorithm}-${dataset}-${datatype}.data"

	if output_time=$(bash ./run.sh "${file}" "${I_TILES}" "${testFile}" | tail -n1); then
		echo "${algorithm},${framework},${dataset},${datatype},${c_tile},${a_tile},${b_tile},${I_TILES},${output_time},1"
	else
		echo "${algorithm},${framework},${dataset},${datatype},${c_tile},${a_tile},${b_tile},${I_TILES},${output_time},0"
	fi
done
