#!/bin/bash

set -euo pipefail

tmpdir=$(mktemp -d)
trap 'rm -rf "${tmpdir}"' EXIT

BUILD_DIR=${BUILD_DIR:-build}
USE_SLURM=${USE_SLURM:-0}
NUM_TASKS=${NUM_TASKS:-4}
NUM_NODES=${NUM_NODES:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}

I_TILES=${I_TILES:-2}

# Slurm settings (defaults specific to <https://gitlab.mff.cuni.cz/mff/hpc/clusters>)
SLURM_PARTITION=${SLURM_PARTITION:-"mpi-homo-short"}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-"kdss"}
SLURM_TIMEOUT=${SLURM_TIMEOUT:-"2:00:00"}


LD_LIBRARY_PATH="${BUILD_DIR}/boost-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "algorithm,framework,dataset,datatype,c_tile,a_tile,b_tile,i_tiles,mean_time,sd_time,valid"
files=$(find "$BUILD_DIR/examples/" -mindepth 2 -maxdepth 2 -type f -executable -name "gemmScatter-*-*-*-*-*-*-*-*-*" | shuf)
printf "Running the following algorithm implementations\n%s\n" "${files}" >&2
for file in ${files}; do
	IFS="-" read -r algorithmRaw framework c_scatter a_scatter b_scatter dataset datatype c_tile a_tile b_tile <<< "${file}"
	IFS="/" read -r buildDir _ _ algorithm <<< "${algorithmRaw}"

	# algorithm=$(sed 's/Scatter//g' <<< "${algorithm}")
	algorithm=${algorithm//Scatter/}

	if [[ -z "${algorithm}" ]]; then
		continue
	fi

	if [[ -z "${framework}" ]]; then
		continue
	fi

	if [[ -z "${c_scatter}" ]]; then
		continue
	fi

	if [[ -z "${a_scatter}" ]]; then
		continue
	fi

	if [[ -z "${b_scatter}" ]]; then
		continue
	fi

	if [[ -z "${dataset}" ]]; then
		continue
	fi

	if [[ "${dataset}" == "EXTRALARGE" ]]; then
		continue
	fi

	testFile="${buildDir}/${algorithm}-${dataset}-${datatype}.data"

	if outputs=$(bash ./run.sh "${file}" "${I_TILES}" "${testFile}"); then
		read -r mean_time sd_time <<< "${outputs}"
		echo "${algorithm},${framework}-${c_scatter}-${a_scatter}-${b_scatter},${dataset},${datatype},${c_tile},${a_tile},${b_tile},${I_TILES},${mean_time},${sd_time},1"
	else
		read -r mean_time sd_time <<< "${outputs}"
		echo "${algorithm},${framework}-${c_scatter}-${a_scatter}-${b_scatter},${dataset},${datatype},${c_tile},${a_tile},${b_tile},${I_TILES},${mean_time},${sd_time},0"
	fi
done
