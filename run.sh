#!/bin/bash

set -e

EXECUTABLE=${1:-gemm-mpi}

BUILD_DIR=${BUILD_DIR:-build}
USE_SLURM=${USE_SLURM:-0}
NUM_TASKS=${NUM_TASKS:-4}
NUM_NODES=${NUM_NODES:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}

# Slurm settings (defaults specific to <https://gitlab.mff.cuni.cz/mff/hpc/clusters>)
SLURM_PARTITION=${SLURM_PARTITION:-mpi-homo-short}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-kdss}

if [ ! -f "${BUILD_DIR}/${EXECUTABLE}" ]; then
	echo "Error: Executable not found. Run build.sh first." >&2
	exit 1
fi

TASKS_PER_NODE=$((NUM_TASKS / NUM_NODES))
REMAINDER=$((NUM_TASKS % NUM_NODES))

if [ "${REMAINDER}" -ne 0 ]; then
	echo "Error: the number of tasks (${NUM_TASKS}) is not divisible by the number of nodes (${NUM_NODES})"
	exit 1
fi

case "${EXECUTABLE}" in
	*-mpi*)
		if [ "${USE_SLURM}" -eq 1 ]; then
			srun -n "${NUM_TASKS}" -N "${NUM_NODES}" --ntasks-per-node="${TASKS_PER_NODE}" -c "${CPUS_PER_TASK}" \
				-p "${SLURM_PARTITION}" -A "${SLURM_ACCOUNT}" \
				-- "${BUILD_DIR}/${EXECUTABLE}"
		elif [ "${CPUS_PER_TASK}" -gt 1 ]; then
			echo "Error: For multi-cpu-per-task runs, use SLURM." >&2
			exit 1
		else
			mpirun -np "${NUM_TASKS}" --npernode "${TASKS_PER_NODE}" "${BUILD_DIR}/${EXECUTABLE}"
		fi
	;;
	*)
		if [ "${USE_SLURM}" -eq 1 ]; then
			srun -n 1 -N 1 --ntasks-per-node="${TASKS_PER_NODE}" -c "${CPUS_PER_TASK}" \
				-p "${SLURM_PARTITION}" -A "${SLURM_ACCOUNT}" \
				-- "${BUILD_DIR}/${EXECUTABLE}"
		else
			"${BUILD_DIR}/${EXECUTABLE}"
		fi
	;;
esac
