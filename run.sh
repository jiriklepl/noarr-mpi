#!/bin/bash

set -e

EXECUTABLE=${1:-gemm-mpi}

BUILD_DIR=${BUILD_DIR:-build}
USE_SLURM=${USE_SLURM:-0}
NUM_TASKS=${NUM_TASKS:-$(nproc)}
NUM_NODES=${NUM_NODES:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}

# make NUM_TASKS divisible by 4
NUM_TASKS=$((NUM_TASKS - NUM_TASKS % 4))

if [ ! -f "${BUILD_DIR}/${EXECUTABLE}" ]; then
  echo "Executable not found. Run build.sh first." >&2
  exit 1
fi

case "${EXECUTABLE}" in
  *-mpi*)
	if [ "${USE_SLURM}" -eq 1 ]; then
	  srun -n "${NUM_TASKS}" -N "${NUM_NODES}" -c "${CPUS_PER_TASK}" -- "${BUILD_DIR}/${EXECUTABLE}"
	elif [ "${NUM_NODES}" -gt 1 ] || [ "${CPUS_PER_TASK}" -gt 1 ]; then
	  echo "For multi-node or multi-cpu-per-task runs, use SLURM." >&2
	  exit 1
	else
	  mpirun -N "${NUM_TASKS}" --host="$(hostname):${NUM_TASKS}" "${BUILD_DIR}/${EXECUTABLE}"
	fi
	;;
  *)
	"${BUILD_DIR}/${EXECUTABLE}"
	;;
esac
