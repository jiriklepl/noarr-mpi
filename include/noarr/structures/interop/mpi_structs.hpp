#ifndef NOARR_STRUCTURES_INTEROP_MPI_STRUCTS_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_STRUCTS_HPP

#include <mpi.h>

#include <noarr/structures/base/contain.hpp>
#include <noarr/structures/structs/setters.hpp>

#include "../interop/mpi_utility.hpp"

namespace noarr {

template<IsDim auto Dim>
inline auto mpi_bind(MPI_Comm comm) {
	int rank = 0;
	int size = 0;

	MPICHK(MPI_Comm_rank(comm, &rank));
	MPICHK(MPI_Comm_size(comm, &size));

	return set_length<Dim>(size) ^ fix<Dim>(rank);
}

template<IsDim auto Dim>
inline auto mpi_fix(MPI_Comm comm) {
	int rank = 0;

	MPICHK(MPI_Comm_rank(comm, &rank));

	return fix<Dim>(rank);
}

template<IsDim auto Dim, auto MajorDim = dim<[]() {}>{}>
inline auto mpi_block(MPI_Comm comm) {
	int rank = 0;
	int size = 0;

	MPICHK(MPI_Comm_rank(comm, &rank));
	MPICHK(MPI_Comm_size(comm, &size));

	return into_blocks<Dim, MajorDim>(size) ^ fix<MajorDim>(rank);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_STRUCTS_HPP
