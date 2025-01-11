#ifndef NOARR_STRUCTURES_INTEROP_MPI_ALGORITHMS_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_ALGORITHMS_HPP

#include <mpi.h>

#include "../interop/mpi_bag.hpp"
#include "../interop/mpi_transform.hpp"
#include "../interop/mpi_traverser.hpp"

namespace noarr {

template<class... Bags>
requires (... && IsBag<Bags>)
constexpr auto mpi_run(IsMpiTraverser auto trav, const Bags &...bags) {
	return [trav,
	        ... custom_types =
	            mpi_transform_builder{}.process(bags.structure() ^ fix(trav.state()) ^ set_length(trav.state())),
	        ... bags = bags.get_ref()](auto &&F) {
		trav | for_dims<>(
				   [=, &F, ... types = MPI_Datatype(custom_types)](auto inner) { F(inner, mpi_bag(bags, types)...); });
	};
}

template<auto Dim, class... Bags>
requires (IsDim<decltype(Dim)> && ... && IsBag<Bags>)
constexpr auto mpi_for(IsMpiTraverser auto trav, const Bags &...bags) {
	const auto comm = trav.get_comm();

	return [trav, comm, ... custom_types = mpi_transform_builder{}.process(bags.structure()), ... bags = bags.get_ref(),
	        // privatized bags
	        ... privatized_structs = vectors_like(bags.structure())](auto &&init, auto &&for_each, auto &&finalize) {
		trav | for_dims<>([=, &init, &for_each, &finalize, ... types = MPI_Datatype(custom_types)](auto inner) {
			init(inner, mpi_bag(bags, types)...);

			for_each(inner, mpi_bag(bags, types)...); // TODO: not like this...

			finalize(inner, mpi_bag(bags, types)...);
		});
	};
}

inline void mpi_bcast(auto structure, ToMPIComm auto has_comm, int rank) {
	MPICHK(MPI_Bcast(structure.data(), 1, structure.get_mpi_type(), rank, convert_to_MPI_Comm(has_comm)));
}

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_ALGORITHMS_HPP
