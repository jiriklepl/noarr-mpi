#ifndef NOARR_STRUCTURES_INTEROP_MPI_ALGORITHMS_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_ALGORITHMS_HPP

#include <utility>

#include <mpi.h>

#include <noarr/structures/extra/shortcuts.hpp>

#include "../interop/mpi_bag.hpp"
#include "../interop/mpi_transform.hpp"
#include "../interop/mpi_traverser.hpp"
#include "noarr/structures/interop/mpi_utility.hpp"

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

inline void mpi_bcast(auto structure, const ToMPIComm auto &has_comm, int rank) {
	MPICHK(MPI_Bcast(structure.data(), 1, structure.get_mpi_type(), rank, convert_to_MPI_Comm(has_comm)));
}

inline void mpi_barrier(const ToMPIComm auto &has_comm) { MPICHK(MPI_Barrier(convert_to_MPI_Comm(has_comm))); }

inline std::pair<MPI_Aint, MPI_Aint> mpi_type_get_extent(const ToMPIDatatype auto &has_type) {
	std::pair<MPI_Aint, MPI_Aint> result;

	const auto type = convert_to_MPI_Datatype(has_type);

	auto &[lb, extent] = result;
	MPICHK(MPI_Type_get_extent(type, &lb, &extent));

	return result;
}

template<class T>
struct is_length_in : std::false_type {};

template<auto Dim>
requires IsDim<decltype(Dim)>
struct is_length_in<length_in<Dim>> : std::true_type {};

template<class T>
struct is_index_in : std::false_type {};

template<auto Dim>
requires IsDim<decltype(Dim)>
struct is_index_in<index_in<Dim>> : std::true_type {};

template<class T>
concept IsIndexIn = is_index_in<std::remove_cvref_t<T>>::value;

namespace helpers {

template<IsDim auto... Dims>
struct remove_indices {
	template<class Tag>
	static constexpr bool value = !IsIndexIn<Tag> || !(... || (Tag::dims::template contains<Dims>));
};

} // namespace helpers

template<auto AlongDim, auto... AllDims, class MPITraverser>
requires (IsDim<decltype(AlongDim)> && ... && IsDim<decltype(AllDims)>) && IsMpiTraverser<MPITraverser>
inline auto mpi_comm_split_along(MPITraverser traverser) -> mpi_comm_guard {
	static_assert(dim_sequence<AllDims...>::template contains<AlongDim>,
	              "The dimension must be present in the sequence");

	const auto state = traverser.state().items_restrict(
		typename helpers::state_filter_items<typename decltype(traverser.state())::items_pack,
	                                         helpers::remove_indices<AllDims...>>::result());
	const auto space = noarr::scalar<char>() ^ noarr::vectors_like<AllDims...>(traverser.get_struct(), state);
	const auto comm = traverser.get_comm();

	MPI_Comm new_comm = MPI_COMM_NULL;
	MPICHK(MPI_Comm_split(comm,
	                      space | noarr::offset(noarr::filter_indices<AllDims...>(
									  traverser.state() - noarr::filter_indices<AlongDim>(traverser.state()))),
	                      noarr::get_index<AlongDim>(traverser), &new_comm));

	return mpi_comm_guard{new_comm};
}

inline int mpi_get_rank(const ToMPIComm auto &has_comm) {
	int rank = -1;
	MPICHK(MPI_Comm_rank(convert_to_MPI_Comm(has_comm), &rank));
	return rank;
}

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_ALGORITHMS_HPP
