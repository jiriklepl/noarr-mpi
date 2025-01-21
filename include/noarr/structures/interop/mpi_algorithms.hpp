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
	const auto space = scalar<char>() ^ vectors_like<AllDims...>(traverser.get_struct(), state);
	const auto comm = traverser.get_comm();

	MPI_Comm new_comm = MPI_COMM_NULL;
	MPICHK(MPI_Comm_split(comm,
	                      space | offset(filter_indices<AllDims...>(
									  traverser.state() - filter_indices<AlongDim>(traverser.state()))),
	                      get_index<AlongDim>(traverser), &new_comm));

	return mpi_comm_guard{new_comm};
}

inline int mpi_get_comm_rank(const ToMPIComm auto &has_comm) {
	int rank = -1;
	MPICHK(MPI_Comm_rank(convert_to_MPI_Comm(has_comm), &rank));
	return rank;
}

inline int mpi_get_comm_size(const ToMPIComm auto &has_comm) {
	int size = -1;
	MPICHK(MPI_Comm_size(convert_to_MPI_Comm(has_comm), &size));
	return size;
}

inline void mpi_scatter(auto from, auto to, IsMpiTraverser auto traverser, int root) {
	// TODO: make this traverser-aware, which simplifies the code like... a lot
	const auto from_struct = convert_to_struct(from) ^ set_length(traverser.state());
	const auto to_struct = convert_to_struct(to) ^ set_length(traverser.state());
	const auto comm = convert_to_MPI_Comm(traverser);

	using from_dim_tree = sig_dim_tree<typename decltype(from_struct)::signature>;
	using to_dim_tree = sig_dim_tree<typename decltype(to_struct)::signature>;

	// we need gatherv; displacements should cover the difference of the dimension trees

	using to_dim_filtered = dim_tree_filter<to_dim_tree, in_signature<typename decltype(to_struct)::signature>>;
	using to_dim_removed =
		dim_tree_filter<to_dim_tree, dim_pred_not<in_signature<typename decltype(from_struct)::signature>>>;

	// to must be a subset of from
	static_assert(std::is_same_v<to_dim_filtered, to_dim_tree> && std::is_same_v<to_dim_removed, dim_sequence<>>,
	              R"(The "to" structure must be a subset of the "from" structure)");

	// contains the dimension that are present in both structures
	// using from_dim_filtered = dim_tree_filter<from_dim_tree, in_signature<typename decltype(from_struct)::signature>>;

	// contains the dimensions that are present in the "from" structure, but not in the "to" structure
	using from_dim_removed =
		dim_tree_filter<from_dim_tree, dim_pred_not<in_signature<typename decltype(to_struct)::signature>>>;

	const std::size_t difference_size = index_space_size<from_dim_removed>(from_struct);

	std::vector<int> displacements(difference_size);
	const std::vector<int> sendcounts(difference_size, 1);

	const int offset = (from_struct ^ traverser.get_order()) | noarr::offset(fix_zeros(to_dim_filtered{}));

	const auto from_substructure = from_struct ^ fix(fix_zeros(from_dim_removed{}));
	const auto from_rep = mpi_transform_builder{}.process(from_substructure);
	const auto [from_lb, from_extent] = mpi_type_get_extent(from_rep);

	const auto to_rep = mpi_transform_builder{}.process(to_struct);

	assert(from_lb == 0);

	MPICHK(MPI_Allgather(&offset, 1, MPI_INT, displacements.data(), 1, MPI_INT, comm));

	// compute the second smallest displacement
	int min_displacement = std::numeric_limits<int>::max();
	int second_min_displacement = std::numeric_limits<int>::max();
	for (const auto displacement : displacements) {
		if (displacement < min_displacement) {
			second_min_displacement = min_displacement;
			min_displacement = displacement;
		} else if (displacement < second_min_displacement) {
			second_min_displacement = displacement;
		}
	}

	MPI_Datatype from_rep_cheat = MPI_DATATYPE_NULL;
	MPICHK(MPI_Type_create_resized(convert_to_MPI_Datatype(from_rep), 0, second_min_displacement, &from_rep_cheat));
	const MPI_custom_type from_rep_cheat_custom(from_rep_cheat);

	for (auto &displacement : displacements) {
		displacement /= second_min_displacement;
	}

	MPICHK(MPI_Scatterv(from.data(), sendcounts.data(), displacements.data(), convert_to_MPI_Datatype(from_rep_cheat),
	                    to.data(), 1, convert_to_MPI_Datatype(to_rep), root, comm));
}

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_ALGORITHMS_HPP
