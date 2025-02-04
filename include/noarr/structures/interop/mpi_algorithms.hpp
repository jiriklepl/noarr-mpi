#ifndef NOARR_STRUCTURES_INTEROP_MPI_ALGORITHMS_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_ALGORITHMS_HPP

#include <utility>

#include <mpi.h>

#include <noarr/structures/base/contain.hpp>
#include <noarr/structures/extra/shortcuts.hpp>

#include "../interop/mpi_bag.hpp"
#include "../interop/mpi_transform.hpp"
#include "../interop/mpi_traverser.hpp"
#include "../interop/mpi_utility.hpp"

namespace noarr {

namespace helpers {

template<class Trav, class... Bags>
struct mpi_run_t : flexible_contain<Trav, Bags...> {
	using base = flexible_contain<Trav, Bags...>;
	using base::base;

	template<class F>
	constexpr decltype(auto) operator()(F &&f) const {
		return execute_impl(std::make_index_sequence<sizeof...(Bags)>{}, std::forward<F>(f));
	}

	template<class F>
	friend decltype(auto) operator|(const mpi_run_t &run, F &&f) {
		return run(std::forward<F>(f));
	}

private:
	template<std::size_t... Is, class F>
	constexpr decltype(auto) execute_impl(std::index_sequence<Is...> /*is*/, F &&f) const {
		const auto trav = this->template get<0>();
		return std::forward<F>(f)(trav, mpi_bag(this->template get<Is + 1>(),
		                                        mpi_transform(trav, this->template get<Is + 1>().structure()))...);
	}
};

template<class Trav, class... Bags>
mpi_run_t(Trav, Bags...) -> mpi_run_t<Trav, Bags...>;

} // namespace helpers

template<class... Bags>
requires (... && IsBag<Bags>)
constexpr auto mpi_run(IsMpiTraverser auto trav, const Bags &...bags) {
	return helpers::mpi_run_t(trav, bags.get_ref()...);
}

inline void mpi_bcast(const ToStruct auto &has_struct, const IsMpiTraverser auto &trav, int rank) {
	const auto structure = convert_to_struct(has_struct);
	const auto type = mpi_transform(trav, structure);
	MPICHK(MPI_Bcast(has_struct.data(), 1, convert_to_MPI_Datatype(type), rank, convert_to_MPI_Comm(trav)));
}

template<class T>
inline void mpi_bcast(T &scalar, const ToMPIComm auto &has_comm, int rank)
requires choose_mpi_type<T>::value
{
	MPICHK(MPI_Bcast(&scalar, 1, choose_mpi_type<T>::get(), rank, convert_to_MPI_Comm(has_comm)));
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
inline auto mpi_comm_split_along(const MPITraverser &traverser) -> mpi_comm_guard {
	static_assert(dim_sequence<AllDims...>::template contains<AlongDim>,
	              "The dimension must be present in the sequence");

	const auto state = traverser.state().items_restrict(
		typename helpers::state_filter_items<typename decltype(traverser.state())::items_pack,
	                                         helpers::remove_indices<AllDims...>>::result());
	const auto space = scalar<char>() ^ vectors_like<AllDims...>(traverser.get_struct(), state);
	const auto comm = traverser.get_comm();

	MPI_Comm new_comm = MPI_COMM_NULL;
	MPICHK(MPI_Comm_split(
		comm,
		space | offset(filter_indices<AllDims...>(traverser.state() - filter_indices<AlongDim>(traverser.state()))),
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

inline void mpi_scatter(const auto &from, const auto &to, const IsMpiTraverser auto &trav, int root) {
	const auto from_struct = convert_to_struct(from);
	const auto to_struct = convert_to_struct(to);
	const auto comm = convert_to_MPI_Comm(trav);

	using from_dim_tree = sig_dim_tree<typename decltype(from_struct ^ set_length(trav))::signature>;
	using to_dim_tree = sig_dim_tree<typename decltype(to_struct ^ set_length(trav))::signature>;

	using to_dim_filtered =
		dim_tree_filter<to_dim_tree, in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>;
	using to_dim_removed =
		dim_tree_filter<to_dim_tree,
	                    dim_pred_not<in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>>;

	using from_dim_filtered =
		dim_tree_filter<from_dim_tree, in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>;
	using from_dim_removed =
		dim_tree_filter<from_dim_tree,
	                    dim_pred_not<in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>>;

	// to must be a subset of from
	static_assert(std::is_same_v<to_dim_filtered, to_dim_tree> && std::is_same_v<to_dim_removed, dim_sequence<>>,
	              R"(The "from" structure must be a subset of the "to" structure)");

	// TODO: this is incomplete
	const auto from_rep = mpi_transform_impl(from_struct, to_dim_filtered{}, trav.state(/*TODO*/0));
	const auto to_rep = mpi_transform_impl(to_struct, to_dim_filtered{}, trav.state(/*TODO*/0));

	// TODO: the following may be incorrect
	const auto difference_size = mpi_get_comm_size(comm);

	std::vector<int> displacements(difference_size);
	const std::vector<int> sendcounts(difference_size, 1);

	const int offset = (from_struct ^ set_length(trav) ^ fix(trav)) | noarr::offset(fix_zeros(from_dim_tree{}));

	MPICHK(MPI_Allgather(&offset, 1, MPI_INT, displacements.data(), 1, MPI_INT, comm));

	// compute the second smallest displacement
	int min_displacement = std::numeric_limits<int>::max();
	int second_min_displacement = std::numeric_limits<int>::max();
	for (const auto displacement : displacements) {
		if (displacement < min_displacement) {
			second_min_displacement = min_displacement;
			min_displacement = displacement;
		} else if (displacement < second_min_displacement && displacement != min_displacement) {
			second_min_displacement = displacement;
		}
	}

	MPI_Datatype from_rep_resized = MPI_DATATYPE_NULL;
	MPICHK(MPI_Type_create_resized(convert_to_MPI_Datatype(from_rep), 0, second_min_displacement, &from_rep_resized));
	const MPI_custom_type from_rep_resized_custom(from_rep_resized);

	for (auto &displacement : displacements) {
		displacement /= second_min_displacement;
	}

	MPICHK(MPI_Scatterv(from.data(), sendcounts.data(), displacements.data(), convert_to_MPI_Datatype(from_rep_resized),
	                    to.data(), 1, convert_to_MPI_Datatype(to_rep), root, comm));
}

inline void mpi_gather(const auto &from, const auto &to, const IsMpiTraverser auto &trav, int root) {
	const auto from_struct = convert_to_struct(from);
	const auto to_struct = convert_to_struct(to);
	const auto comm = convert_to_MPI_Comm(trav);

	using from_dim_tree = sig_dim_tree<typename decltype(from_struct ^ set_length(trav))::signature>;
	using to_dim_tree = sig_dim_tree<typename decltype(to_struct ^ set_length(trav))::signature>;

	using to_dim_filtered =
		dim_tree_filter<to_dim_tree, in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>;
	using to_dim_removed =
		dim_tree_filter<to_dim_tree,
	                    dim_pred_not<in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>>;

	using from_dim_filtered =
		dim_tree_filter<from_dim_tree, in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>;
	using from_dim_removed =
		dim_tree_filter<from_dim_tree,
	                    dim_pred_not<in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>>;

	// from must be a subset of to
	static_assert(std::is_same_v<from_dim_filtered, from_dim_tree> && std::is_same_v<from_dim_removed, dim_sequence<>>,
	              R"(The "to" structure must be a subset of the "from" structure)");

	// TODO: this is incomplete
	const auto from_rep = mpi_transform_impl(from_struct, from_dim_filtered{}, trav.state(/*TODO*/0));
	const auto to_rep = mpi_transform_impl(to_struct, from_dim_filtered{}, trav.state(/*TODO*/0));

	// TODO: the following may be incorrect
	const auto difference_size = mpi_get_comm_size(comm);

	std::vector<int> displacements(difference_size);
	const std::vector<int> sendcounts(difference_size, 1);

	const int offset = (to_struct ^ set_length(trav) ^ fix(trav)) | noarr::offset(fix_zeros(to_dim_tree{}));

	MPICHK(MPI_Allgather(&offset, 1, MPI_INT, displacements.data(), 1, MPI_INT, comm));

	// compute the second smallest displacement
	int min_displacement = std::numeric_limits<int>::max();
	int second_min_displacement = std::numeric_limits<int>::max();
	for (const auto displacement : displacements) {
		if (displacement < min_displacement) {
			second_min_displacement = min_displacement;
			min_displacement = displacement;
		} else if (displacement < second_min_displacement && displacement != min_displacement) {
			second_min_displacement = displacement;
		}
	}

	MPI_Datatype to_rep_resized = MPI_DATATYPE_NULL;
	MPICHK(MPI_Type_create_resized(convert_to_MPI_Datatype(to_rep), 0, second_min_displacement, &to_rep_resized));
	const MPI_custom_type to_rep_resized_custom(to_rep_resized);

	for (auto &displacement : displacements) {
		displacement /= second_min_displacement;
	}

	MPICHK(MPI_Gatherv(from.data(), 1, convert_to_MPI_Datatype(from_rep), to.data(), sendcounts.data(),
	                   displacements.data(), convert_to_MPI_Datatype(to_rep_resized), root, comm));
}

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_ALGORITHMS_HPP
