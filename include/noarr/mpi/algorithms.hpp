#ifndef NOARR_MPI_ALGORITHMS_HPP
#define NOARR_MPI_ALGORITHMS_HPP

#include <cstddef>

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <noarr/structures/base/contain.hpp>
#include <noarr/structures/extra/shortcuts.hpp>

#include "../mpi/bag.hpp"
#include "../mpi/transform.hpp"
#include "../mpi/traverser.hpp"
#include "../mpi/utility.hpp"

namespace noarr::mpi {

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

inline void broadcast(const ToStruct auto &has_struct, const IsMpiTraverser auto &trav, int rank) {
	const auto structure = convert_to_struct(has_struct);
	auto type = mpi_transform(trav, structure);
	type.commit();
	MPICHK(MPI_Bcast(has_struct.data(), 1, convert_to_MPI_Datatype(type), rank, convert_to_MPI_Comm(trav)));
}

template<class T>
inline void broadcast(T &scalar, const ToMPIComm auto &has_comm, int rank)
requires choose_mpi_type<T>::value
{
	MPICHK(MPI_Bcast(&scalar, 1, choose_mpi_type<T>::get(), rank, convert_to_MPI_Comm(has_comm)));
}

inline void barrier(const ToMPIComm auto &has_comm) { MPICHK(MPI_Barrier(convert_to_MPI_Comm(has_comm))); }

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
inline mpi_comm_guard mpi_comm_split_along(const MPITraverser &traverser) {
	static_assert(dim_sequence<AllDims...>::template contains<AlongDim>,
	              "The dimension must be present in the sequence");

	const auto state = traverser.state().items_restrict(
		typename noarr::helpers::state_filter_items<typename decltype(traverser.state())::items_pack,
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

namespace helpers {

class no_type_cache {};

inline decltype(auto) scatter_types(auto &&cache, const auto &gathered, const auto &scattered,
                                    const IsMpiTraverser auto &trav, int min_rank) {
	const auto gathered_struct = convert_to_struct(gathered);
	const auto scattered_struct = convert_to_struct(scattered);

	using gathered_sig = typename decltype(gathered_struct ^ set_length(trav))::signature;
	using scattered_sig = typename decltype(scattered_struct ^ set_length(trav))::signature;
	using trav_sig = typename decltype(trav.top_struct())::signature;
	using gathered_bound_sig = typename decltype(gathered_struct ^ set_length(trav) ^ fix(trav))::signature;

	using gathered_dim_tree = sig_dim_tree<gathered_sig>;
	using scattered_dim_tree = sig_dim_tree<scattered_sig>;
	using trav_dim_tree = sig_dim_tree<trav_sig>;
	using gathered_bound_dim_tree = sig_dim_tree<gathered_bound_sig>;

	using scattered_dim_filtered = dim_tree_filter<scattered_dim_tree, in_signature<gathered_sig>>;
	using scattered_dim_removed = dim_tree_filter<scattered_dim_tree, dim_pred_not<in_signature<gathered_sig>>>;
	using gathered_dim_filtered = dim_tree_filter<gathered_dim_tree, in_signature<scattered_sig>>;

	using scattered_dim_in_trav = dim_tree_filter<scattered_dim_tree, in_signature<trav_sig>>;
	using gathered_dim_in_trav = dim_tree_filter<gathered_bound_dim_tree, in_signature<trav_sig>>;

	using trav_dim_in_gathered = dim_tree_filter<trav_dim_tree, in_signature<gathered_sig>>;
	using trav_dim_in_scattered = dim_tree_filter<trav_dim_tree, in_signature<scattered_sig>>;

	// scattered must be a subset of gathered
	static_assert(std::is_same_v<scattered_dim_filtered, scattered_dim_tree> &&
	                  !std::is_same_v<scattered_dim_filtered, dim_sequence<>> &&
	                  std::is_same_v<scattered_dim_removed, dim_sequence<>>,
	              R"(The "scattered" structure must be a nontrivial subset of the "gathered" structure)");

	static_assert(
		std::is_same_v<gathered_dim_filtered, gathered_dim_in_trav> &&
			std::is_same_v<scattered_dim_tree, scattered_dim_in_trav>,
		R"(The traverser must contain all dimensions of the "gathered" structure and bind the difference in the index spaces)");

	static_assert(
		std::is_same_v<trav_dim_in_gathered, trav_dim_in_scattered>,
		R"(The traverser must contain the same dimensions for both the "gathered" and "scattered" structures)");

	if constexpr (std::is_same_v<std::remove_cvref_t<decltype(cache)>, no_type_cache>) {
		const auto gathered_rep = mpi_transform(gathered_struct, scattered_dim_filtered{}, trav.state(min_rank));
		auto scattered_rep = mpi_transform(scattered_struct, scattered_dim_filtered{}, trav.state(min_rank));
		scattered_rep.commit();

		MPI_Datatype gathered_rep_resized = MPI_DATATYPE_NULL;
		MPICHK(MPI_Type_create_resized(convert_to_MPI_Datatype(gathered_rep), 0, sizeof(char), &gathered_rep_resized));
		auto gathered_rep_resized_custom = MPI_custom_type(gathered_rep_resized);
		gathered_rep_resized_custom.commit();

		return std::tuple<MPI_custom_type, MPI_custom_type>{std::move(gathered_rep_resized_custom),
		                                                    std::move(scattered_rep)};
	} else {
		decltype(auto) gathered_rep =
			mpi_transform(cache, gathered_struct, scattered_dim_filtered{}, trav.state(min_rank));
		decltype(auto) scattered_rep =
			mpi_transform(cache, scattered_struct, scattered_dim_filtered{}, trav.state(min_rank));

		MPI_Datatype gathered_rep_resized = MPI_DATATYPE_NULL;
		MPICHK(MPI_Type_create_resized(convert_to_MPI_Datatype(gathered_rep), 0, sizeof(char), &gathered_rep_resized));
		auto gathered_rep_resized_custom = MPI_custom_type(gathered_rep_resized);
		gathered_rep_resized_custom.commit();
		return std::tuple<MPI_custom_type, MPI_custom_type &>{std::move(gathered_rep_resized_custom),
		                                                      std::ref(scattered_rep)};
	}
}

inline std::vector<int> scatter_displacements(const auto &gathered, const IsMpiTraverser auto &trav) {
	const auto gathered_struct = convert_to_struct(gathered);

	const auto comm = convert_to_MPI_Comm(trav);
	const int comm_size = mpi_get_comm_size(comm);

	using gathered_sig = typename decltype(gathered_struct ^ set_length(trav))::signature;
	using gathered_dim_tree = sig_dim_tree<gathered_sig>;

	auto displacements = std::vector<int>(static_cast<std::size_t>(comm_size));

	for (int i = 0; i < comm_size; ++i) {
		const auto state = trav.state(i);
		const auto offset_getter = offset(fix_zeros(gathered_dim_tree{}));
		displacements[i] = (gathered_struct ^ set_length(state) ^ fix(state)) | offset_getter;
	}

	return displacements;
}

} // namespace helpers

template<class TypeCache = helpers::no_type_cache>
inline void scatter(const auto &gathered, const auto &scattered, const IsMpiTraverser auto &trav, int root,
                    TypeCache &&type_cache = {}) {
	const auto comm = convert_to_MPI_Comm(trav);
	const int comm_size = mpi_get_comm_size(comm);

	const auto sendcounts = std::vector<int>(comm_size, 1);

	const auto displacements = helpers::scatter_displacements(gathered, trav);
	const auto min_rank = std::min_element(displacements.begin(), displacements.end()) - displacements.begin();

	const auto &[gathered_rep, scattered_rep] =
		helpers::scatter_types(std::forward<TypeCache>(type_cache), gathered, scattered, trav, min_rank);

	MPICHK(MPI_Scatterv(gathered.data(), sendcounts.data(), displacements.data(), convert_to_MPI_Datatype(gathered_rep),
	                    scattered.data(), 1, convert_to_MPI_Datatype(scattered_rep), root, comm));
}

template<class TypeCache = helpers::no_type_cache>
inline void gather(const auto &scattered, const auto &gathered, const IsMpiTraverser auto &trav, int root,
                   TypeCache &&type_cache = {}) {
	const auto comm = convert_to_MPI_Comm(trav);
	const int comm_size = mpi_get_comm_size(comm);

	const auto sendcounts = std::vector<int>(comm_size, 1);

	const auto displacements = helpers::scatter_displacements(gathered, trav);
	const auto min_rank = std::min_element(displacements.begin(), displacements.end()) - displacements.begin();

	const auto &[gathered_rep, scattered_rep] =
		helpers::scatter_types(std::forward<TypeCache>(type_cache), gathered, scattered, trav, min_rank);

	MPICHK(MPI_Gatherv(scattered.data(), 1, convert_to_MPI_Datatype(scattered_rep), gathered.data(), sendcounts.data(),
	                   displacements.data(), convert_to_MPI_Datatype(gathered_rep), root, comm));
}

} // namespace noarr::mpi

#endif // NOARR_MPI_ALGORITHMS_HPP
