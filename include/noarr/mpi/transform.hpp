#ifndef NOARR_MPI_TRANSFORM_HPP
#define NOARR_MPI_TRANSFORM_HPP

#include <cassert>
#include <cstddef>

#include <atomic>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <mpi.h>

#include <noarr/structures/base/utility.hpp>
#include <noarr/structures/extra/sig_utils.hpp>
#include <noarr/structures/extra/struct_traits.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/introspection/lower_bound_along.hpp>
#include <noarr/structures/introspection/stride_along.hpp>
#include <noarr/structures/introspection/uniform_along.hpp>

#include "../mpi/utility.hpp"

namespace noarr::mpi {

namespace helpers {

enum class structure_type_id : std::size_t {};
enum class dim_tree_type_id : std::size_t {};
enum class state_type_id : std::size_t {};
enum class structure_id : std::size_t {};
enum class state_id : std::size_t {};

template<class FlexibleContain>
class structure_map;

template<class... Ts>
class structure_map<flexible_contain<Ts...>> {
	struct contain_compare {
		template<class... Ts_>
		constexpr bool less(const flexible_contain<Ts_...> &lhs, const flexible_contain<Ts_...> &rhs) const {
			return [this]<std::size_t... Is>(const auto &lhs, const auto &rhs, std::index_sequence<Is...>) {
				return (... && !less(rhs.template get<Is>(), lhs.template get<Is>())) &&
				       (... || less(lhs.template get<Is>(), rhs.template get<Is>()));
			}(lhs, rhs, std::index_sequence_for<Ts_...>{});
		}

		template<class T>
		constexpr bool less(const T & /* lhs */, const T & /* rhs */) const
		requires std::is_empty_v<T>
		{
			return false;
		}

		constexpr bool less(const auto &lhs, const auto &rhs) const
		requires requires {
			{ lhs < rhs } -> std::convertible_to<bool>;
		}
		{
			return lhs < rhs;
		}
	};

	struct key_less {
		template<class... Ts_>
		constexpr bool operator()(const flexible_contain<Ts_...> &lhs, const flexible_contain<Ts_...> &rhs) const {
			return contain_compare{}.less(lhs, rhs);
		}
	};

public:
	using key = flexible_contain<Ts...>;
	using value = std::size_t;

	structure_id get(const key &k) { return static_cast<structure_id>(map.try_emplace(k, map.size()).first->second); }

private:
	std::map<key, value, key_less> map;
};

template<class State>
class state_map;

template<class... StateItems>
class state_map<state<StateItems...>> {
public:
	using key = state<StateItems...>;
	using value = std::size_t;

	state_id get(const key &k) { return static_cast<state_id>(map.try_emplace(k, map.size()).first->second); }

private:
	std::map<state<StateItems...>, std::size_t> map;
};

// type_cache is not thread-safe
class type_cache {
public:
	using key = std::tuple<structure_type_id, dim_tree_type_id, state_type_id, structure_id, state_id>;
	using value = MPI_custom_type;

	value *get(const auto &structure, const auto &dim_tree, const auto &state) {
		return get(key{get_structure_type_id(structure), get_dim_tree_type_id(dim_tree), get_state_type_id(state),
		               get_structure_id(structure), get_state_id(state)});
	}

	value &set(const auto &structure, const auto &dim_tree, const auto &state, value v) {
		return set(key{get_structure_type_id(structure), get_dim_tree_type_id(dim_tree), get_state_type_id(state),
		               get_structure_id(structure), get_state_id(state)},
		           std::move(v));
	}

private:
	value *get(const key &k) {
		auto it = cache.find(k);

		if (it == cache.end()) {
			return nullptr;
		}

		return &it->second;
	}

	value &set(key k, value v) { return cache.try_emplace(std::move(k), std::move(v)).first->second; }

	std::atomic<std::size_t> structure_type_counter_{0};
	std::atomic<std::size_t> dim_tree_type_counter_{0};
	std::atomic<std::size_t> state_type_counter_{0};

	structure_type_id get_structure_type_id(auto && /*unused*/) {
		static const auto id = structure_type_counter_.fetch_add(1);
		return static_cast<structure_type_id>(id);
	}

	dim_tree_type_id get_dim_tree_type_id(auto && /*unused*/) {
		static const auto id = dim_tree_type_counter_.fetch_add(1);
		return static_cast<dim_tree_type_id>(id);
	}

	state_type_id get_state_type_id(auto && /*unused*/) {
		static const auto id = state_type_counter_.fetch_add(1);
		return static_cast<state_type_id>(id);
	}

	template<class... Ts>
	static structure_id get_structure_id(const flexible_contain<Ts...> &structure) {
		static auto map = structure_map<flexible_contain<Ts...>>{};
		return map.get(structure);
	}

	template<class... StateItems>
	static state_id get_state_id(const state<StateItems...> &s) {
		static auto map = state_map<state<StateItems...>>{};
		return map.get(s);
	}

	std::map<key, value> cache;
};

template<class Structure, IsState State>
inline MPI_custom_type mpi_transform_impl(const Structure &structure, const dim_sequence<> & /*ds*/, State state) {
	using scalar_type = scalar_t<Structure, State>;

	constexpr bool has_offset = has_offset_of<scalar<scalar_type>, Structure, State>();

	if constexpr (has_offset) {
		const auto offset = offset_of<scalar<scalar_type>>(structure, state);
		const auto datatype = choose_mpi_type_v<scalar_type>();

		if (offset == 0) {
			return MPI_custom_type{datatype, false};
		}

		const int block_lengths = 1;
		const auto offsets = static_cast<MPI_Aint>(offset);

		MPI_Datatype new_Datatype = MPI_DATATYPE_NULL;
		MPICHK(MPI_Type_create_hindexed(1, &block_lengths, &offsets, datatype, &new_Datatype));

		return MPI_custom_type{new_Datatype};
	} else {
		throw std::runtime_error("Unsupported: no offset");
	}
}

template<class Structure, auto Dim, class Branches, IsState State>
inline MPI_custom_type mpi_transform_impl(const Structure &structure, const dim_tree<Dim, Branches> & /*dt*/,
                                          State state)
requires (Structure::signature::template any_accept<Dim>)
{
	constexpr bool has_lower_bound = HasLowerBoundAlong<Structure, Dim, State>;
	constexpr bool has_stride_along = HasStrideAlong<Structure, Dim, State>;
	constexpr bool is_uniform_along = IsUniformAlong<Structure, Dim, State>;
	constexpr bool has_length = Structure::template has_length<Dim, State>();

	if constexpr (has_lower_bound && has_stride_along && is_uniform_along && has_length) {
		const auto lower_bound = lower_bound_along<Dim>(structure, state);
		const auto lb_at = lower_bound_at<Dim>(structure, state);
		const auto stride = stride_along<Dim>(structure, state);
		const auto length = structure.template length<Dim>(state);

		MPI_custom_type sub_transformed = mpi_transform_impl(structure, Branches{}, state.template with<index_in<Dim>>(lb_at));

		if (lb_at != 0) {
			assert(lb_at == length - 1 && "lower bound is not at the beginning or end of the dimension");
			assert(stride < 0 && "stride must be negative when lower bound is at the end of the dimension");
			std::vector<MPI_Aint> offsets(length);
			const std::vector<int> block_lengths(length, 1);
			for (std::size_t i = 0; i < length; ++i) {
				offsets[i] = (length - 1 - i) * -stride;
			}

			MPI_Datatype new_Datatype = MPI_DATATYPE_NULL;
			MPICHK(MPI_Type_create_hindexed(length, block_lengths.data(), offsets.data(), (MPI_Datatype)sub_transformed, &new_Datatype));
			sub_transformed.reset(new_Datatype);
		} else if (stride > 0 && length != 1) {
			MPI_Datatype new_Datatype = MPI_DATATYPE_NULL;
			MPICHK(MPI_Type_create_hvector(length, 1, stride, (MPI_Datatype)sub_transformed, &new_Datatype));
			sub_transformed.reset(new_Datatype);
		}

		if (lower_bound != 0) {
			const auto lb = static_cast<MPI_Aint>(lower_bound);
			MPI_Datatype new_Datatype = MPI_DATATYPE_NULL;
			const int block_lengths = 1;
			MPICHK(MPI_Type_create_hindexed(1, &block_lengths, &lb, (MPI_Datatype)sub_transformed, &new_Datatype));
			sub_transformed.reset(new_Datatype);
		}

		return sub_transformed;
	} else {
		if constexpr (!has_lower_bound) {
			throw std::runtime_error("Unsupported: lower bound cannot be determined");
		} else if constexpr (!has_stride_along) {
			throw std::runtime_error("Unsupported: stride cannot be determined");
		} else if constexpr (!is_uniform_along) {
			throw std::runtime_error("Unsupported: structure is not uniform along the dimension");
		} else if constexpr (!has_length) {
			throw std::runtime_error("Unsupported: length cannot be determined");
		} else {
			throw std::runtime_error("Unsupported transformation");
		}
	}
}

template<class Structure, auto Dim, class Branches, IsState State>
inline MPI_custom_type mpi_transform_impl(const Structure &structure, const dim_tree<Dim, Branches> & /*dt*/,
                                          State state)
requires (!Structure::signature::template any_accept<Dim>)
{
	return mpi_transform_impl(structure, Branches{}, state);
}

} // namespace helpers

using type_cache = helpers::type_cache;

template<class Structure, class DimTree, IsState State>
inline auto mpi_transform(const Structure &structure, const DimTree &dim_tree, State state) {
	return helpers::mpi_transform_impl(structure, dim_tree, state);
}

template<ToTraverser Trav, class Structure>
constexpr auto mpi_transform(const Trav &trav, const Structure &structure) {
	using dim_tree = sig_dim_tree<typename decltype(trav.top_struct())::signature>;

	return mpi_transform(structure, dim_tree{}, trav.state());
}

template<ToTraverser Structure, class Trav>
constexpr auto mpi_transform(type_cache &cache, const Structure &structure, const Trav &trav) {
	using dim_tree = sig_dim_tree<typename decltype(trav.top_struct())::signature>;

	return mpi_transform(cache, structure, dim_tree{}, trav.state());
}

template<class Structure, class DimTree, IsState State>
inline MPI_custom_type &mpi_transform(type_cache &cache, const Structure &structure, const DimTree &dim_tree,
                                      State state) {
	auto type = cache.get(structure, dim_tree, state);

	if (type != nullptr) {
		return *type;
	}

	type = &cache.set(structure, dim_tree, state, mpi_transform(structure, dim_tree, state));
	type->commit();

	return *type;
}

template<ToTraverser Trav, class Structure, class... Structures>
constexpr auto mpi_transform(const Trav &trav, const Structure &structure, const Structures &...structs) {
	return std::make_tuple(mpi_transform(trav, structure), mpi_transform(trav, structs)...);
}

template<ToTraverser Trav, class Structure, class... Structures>
constexpr auto mpi_transform(type_cache &cache, const Trav &trav, const Structure &structure,
                             const Structures &...structs) {
	return std::tuple(mpi_transform(cache, structure, trav.state(), structure),
	                  mpi_transform(cache, structs, trav.state(), structs)...);
}

template<class Structure>
constexpr auto mpi_transform(const Structure &structure) {
	return mpi_transform(traverser(structure), structure);
}

template<class Structure>
constexpr auto mpi_transform(type_cache &cache, const Structure &structure) {
	return mpi_transform(cache, traverser(structure), structure);
}

} // namespace noarr::mpi

#endif // NOARR_MPI_TRANSFORM_HPP
