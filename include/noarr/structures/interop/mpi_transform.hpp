#ifndef NOARR_STRUCTURES_INTEROP_MPI_TRANSFORM_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_TRANSFORM_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <array>

#include <mpi.h>

#include <noarr/structures/introspection/lower_bound_along.hpp>
#include <noarr/structures/introspection/stride_along.hpp>
#include <noarr/structures/introspection/uniform_along.hpp>

#include "../interop/mpi_traverser.hpp"
#include "../interop/mpi_utility.hpp"

namespace noarr {

namespace helpers {

// template<auto Dim, class Branches, class Structure>
// auto mpi_transform(const Structure& structure, const dim_tree<Dim, Branches> &/*unused*/) {
//	TODO: implement
// }

template<class Structure, IsState State>
inline auto mpi_transform_impl(const Structure &structure, const dim_sequence<> & /*unused*/, State state)
	-> MPI_custom_type {
	using scalar_type = scalar_t<Structure, State>;

	constexpr bool has_offset = has_offset_of<scalar<scalar_type>, Structure, State>();

	if constexpr (has_offset) {
		const auto offset = offset_of<scalar<scalar_type>>(structure, state);
		const auto datatype = choose_mpi_type_v<scalar_type>();

		MPI_Datatype new_Datatype = MPI_DATATYPE_NULL;
		if (offset == 0) {
			MPICHK(MPI_Type_dup(datatype, &new_Datatype));
		} else {
			const auto block_lengths = std::array{1};
			const auto offsets = std::array{static_cast<MPI_Aint>(offset)};

			MPICHK(MPI_Type_create_hindexed(1, block_lengths.data(), offsets.data(), datatype, &new_Datatype));
		}

		return MPI_custom_type{new_Datatype};
	} else {
		throw std::runtime_error("Unsupported: no offset");
	}
}

template<auto Dim, class Branches, class Structure, IsState State>
inline auto mpi_transform_impl(const Structure &structure, const dim_tree<Dim, Branches> & /*unused*/, State state)
	-> MPI_custom_type
requires (Structure::signature::template any_accept<Dim>)
{
	constexpr bool has_lower_bound = HasLowerBoundAlong<Structure, Dim, State>;
	constexpr bool has_stride_along = HasStrideAlong<Structure, Dim, State>;
	constexpr bool is_uniform_along = IsUniformAlong<Structure, Dim, State>;
	constexpr bool has_length = Structure::template has_length<Dim, State>();

	if constexpr (has_lower_bound && has_stride_along && is_uniform_along && has_length) {
		const auto lower_bound = lower_bound_along<Dim>(structure, state);
		const auto stride = stride_along<Dim>(structure, state);
		const auto length = structure.template length<Dim>(state);

		// TODO: probably wanna add { lower_bound_assumption(structure, state) } -> IsState
		const auto sub_transformed =
			mpi_transform_impl(structure, Branches{}, state.template with<index_in<Dim>>(/*TODO*/ 0));

		if (lower_bound != 0) {
			throw std::runtime_error("Unsupported: lower bound is not zero");
		}

		MPI_Datatype new_Datatype = MPI_DATATYPE_NULL;

		if (stride == 0 || length == 1) {
			MPICHK(MPI_Type_dup((MPI_Datatype)sub_transformed, &new_Datatype));
		} else {
			MPICHK(MPI_Type_create_hvector(length, 1, stride, (MPI_Datatype)sub_transformed, &new_Datatype));
		}

		return MPI_custom_type{new_Datatype};
	} else {
		if constexpr (!has_lower_bound) {
			throw std::runtime_error("Unsupported: lower bound is not set");
		} else if constexpr (!has_stride_along) {
			throw std::runtime_error("Unsupported: stride is not set");
		} else if constexpr (!is_uniform_along) {
			throw std::runtime_error("Unsupported: non-uniform stride");
		} else if constexpr (!has_length) {
			throw std::runtime_error("Unsupported: length is not set for dimension " + std::to_string(Dim));
		} else {
			throw std::runtime_error("Unsupported transformation");
		}
	}
}

template<auto Dim, class Branches, class Structure, IsState State>
inline auto mpi_transform_impl(const Structure &structure, const dim_tree<Dim, Branches> & /*unused*/, State state)
	-> MPI_custom_type
requires (!Structure::signature::template any_accept<Dim>)
{
	return mpi_transform_impl(structure, Branches{}, state);
}

} // namespace helpers

template<class Structure, ToTraverser Trav>
constexpr auto mpi_transform(const Trav &trav, const Structure &structure) {
	using dim_tree = sig_dim_tree<typename decltype(trav.top_struct())::signature>;

	return mpi_transform_impl(structure, dim_tree{}, trav.state());
}

template<class Structure, class... Structures, ToTraverser Trav>
constexpr auto mpi_transform(const Trav &trav, const Structure &structure, const Structures &...structs) {
	return std::make_tuple(mpi_transform(trav, structure), mpi_transform(trav, structs)...);
}

template<class Structure>
constexpr auto mpi_transform(const Structure &structure) {
	return mpi_transform(noarr::traverser(structure), structure);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_TRANSFORM_HPP
