#include <cassert>
#include <expected>
#include <iostream>
#include <typeinfo>

#include <mpi.h>

#include <noarr/introspection.hpp>
#include <noarr/traversers.hpp>

#include "noarr/structures/interop/mpi_algorithms.hpp"
#include "noarr/structures/interop/mpi_traverser.hpp"
#include "noarr/structures/interop/mpi_utility.hpp"

// ---------------------------------------------------------------------------

// for each of the functions, we want `fun(mpi_traverser, ...)`, `mpi_traverser.fun(...)` and `mpi_traverser | fun(...)`
// variants

// 1. the traverser has a partitioned structure as a parameter and is merged into one tiled structure + a communicator
// dimension

// TODO: mpi_for(mpi_traverser, structure : noarr_bag, init, for_each, finalize): distributes the structure across the
// ranks, calls the function, gathers the results

// - init(strucure) on root structure (according to the rank dimension)
// - *scatter(structure, privatized_structure)*
// - for_each(privatized_structure) on privatized structures (according to the rank dimension)
// - *gather(privatized_structure, structure)*
// - finalize(structure) on root structure (according to the rank dimension)

// - new_structure(mpi_traverser, structure : noarr_structure) -> mpi_noarr_structure
// - new_structure(mpi_traverser, structure : noarr_bag) -> mpi_noarr_bag

// - communicate<dims...>(mpi_traverser) -> mpi_communicator<dims...>: creates a communicator for the given dimensions

// TODO: mpi_joinreduce(mpi_traverser, structure : noarr_bag, init, for_each, join) like tbb::parallel_reduce;
// automatically performs a scatter, gather is done by 1-1 sending

// - *allocate(privatized_structure)* or *scatter(structure, privatized_structure)* (depending on the bound dimension)
// - init(privatized_structure) on privatized structure
// - for_each(privatized_structure) on privatized structures
// - *send/receive(received_structure)*
// - join(privatized_structure, received_structure) on two privatized structures

// TODO: parallel_scan?

// EXAMPLES of simple use cases:

// - A * x = b (matrix-vector multiplication)
//   - A is a matrix (in)
//   - x is a vector (in)
//   - b is a vector (out)
//   - we want to distribute the matrix across the ranks and then gather the results
//   - A: (i, j) ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
//   - x: ('j')
//   - b: ('i') ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
// - x . y = z (dot product)
//   - x is a vector (in)
//   - y is a vector (in)
//   - z is a scalar (out)
//   - we want to distribute chunks of x and y across the ranks and then reduce the results
//   - x: ('i') ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
//   - y: ('i') ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
//   - z: scalar<int>() ^ bcast<'I'>() ^ mpi_bind<'I'>(comm)
//   - we wanna use the reduce collective (or the join phase)
// - histogram (counting the number of elements in each bin)
//   - histogram is a vector (out)
//   - data is a vector (in)
//   - we want to distribute the data across the ranks and then reduce the results
//   - data: ('i') ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
//   - histogram: ('h') ^ bcast<'I'>() ^ mpi_bind<'I'>(comm)
//   - we wanna use (the join phase or) the reduce collective
// - parallel sort
//   - data is a vector (in-out)
//   - we want to distribute the data across the ranks and then merge the sorted results
//   - data: ('i') ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
//   - we wanna use the join phase; it merges the sorted results
// - parallel pi calculation; see https://github.com/pmodels/mpich/blob/main/examples/cpi.c
// - spectrogram calculation (essentially a 2D FFT; similar to histogram, but not exactly); see
// https://github.com/jbornschein/mpi4py-examples/blob/master/04-image-spectrogram

// EXAMPLES of more complex use cases:

// - Mandelbrot set calculation

// types of structures used in the abstraction:
// - (in) bags with MPI_Datatype (custom types)
// - (out) bags with MPI_Datatype (custom types)
// - (in-out) bags with MPI_Datatype (custom types)

// related work:

// - Boost.MPI: https://www.boost.org/doc/libs/1_87_0/doc/html/mpi/tutorial.html
// - EMPI: https://cosenza.eu/papers/SalimiBeniCCGRID23.pdf
// - A lightweight C++ MPI library:
// - Towards Modern C++ Language support for MPI

using namespace noarr;

// template<auto Dim, class Branches, class Structure>
// auto new_transform(const Structure& structure, const dim_tree<Dim, Branches> &/*unused*/) {
//	TODO: implement
// }

template<class Structure, IsState State>
auto new_transform_impl(const Structure& structure, const dim_sequence<> &/*unused*/, State state) -> MPI_custom_type {
	// TODO: implement
	using scalar = scalar_t<Structure, State>;
	const auto datatype = choose_mpi_type_v<scalar>();
	std::cerr << "scalar: " << typeid(scalar).name() << " -> " << datatype << std::endl;
	MPI_Datatype new_Datatype = MPI_DATATYPE_NULL;
	MPICHK(MPI_Type_dup(datatype, &new_Datatype));
	std::cerr << "MPI_Type_dup(base: " << datatype << ", derived: " << new_Datatype << ")" << std::endl;
	return MPI_custom_type{new_Datatype};
}

template<auto Dim, class Branches, class Structure, IsState State>
auto new_transform_impl(const Structure& structure, const dim_tree<Dim, Branches> &/*unused*/, State state) -> MPI_custom_type {

	// TODO: constexpr bool contiguous = IsContiguous<Structure, State>;
	constexpr bool has_lower_bound = HasLowerBoundAlong<Structure, Dim, State>;
	constexpr bool has_stride_along = HasStrideAlong<Structure, Dim, State>;
	constexpr bool is_uniform_along = IsUniformAlong<Structure, Dim, State>;
	constexpr bool has_length = Structure::template has_length<Dim, State>();

	if constexpr (has_lower_bound && has_stride_along && is_uniform_along && has_length) {
		const auto lower_bound = lower_bound_along<Dim>(structure, state);
		const auto stride = stride_along<Dim>(structure, state);
		const auto length = structure.template length<Dim>(state);

		// TODO: probably wanna add { lower_bound_assumption(structure, state) } -> IsState
		const MPI_custom_type sub_transformed = new_transform_impl(structure, Branches{}, state);

		if (lower_bound != 0) {
			throw std::runtime_error("Unsupported: lower bound is not zero");
		}

		MPI_Datatype new_Datatype = MPI_DATATYPE_NULL;

		if (stride == 0 || length == 1) {
			MPICHK(MPI_Type_dup((MPI_Datatype)sub_transformed, &new_Datatype));
			std::cerr << "MPI_Type_dup(base: " << (MPI_Datatype)sub_transformed << ", derived: " << new_Datatype << ")" << std::endl;
		} else {
			MPICHK(MPI_Type_create_hvector(length, 1, stride, (MPI_Datatype)sub_transformed, &new_Datatype));
			std::cerr << "MPI_Type_create_hvector(" << length << ", 1, " << stride << ", base: " << (MPI_Datatype)sub_transformed
					  << ", derived: " << new_Datatype << ")" << std::endl;
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

template<class Structure, IsTraverser Trav>
auto new_transform(const Trav& trav, const Structure& structure) {
	using dim_tree = sig_dim_tree<typename decltype(trav.top_struct())::signature>;

	return new_transform_impl(structure, dim_tree{}, trav.state());
}

template<class... Structures, IsTraverser Trav>
auto new_transform(const Trav& trav, const Structures &... structs) {
	return std::make_tuple(new_transform(trav, structs)...);
}

void new_scatter(const auto& from, const auto& to, const IsMpiTraverser auto& trav, int root) {
	const auto from_struct = convert_to_struct(from);
	const auto to_struct = convert_to_struct(to);
	const auto comm = convert_to_MPI_Comm(trav);

	using from_dim_tree = sig_dim_tree<typename decltype(from_struct ^ set_length(trav))::signature>;
	using to_dim_tree = sig_dim_tree<typename decltype(to_struct ^ set_length(trav))::signature>;

	using to_dim_filtered = dim_tree_filter<to_dim_tree, in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>;
	using to_dim_removed =
		dim_tree_filter<to_dim_tree, dim_pred_not<in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>>;

	using from_dim_filtered = dim_tree_filter<from_dim_tree, in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>;
	using from_dim_removed =
		dim_tree_filter<from_dim_tree, dim_pred_not<in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>>;

	// to must be a subset of from
	static_assert(std::is_same_v<to_dim_filtered, to_dim_tree> && std::is_same_v<to_dim_removed, dim_sequence<>>,
	              R"(The "from" structure must be a subset of the "to" structure)");

	// TODO: this is incomplete
	const auto from_rep = new_transform_impl(from_struct, from_dim_filtered{}, trav.state());
	const auto to_rep = new_transform_impl(to_struct, to_dim_filtered{}, trav.state());

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
	                    to.data(), 1, convert_to_MPI_Datatype(to), root, comm));
}


void new_gather(const auto& from, const auto& to, const IsMpiTraverser auto& trav, int root) {
	const auto from_struct = convert_to_struct(from);
	const auto to_struct = convert_to_struct(to);
	const auto comm = convert_to_MPI_Comm(trav);

	using from_dim_tree = sig_dim_tree<typename decltype(from_struct ^ set_length(trav))::signature>;
	using to_dim_tree = sig_dim_tree<typename decltype(to_struct ^ set_length(trav))::signature>;

	using to_dim_filtered = dim_tree_filter<to_dim_tree, in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>;
	using to_dim_removed =
		dim_tree_filter<to_dim_tree, dim_pred_not<in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>>;

	using from_dim_filtered = dim_tree_filter<from_dim_tree, in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>;
	using from_dim_removed =
		dim_tree_filter<from_dim_tree, dim_pred_not<in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>>;

	// from must be a subset of to
	static_assert(std::is_same_v<from_dim_filtered, from_dim_tree> && std::is_same_v<from_dim_removed, dim_sequence<>>,
	              R"(The "to" structure must be a subset of the "from" structure)");

	// TODO: this is incomplete
	const auto from_rep = new_transform_impl(from_struct, from_dim_filtered{}, trav.state());
	const auto to_rep = new_transform_impl(to_struct, to_dim_filtered{}, trav.state());

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

	MPICHK(MPI_Gatherv(from.data(), 1, convert_to_MPI_Datatype(from), to.data(), sendcounts.data(),
	                   displacements.data(), convert_to_MPI_Datatype(to_rep_resized), root, comm));
}

auto main(int argc, char **argv) -> int {
	const noarr::MPI_session mpi_session(argc, argv);

	using namespace noarr;
	// to be shadowed by a local definition of order

	// volatile std::size_t x = 300;
	// volatile std::size_t y = 321;
	// volatile std::size_t z = 801;
	const std::size_t x = 2;
	const std::size_t y = 2;
	const std::size_t z = 2;

	std::cerr << "x: " << x << ", y: " << y << ", z: " << z << '\n';
	std::cerr << "Size: " << x * y * z << '\n';

	const int rank = mpi_get_comm_rank(mpi_session);

	auto data_input = noarr::scalar<int>() ^ noarr::vectors<'x', 'y', 'z'>(2 * x, 2 * y, 2 * z);

	// split the data into blocks
	auto data_structure = data_input ^ noarr::into_blocks<'x', 'X'>() ^ noarr::into_blocks<'y', 'Y'>() ^
	                      noarr::into_blocks<'z', 'Z'>() ^ noarr::set_length<'X', 'Y'>(2, 2);

	// data on the root rank
	auto data_blob =
		(rank == 0) ? std::make_unique<char[]>(data_structure | noarr::get_size()) : std::unique_ptr<char[]>{};

	// bind the structure to MPI_COMM_WORLD
	auto pre_trav =
		noarr::traverser(data_structure) ^ noarr::merge_blocks<'X', 'Y', 'r'>() ^ noarr::merge_blocks<'r', 'Z', 'r'>();
	auto trav = noarr::mpi_traverser<'r'>(pre_trav, MPI_COMM_WORLD);
	// const noarr::MPI_custom_type mpi_rep = mpi_transform_builder{}.process(block.structure());

	// privatize a block corresponding to a single MPI rank
	auto block = noarr::bag(noarr::scalar<int>() ^ noarr::vectors_like<'x', 'y', 'z'>(trav.top_struct()));
	// auto block2 = noarr::bag(noarr::scalar<int>() ^ noarr::vectors_like<'y', 'z', 'x'>(trav.top_struct()));
	// std::cerr << block.structure().size(empty_state) << '\n';

	auto data = noarr::bag(data_structure, data_blob.get());

	mpi_barrier(trav);

	// if (rank == 0) {
	// 	auto block_layout = new_transform(convert_to_traverser(trav), block.structure());
	// 	auto block2_layout = new_transform(convert_to_traverser(trav), block2.structure());
	// 	int counter = 0;
	// 	traverser(block) ^ hoist<'x', 'y', 'z'>() | [&](auto state) {
	// 		block[state] = counter++;
	// 	};

	// 	MPICHK(MPI_Sendrecv(block.data(), 1, (MPI_Datatype)block_layout, 0, 0,
	// 	block2.data(), 1, (MPI_Datatype)block2_layout, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

	// 	traverser(block2) ^ hoist<'x', 'y', 'z'>() | [&](auto state) {
	// 		std::cerr << block2[state] << '\n';
	// 	};
	// }
	// return 0;

	// MPI_Aint lb = 0;
	// MPI_Aint extent = 0;

	// MPICHK(MPI_Type_get_extent((MPI_Datatype)mpi_rep, &lb, &extent));
	// std::cerr << "Extent: " << extent << '\n';
	// std::cerr << "Lower bound: " << lb << '\n';

	// TODO `block -> b` is not the most elegant solution
	// mpi_run(trav, data, block)([x, y, z](const auto inner, const auto d, const auto b) {
	// 	// TODO: magical scatter `d -> b`
	// 	// mpi_scatter(d, b, inner, 0);

	// 	mpi_barrier(inner);

	// 	std::cerr << "Checking consistency of the data types" << '\n';

	// 	{
	// 		std::cerr << "Data:" << '\n';
	// 		const auto [d_lb, d_extent] = mpi_type_get_extent(d);
	// 		std::cerr << "  Extent: " << d_extent << '\n';
	// 		std::cerr << "  Lower bound: " << d_lb << '\n';

	// 		std::cerr << "Block:" << '\n';
	// 		const auto [b_lb, b_extent] = mpi_type_get_extent(b);
	// 		std::cerr << "  Extent: " << b_extent << '\n';
	// 		std::cerr << "  Lower bound: " << b_lb << '\n';

	// 		if (d_lb != b_lb * 8 || d_extent != b_extent * 8) {
	// 			std::cerr << "Error: " << d_lb << " != " << b_lb * 8 << " or " << d_extent << " != " << b_extent * 8
	// 					  << std::endl;

	// 			throw std::runtime_error("Data types are not consistent");
	// 		}
	// 	}

	// 	mpi_barrier(inner);

	// 	std::cerr << "Data types are consistent" << '\n';

	// 	// get the indices of the corresponding block
	// 	const auto [X, Y, Z] = noarr::get_indices<'X', 'Y', 'Z'>(inner);

	// 	// communicate along X
	// 	// MPICHK(MPI_Comm_split(MPI_COMM_WORLD, Y * z + Z, X, &x_comm));
	// 	auto x_comm = noarr::mpi_comm_split_along<'X', /*all_dims: */ 'X', 'Y', 'Z'>(inner);

	// 	// communicate along Y
	// 	// MPICHK(MPI_Comm_split(MPI_COMM_WORLD, Z * x + X, Y, &y_comm));
	// 	auto y_comm = noarr::mpi_comm_split_along<'Y', /*all_dims: */ 'X', 'Y', 'Z'>(inner);

	// 	// communicate along Z
	// 	// MPICHK(MPI_Comm_split(MPI_COMM_WORLD, X * y + Y, Z, &z_comm));
	// 	auto z_comm = noarr::mpi_comm_split_along<'Z', /*all_dims: */ 'X', 'Y', 'Z'>(inner);

	// 	// -> we wanna create a shortcut for `MPI_Comm_split(the original communicator, all other indices, the index
	// 	// we are communicating along, &the new communicator)`
	// 	// -> we wanna generalize the above to `split(the original communicator, all other indices, the indices we
	// 	// are communicating along, &the new communicator)`

	// 	// the following is just normal noarr code
	// 	if (X + Y + Z == 0) {
	// 		inner | [b](auto state) {
	// 			auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

	// 			b[state] = 1 + x + y + z;
	// 		};
	// 	}

	// 	// broadcast along the communicators
	// 	// MPICHK(MPI_Bcast(b.data(), 1, b.get_mpi_type(), 0, x_comm));
	// 	mpi_bcast(b, x_comm, 0);

	// 	static_assert(requires {
	// 		// broadcast globally in the traverser; just compile, never execute
	// 		{ mpi_bcast(b, inner, 0) } -> std::same_as<void>;
	// 	});

	// 	// -> we wanna generalize the above to `broadcast(b, the communicator we are broadcasting along)`

	// 	if (Y == 0 && Z == 0) {
	// 		inner | [b](auto state) {
	// 			auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

	// 			if ((std::size_t)b[state] != 1 + x + y + z) {
	// 				std::cerr << "Error: " << b[state] << " != " << 1 + x + y + z << std::endl;
	// 			}
	// 		};
	// 	} else {
	// 		inner | [b](auto state) {
	// 			if ((std::size_t)b[state] != 0) {
	// 				std::cerr << "Error: " << b[state] << " != 0" << std::endl;
	// 			}
	// 		};
	// 	}

	// 	// MPICHK(MPI_Bcast(b.data(), 1, b.get_mpi_type(), 0, y_comm));
	// 	mpi_bcast(b, y_comm, 0);

	// 	if (Z == 0) {
	// 		inner | [b](auto state) {
	// 			auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

	// 			if ((std::size_t)b[state] != 1 + x + y + z) {
	// 				std::cerr << "Error: " << b[state] << " != " << 1 + x + y + z << std::endl;
	// 			}
	// 		};
	// 	} else {
	// 		inner | [b](auto state) {
	// 			if ((std::size_t)b[state] != 0) {
	// 				std::cerr << "Error: " << b[state] << " != 0" << std::endl;
	// 			}
	// 		};
	// 	}

	// 	// MPICHK(MPI_Bcast(b.data(), 1, b.get_mpi_type(), 0, z_comm));
	// 	mpi_bcast(b, z_comm, 0);

	// 	inner | [b](auto state) {
	// 		auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

	// 		if ((std::size_t)b[state] != 1 + x + y + z) {
	// 			std::cerr << "Error: " << b[state] << " != " << 1 + x + y + z << std::endl;
	// 		}
	// 	};

	// 	// TODO: gather the results (`b -> d`)
	// 	mpi_scatter(d, b, inner, 0);
	// 	// mpi_gather(b, d, inner, 0);
	// });

	mpi_run(trav, data, block)([x, y, z](const auto inner, const auto d, const auto b) {
		const auto rank = mpi_get_comm_rank(inner);
		const auto comm_size = mpi_get_comm_size(inner);

		if (rank == 0) {
			// fill the data
			if (rank == 0) {
				std::size_t index = 0;
				(traverser(d) ^ set_length(inner.state())) | [&](auto state) {
					std::cerr << "Rank: " << rank << ", Value: " << (d[state] = index++) << '\n';
				};
			}
		}

		mpi_scatter(d, b, inner, 0);

		for (int i = 0; i < comm_size; ++i) {
			if (rank == i) {
				inner | [&](auto state) { std::cerr << "Value: " << b[state] << ", Rank: " << rank << '\n'; };
				std::cerr.flush();
			}
			mpi_barrier(inner);
		}

		if (rank == 0) {
			// reset the data
			if (rank == 0) {
				(traverser(d) ^ set_length(inner.state())) |
					[&](auto state) { std::cerr << "Rank: " << rank << ", Value: " << (d[state] = 0) << '\n'; };
			}
		}

		new_gather(b, d, inner, 0);

		if (rank == 0) {
			int index = 0;
			bool failed = false;

			(traverser(d) ^ set_length(inner.state())) | [&](auto state) {
				if (d[state] != index++) {
					failed = true;
				}
				std::cerr << "Rank: " << rank << ", Value: " << d[state] << '\n';
			};

			if (failed) {
				throw std::runtime_error("Data is not consistent");
			}
		}
	});

	mpi_barrier(mpi_session);

	std::cerr << "end" << '\n';
}
