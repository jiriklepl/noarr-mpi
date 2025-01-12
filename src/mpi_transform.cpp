#include <cassert>
#include <expected>
#include <iostream>
#include <type_traits>

#include <mpi.h>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures/extra/struct_concepts.hpp>
#include <noarr/traversers.hpp>

#include "noarr/structures/base/signature.hpp"
#include "noarr/structures/base/utility.hpp"
#include "noarr/structures/extra/tokenizer.hpp"

#include "noarr/structures/interop/mpi_algorithms.hpp"
#include "noarr/structures/interop/mpi_bag.hpp"
#include "noarr/structures/interop/mpi_structs.hpp"
#include "noarr/structures/interop/mpi_traverser.hpp"
#include "noarr/structures/interop/mpi_utility.hpp"

// TODO: review the types and implement the missing ones

// ---------------------------------------------------------------------------

using num_t = int;

namespace noarr {

// for each of the functions, we want `fun(mpi_traverser, ...)`, `mpi_traverser.fun(...)` and `mpi_traverser | fun(...)`
// variants

// TODO: mpi_traverser_t <- this is a primitive that holds a traverser and a communicator

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

using TODO_TYPE = int;

// TODO: MPI_Comm custom wrapper with a destructor that calls MPI_Comm_free

// TODO: doesn't work
// inline void mpi_gather(auto structure, ToMPIComm auto has_comm, TODO_TYPE rank) {
// 	MPICHK(MPI_Gather(structure.data(), 1, structure.get_mpi_type(), structure.data(), 1, structure.get_mpi_type(),
// 	                  rank, convert_to_MPI_Comm(has_comm)));
// }

template<class Sig>
struct in_signature {
	using signature = Sig;
	using value_type = bool;

	template<IsDim auto Dim>
	static constexpr bool value = Sig::template any_accept<Dim>;
};

template<IsStruct Struct>
struct index_space_size {
public:
	template<class DimTree>
	static constexpr std::size_t get(Struct structure) noexcept {
		return get_impl(DimTree{}, structure);
	}

	static constexpr std::size_t get(Struct structure) noexcept {
		return get_impl(sig_dim_tree<typename Struct::signature>{}, structure);
	}

private:
	template<auto Dim, class... Branches>
	requires (sizeof...(Branches) != 1)
	static constexpr std::size_t get_impl(dim_tree<Dim, Branches...> /*dt*/, Struct structure) noexcept {
		return (... + get_impl(Branches{}, structure));
	}

	template<auto Dim, class Branch>
	static constexpr std::size_t get_impl(dim_tree<Dim, Branch> /*dt*/, Struct structure) noexcept {
		return (structure | get_length<Dim>()) * get_impl(Branch{}, structure);
	}

	static constexpr std::size_t get_impl(dim_sequence<> /*dt*/, Struct /*structure*/) noexcept { return 1; }
};

// TODO: rename this
template<IsDim auto Dim, class Branch, class... Branches, IsState State = state<>>
constexpr auto fix_first(dim_tree<Dim, Branch, Branches...>, State state = empty_state) noexcept {
	return fix_first(Branch{}, state & idx<Dim>(lit<0>));
}

template<auto... Dims, IsState State = state<>> requires IsDimPack<decltype(Dims)...>
constexpr auto fix_first(dim_sequence<Dims...>, State state = empty_state) noexcept {
	constexpr auto zero = [](auto /*Dim*/) { return lit<0>; };
	return state & idx<Dims...>(zero(Dims)...);
}

inline void mpi_scatter(auto from, auto to, IsMpiTraverser auto traverser, TODO_TYPE root) {
	// TODO: make this traverser-aware, which simplifies the code like... a lot
	const auto from_struct = convert_to_struct(from) ^ noarr::set_length(traverser.state());
	const auto to_struct = convert_to_struct(to) ^ noarr::set_length(traverser.state());
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
	using from_dim_filtered = dim_tree_filter<from_dim_tree, in_signature<typename decltype(from_struct)::signature>>;

	// contains the dimensions that are present in the "from" structure, but not in the "to" structure
	using from_dim_removed =
		dim_tree_filter<from_dim_tree, dim_pred_not<in_signature<typename decltype(to_struct)::signature>>>;

	const std::size_t difference_size =
		index_space_size<decltype(from_struct)>::template get<from_dim_removed>(from_struct);
	std::vector<int> displacements(difference_size);

	const int offset = from_struct ^ traverser.get_order() | noarr::offset(fix_first(to_dim_filtered{}));
	const int rank = mpi_get_rank(comm);

	const auto from_substructure = from_struct ^ fix(fix_first(from_dim_removed{}));

	mpi_barrier(comm);

	if (rank == root) {
		const auto mpi_rep = mpi_transform_builder{}.process(from_substructure);

		const auto [lb, extent] = mpi_type_get_extent(mpi_rep);

		std::cout << "Size: " << (from_substructure | noarr::get_size()) << '\n';
		std::cout << "Extent: " << extent << '\n';
		std::cout << "Lower bound: " << lb << '\n';
	}


	mpi_barrier(comm);

	MPICHK(MPI_Allgather(&offset, 1, MPI_INT, displacements.data(), 1, MPI_INT, comm));

	// TODO: remove this
	if (rank == root) {
		int r = 0;
		for (const auto displacement : displacements) {
			std::cout << "Rank: " << r << ", Displacement: " << displacement << '\n';
			++r;
		}
	}

	mpi_barrier(comm);

	std::cout << "Rank: " << rank << ", Real displacement: " << offset << '\n';

	return; // TODO: remove this

	const auto from_traverser = noarr::traverser(from_struct);
	// the displacements are determined by the `from` structure and `from_dim_removed`
	for_each_impl(
		from_traverser, from_dim_removed{},
		[](auto inner) {
			// test
			std::cout << "X: " << get_index<'X'>(inner) << ", Y: " << get_index<'Y'>(inner)
					  << ", Z: " << get_index<'Z'>(inner) << '\n';
			// TODO:
		    // displacements[TODO] = TODO;
		},
		convert_to_state(from_traverser));
};

template<IsDim auto... Dims>
struct remove_indices {
	template<class Tag>
	static constexpr bool value = !IsIndexIn<Tag> || !(... || (Tag::dims::template contains<Dims>));
};

} // namespace noarr

auto main(int argc, char **argv) -> int try {
	const noarr::MPI_session mpi_session(argc, argv);

	using namespace noarr;
	// to be shadowed by a local definition of order

	volatile std::size_t x = 300;
	volatile std::size_t y = 321;
	volatile std::size_t z = 801;

	std::cerr << "x: " << x << ", y: " << y << ", z: " << z << '\n';
	std::cerr << "Size: " << x * y * z << '\n';

	const int rank = mpi_get_rank(mpi_session);

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
	std::cerr << block.structure().size(empty_state) << '\n';

	auto data = noarr::bag(data_structure, data_blob.get());

	// MPI_Aint lb = 0;
	// MPI_Aint extent = 0;

	// MPICHK(MPI_Type_get_extent((MPI_Datatype)mpi_rep, &lb, &extent));
	// std::cerr << "Extent: " << extent << '\n';
	// std::cerr << "Lower bound: " << lb << '\n';

	// TODO `block -> b` is not the most elegant solution
	mpi_run(trav, data, block)([x, y, z](const auto inner, const auto d, const auto b) {
		// TODO: magical scatter `d -> b`
		// mpi_scatter(d, b, inner, 0);

		mpi_barrier(inner);

		std::cerr << "Checking consistency of the data types" << '\n';

		{
			const auto [d_lb, d_extent] = mpi_type_get_extent(d);
			std::cerr << "Extent: " << d_extent << '\n';
			std::cerr << "Lower bound: " << d_lb << '\n';

			const auto [b_lb, b_extent] = mpi_type_get_extent(b);
			std::cerr << "Extent: " << b_extent << '\n';
			std::cerr << "Lower bound: " << b_lb << '\n';

			if (d_lb != b_lb * 8 || d_extent != b_extent * 8) {
				std::cerr << "Error: " << d_lb << " != " << b_lb * 8 << " or " << d_extent << " != " << b_extent * 8
						  << std::endl;
			}
		}

		mpi_barrier(inner);

		std::cerr << "Data types are consistent" << '\n';

		// get the indices of the corresponding block
		const auto [X, Y, Z] = noarr::get_indices<'X', 'Y', 'Z'>(inner);

		// communicate along X
		// MPICHK(MPI_Comm_split(MPI_COMM_WORLD, Y * z + Z, X, &x_comm));
		auto x_comm = noarr::mpi_comm_split_along<'X', /*all_dims: */ 'X', 'Y', 'Z'>(inner);

		// communicate along Y
		// MPICHK(MPI_Comm_split(MPI_COMM_WORLD, Z * x + X, Y, &y_comm));
		auto y_comm = noarr::mpi_comm_split_along<'Y', /*all_dims: */ 'X', 'Y', 'Z'>(inner);

		// communicate along Z
		// MPICHK(MPI_Comm_split(MPI_COMM_WORLD, X * y + Y, Z, &z_comm));
		auto z_comm = noarr::mpi_comm_split_along<'Z', /*all_dims: */ 'X', 'Y', 'Z'>(inner);

		// -> we wanna create a shortcut for `MPI_Comm_split(the original communicator, all other indices, the index
		// we are communicating along, &the new communicator)`
		// -> we wanna generalize the above to `split(the original communicator, all other indices, the indices we
		// are communicating along, &the new communicator)`

		// the following is just normal noarr code
		if (X + Y + Z == 0) {
			inner | [b](auto state) {
				auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

				b[state] = 1 + x + y + z;
			};
		}

		// broadcast along the communicators
		// MPICHK(MPI_Bcast(b.data(), 1, b.get_mpi_type(), 0, x_comm));
		mpi_bcast(b, x_comm, 0);

		static_assert(requires {
			// broadcast globally in the traverser; just compile, never execute
			{ mpi_bcast(b, inner, 0) } -> std::same_as<void>;
		});

		// -> we wanna generalize the above to `broadcast(b, the communicator we are broadcasting along)`

		if (Y == 0 && Z == 0) {
			inner | [b](auto state) {
				auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

				if ((std::size_t)b[state] != 1 + x + y + z) {
					std::cerr << "Error: " << b[state] << " != " << 1 + x + y + z << std::endl;
				}
			};
		} else {
			inner | [b](auto state) {
				if ((std::size_t)b[state] != 0) {
					std::cerr << "Error: " << b[state] << " != 0" << std::endl;
				}
			};
		}

		// MPICHK(MPI_Bcast(b.data(), 1, b.get_mpi_type(), 0, y_comm));
		mpi_bcast(b, y_comm, 0);

		if (Z == 0) {
			inner | [b](auto state) {
				auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

				if ((std::size_t)b[state] != 1 + x + y + z) {
					std::cerr << "Error: " << b[state] << " != " << 1 + x + y + z << std::endl;
				}
			};
		} else {
			inner | [b](auto state) {
				if ((std::size_t)b[state] != 0) {
					std::cerr << "Error: " << b[state] << " != 0" << std::endl;
				}
			};
		}

		// MPICHK(MPI_Bcast(b.data(), 1, b.get_mpi_type(), 0, z_comm));
		mpi_bcast(b, z_comm, 0);

		inner | [b](auto state) {
			auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

			if ((std::size_t)b[state] != 1 + x + y + z) {
				std::cerr << "Error: " << b[state] << " != " << 1 + x + y + z << std::endl;
			}
		};

		// TODO: gather the results (`b -> d`)
		mpi_scatter(d, b, inner, 0);
		// mpi_gather(b, d, inner, 0);
	});

	mpi_barrier(mpi_session);

	std::cerr << "end" << '\n';
} catch (const std::exception &e) {

	std::cerr << "Exception: " << e.what() << '\n';
	return 1;
} catch (...) {
	std::cerr << "Unknown exception" << '\n';
	return 1;
}
