#include <cassert>
#include <expected>
#include <iostream>

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

	const auto rank = mpi_get_comm_rank(mpi_session);
	const auto comm_size = mpi_get_comm_size(mpi_session);

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

	if (rank == 0) {
		// fill the data
		if (rank == 0) {
			std::size_t index = 0;
			(traverser(data) ^ set_length(trav.state())) |
				[&](auto state) { std::cerr << "Rank: " << rank << ", Value: " << (data[state] = index++) << '\n'; };
		}
	}

	mpi_scatter(data, block, trav, 0);

	for (int i = 0; i < comm_size; ++i) {
		if (rank == i) {
			trav | [&](auto state) { std::cerr << "Value: " << block[state] << ", Rank: " << rank << '\n'; };
			std::cerr.flush();
		}
		mpi_barrier(trav);
	}

	if (rank == 0) {
		// reset the data
		if (rank == 0) {
			(traverser(data) ^ set_length(trav.state())) |
				[&](auto state) { std::cerr << "Rank: " << rank << ", Value: " << (data[state] = 0) << '\n'; };
		}
	}

	mpi_gather(block, data, trav, 0);

	if (rank == 0) {
		int index = 0;
		bool failed = false;

		(traverser(data) ^ set_length(trav.state())) | [&](auto state) {
			if (data[state] != index++) {
				failed = true;
			}
			std::cerr << "Rank: " << rank << ", Value: " << data[state] << '\n';
		};

		if (failed) {
			throw std::runtime_error("Data is not consistent");
		}
	}

	mpi_barrier(mpi_session);

	std::cerr << "end" << '\n';

	if (rank == 1) {
		block[idx<'x', 'y', 'z'>(0, 0, 1)] = 42;
	}

	// mpi_bcast(block.get_ref(), trav, 1);
	mpi_bcast(block.get_ref() ^ fix<'x', 'y', 'z'>(0, 0, 1), trav, 1);

	if (rank == 0) {
		std::cerr << "Value: " << block[idx<'x', 'y', 'z'>(0, 0, 1)] << '\n';
	}

	mpi_barrier(mpi_session);
}
