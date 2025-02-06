#include <cassert>
#include <expected>
#include <iostream>

#include <mpi.h>

#include <noarr/introspection.hpp>
#include <noarr/traversers.hpp>

#include "noarr/structures/interop/mpi_algorithms.hpp"
#include "noarr/structures/interop/mpi_traverser.hpp"
#include "noarr/structures/interop/mpi_utility.hpp"

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
