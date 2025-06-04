// This test checks whether the send/receive functions automatically permutate the data structure

#include "noarr/mpi/utility.hpp"
#include <cstdlib>

#include <noarr/traversers.hpp>

#include <noarr/mpi.hpp>

using namespace noarr;
namespace mpi = noarr::mpi;

constexpr int root = 0;

int main() {
	mpi::MPI_session mpi_session;

	const auto from_structure = scalar<int>() ^ vectors<'x', 'y', 'z'>(2, 2, 2);
	const auto to_structure = scalar<int>() ^ vectors<'z', 'y','x'>(2, 2, 2);
	const auto distr_strategy = bcast<'r'>();

	const auto trav = traverser(from_structure);
	const auto mpi_trav = mpi::mpi_traverser<'r'>(trav ^ distr_strategy, mpi_session);

	int return_code = EXIT_SUCCESS;

	if (mpi::mpi_get_comm_rank(mpi_trav) == root) {
		const auto from_bag = bag(from_structure);
		const auto to_bag = bag(to_structure);

		int i = 0;

		traverser(from_bag) | for_dims<'x', 'y', 'z'>([&](auto state) {
			from_bag[state] = i++;
		});

		{
			auto send_request = mpi::isend(from_bag, mpi_trav, root);
			auto recv_request = mpi::irecv(to_bag, mpi_trav, root);

			recv_request.wait();
			send_request.wait();
		}

		i = 0;

		traverser(to_bag) | for_dims<'x', 'y', 'z'>([&](auto state) {
			if (to_bag[state] != i++) {
				std::cerr << "Error: expected " << i - 1 << ", got " << to_bag[state] << '\n';
				return_code = EXIT_FAILURE;
			}
		});
	}

	mpi::barrier(mpi_trav);

	mpi::broadcast(return_code, mpi_trav, root);

	return return_code;
}
