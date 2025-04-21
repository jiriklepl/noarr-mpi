// This test checks whether the scatter function is sufficiently type-safe.

#include <noarr/mpi.hpp>

using namespace noarr;

constexpr int root = 0;

int main() {
	MPI_session mpi_session;

	const auto root_structure = scalar<int>() ^ vectors<'x', 'y', 'z'>(2, 2, 2);
	const auto grid = into_blocks<'x', 'X'>() ^ into_blocks<'y', 'Y'>() ^ into_blocks<'z', 'Z'>();
	const auto distr_strategy =
		set_length<'X', 'Y'>(2, 2) ^ merge_blocks<'X', 'Y', '_'>() ^ merge_blocks<'_', 'Z', 'r'>();

	const auto trav = traverser(root_structure ^ grid);
	const auto mpi_trav = mpi_traverser<'r'>(trav ^ distr_strategy, mpi_session);

	auto root_bag = bag(root_structure ^ grid, nullptr);
	auto tile_bag = bag(scalar<int>() ^ vectors<'x', 'y', 'z', 'w'>(2, 3, 4, 5));

	// The following function call is incorrect (it has an extra 'w' dimension), fails at compile time
	mpi_scatter(root_bag, tile_bag, mpi_trav, root);
}
