#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/introspection.hpp>
#include <noarr/traversers.hpp>

#include "noarr/structures/interop/mpi_algorithms.hpp"
#include "noarr/structures/interop/mpi_traverser.hpp"
#include "noarr/structures/interop/mpi_utility.hpp"

#include "defines.hpp"
#include "gemm.hpp"

using num_t = DATA_TYPE;

namespace {

constexpr auto i_vec = noarr::vector<'i'>();
constexpr auto j_vec = noarr::vector<'j'>();
constexpr auto k_vec = noarr::vector<'k'>();

const struct tuning {
	DEFINE_PROTO_STRUCT(c_layout, j_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(a_layout, k_vec ^ i_vec);
	DEFINE_PROTO_STRUCT(b_layout, j_vec ^ k_vec);

#ifdef C_TILE_J_MAJOR
	DEFINE_PROTO_STRUCT(c_tile_layout, i_vec ^ j_vec);
#else
	DEFINE_PROTO_STRUCT(c_tile_layout, j_vec ^ i_vec);
#endif

#ifdef A_TILE_K_MAJOR
	DEFINE_PROTO_STRUCT(a_tile_layout, i_vec ^ k_vec);
#else
	DEFINE_PROTO_STRUCT(a_tile_layout, k_vec ^ i_vec);
#endif

#ifdef B_TILE_J_MAJOR
	DEFINE_PROTO_STRUCT(b_tile_layout, k_vec ^ j_vec);
#else
	DEFINE_PROTO_STRUCT(b_tile_layout, j_vec ^ k_vec);
#endif
} tuning;

// initialization function
void init_array(auto inner, num_t &alpha, const auto &C, num_t &beta, const auto &A, const auto &B) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	(traverser(C) ^ set_length(inner)) | [&](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		C[state] = (num_t)((i * j + 1) % (C | get_length<'i'>(inner.state()))) / (C | get_length<'i'>(inner.state()));
	};

	(traverser(A) ^ set_length(inner)) | [&](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = (num_t)(i * (k + 1) % (A | get_length<'k'>(inner.state()))) / (A | get_length<'k'>(inner.state()));
	};

	(traverser(B) ^ set_length(inner)) | [&](auto state) {
		auto [k, j] = get_indices<'k', 'j'>(state);
		B[state] = (num_t)(k * (j + 2) % (B | get_length<'j'>(inner.state()))) / (B | get_length<'j'>(inner.state()));
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(auto trav, num_t alpha, auto C, num_t beta, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	trav | for_dims<'i'>([=](auto inner) {
		inner | for_each<'j'>([=](auto state) {
			C[state] *= beta;
		});

		inner | [=](auto state) {
			C[state] += alpha * A[state] * B[state];
		};
	});
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;
	namespace chrono = std::chrono;

	const noarr::MPI_session mpi_session(argc, argv);
	const int rank = mpi_get_comm_rank(mpi_session);
	constexpr int root = 0;

	const auto set_lengths = noarr::set_length<'i'>(NI) ^ noarr::set_length<'j'>(NJ) ^ noarr::set_length<'k'>(NK);

	const auto scalar = noarr::scalar<num_t>();

	const auto C_structure = scalar ^ tuning.c_layout ^ set_lengths;
	const auto A_structure = scalar ^ tuning.a_layout ^ set_lengths;
	const auto B_structure = scalar ^ tuning.b_layout ^ set_lengths;

	const auto grid_i = noarr::into_blocks<'i', 'I'>();
	const auto grid_j = noarr::into_blocks<'j', 'J'>();
	const auto grid = grid_i ^ grid_j;

	const auto C_data =
		(rank == root) ? std::make_unique<char[]>(C_structure | noarr::get_size()) : std::unique_ptr<char[]>{};
	const auto A_data =
		(rank == root) ? std::make_unique<char[]>(A_structure | noarr::get_size()) : std::unique_ptr<char[]>{};
	const auto B_data =
		(rank == root) ? std::make_unique<char[]>(B_structure | noarr::get_size()) : std::unique_ptr<char[]>{};

	const auto C = noarr::bag(C_structure ^ grid, C_data.get());
	const auto A = noarr::bag(A_structure ^ grid, A_data.get());
	const auto B = noarr::bag(B_structure ^ grid, B_data.get());

	const auto trav = noarr::traverser(C, A, B) ^ noarr::set_length<'I'>(2) ^ noarr::merge_blocks<'I', 'J', 'r'>();
	const auto mpi_trav = noarr::mpi_traverser<'r'>(trav, MPI_COMM_WORLD);

	const auto tileC = noarr::bag(scalar ^ tuning.c_tile_layout ^ lengths_like<'j', 'i'>(mpi_trav));
	const auto tileA = noarr::bag(scalar ^ tuning.a_tile_layout ^ lengths_like<'k', 'i'>(mpi_trav));
	const auto tileB = noarr::bag(scalar ^ tuning.b_tile_layout ^ lengths_like<'j', 'k'>(mpi_trav));

	num_t alpha{};
	num_t beta{};

	// initialize data
	if (rank == root) {
		init_array(mpi_trav, alpha, bag(C_structure, C.data()), beta, bag(A_structure, A.data()),
		           bag(B_structure, B.data()));
	}

	mpi_bcast(alpha, mpi_trav, 0);
	mpi_bcast(beta, mpi_trav, 0);

	const auto start = chrono::high_resolution_clock::now();
	mpi_scatter(C, tileC, mpi_trav, 0);
	mpi_scatter(A, tileA, mpi_trav, 0);
	mpi_scatter(B, tileB, mpi_trav, 0);

	// run kernel
	kernel_gemm(mpi_trav, alpha, tileC.get_ref(), beta, tileA.get_ref(), tileB.get_ref());

	mpi_gather(tileC, C, mpi_trav, 0);

	const auto end = chrono::high_resolution_clock::now();

	const auto duration = chrono::duration<double>(end - start);

	// print results
	if (rank == root) {
		std::cerr << std::fixed << std::setprecision(6);
		std::cerr << duration.count() << std::endl;
		if (argc > 0 && argv[0] != ""s) {
			std::cout << std::fixed << std::setprecision(2);
			noarr::serialize_data(std::cout, C.get_ref() ^ set_length(mpi_trav) ^ noarr::hoist<'I', 'i', 'J', 'j'>());
		}
	}

	mpi_barrier(mpi_trav);
}
