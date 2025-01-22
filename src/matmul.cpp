#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>

#include "noarr/structures/interop/mpi_algorithms.hpp"
#include "noarr/structures/interop/mpi_traverser.hpp"
#include "noarr/structures/interop/mpi_utility.hpp"

#define MINI_DATASET
#define DATA_TYPE_IS_FLOAT

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
} tuning;

// initialization function
void init_array(auto inner, num_t &alpha, num_t &beta, auto C, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	traverser(C) ^ set_length(inner.state()) | [=](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		C[state] = (num_t)((i * j + 1) % (C | get_length<'i'>(inner.state()))) / (C | get_length<'i'>(inner.state()));
	};

	traverser(A) ^ set_length(inner.state()) | [=](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = (num_t)(i * (k + 1) % (A | get_length<'k'>(inner.state()))) / (A | get_length<'k'>(inner.state()));
	};

	traverser(B) ^ set_length(inner.state()) | [=](auto state) {
		auto [k, j] = get_indices<'k', 'j'>(state);
		B[state] = (num_t)(k * (j + 2) % (B | get_length<'j'>(inner.state()))) / (B | get_length<'j'>(inner.state()));
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(auto inner, num_t alpha, num_t beta, auto subC, auto subA, auto subB) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	inner | for_each<'i', 'j'>([=](auto state) {
		subC[state] *= beta;
	});

	inner | for_each<'i', 'j', 'k'>([=](auto state) {
		subC[state] += alpha * subA[state] * subB[state];
	});
}

} // namespace

// TODO: fails if NJ % 4 != 0 || NI % 2 != 0

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	const noarr::MPI_session mpi_session(argc, argv);
	const int rank = mpi_get_comm_rank(mpi_session);

	// problem size
	const std::size_t ni = NI;
	const std::size_t nj = NJ;
	const std::size_t nk = NK;

	const auto set_lengths = noarr::set_length<'i'>(ni) ^ noarr::set_length<'j'>(nj) ^ noarr::set_length<'k'>(nk);

	const auto C_structure = noarr::scalar<num_t>() ^ tuning.c_layout ^ set_lengths;
	const auto A_structure = noarr::scalar<num_t>() ^ tuning.a_layout ^ set_lengths;
	const auto B_structure = noarr::scalar<num_t>() ^ tuning.b_layout ^ set_lengths;

	const auto grid_i = noarr::into_blocks<'i', 'I'>();
	const auto grid_j = noarr::into_blocks<'j', 'J'>();

	const auto C_grid = C_structure ^ grid_i ^ grid_j;
	const auto A_grid = A_structure ^ grid_i;
	const auto B_grid = B_structure ^ grid_j;

	const auto C_data = (rank == 0) ? std::make_unique<char[]>(C_grid | noarr::get_size()) : std::unique_ptr<char[]>{};
	const auto A_data = (rank == 0) ? std::make_unique<char[]>(A_grid | noarr::get_size()) : std::unique_ptr<char[]>{};
	const auto B_data = (rank == 0) ? std::make_unique<char[]>(B_grid | noarr::get_size()) : std::unique_ptr<char[]>{};

	const auto trav = noarr::traverser(C_grid, A_grid, B_grid) ^ noarr::set_length<'I'>(2) ^ noarr::merge_blocks<'I', 'J', 'r'>();
	const auto mpi_trav = noarr::mpi_traverser<'r'>(trav, MPI_COMM_WORLD);

	const auto C = noarr::bag(C_grid, C_data.get());
	const auto A = noarr::bag(A_grid, A_data.get());
	const auto B = noarr::bag(B_grid, B_data.get());

	const auto subC_structure = noarr::scalar<num_t>() ^ tuning.c_layout ^ lengths_like<'i', 'j'>(mpi_trav.top_struct());
	const auto subA_structure = noarr::scalar<num_t>() ^ tuning.a_layout ^ lengths_like<'i', 'k'>(mpi_trav.top_struct());
	const auto subB_structure = noarr::scalar<num_t>() ^ tuning.b_layout ^ lengths_like<'j', 'k'>(mpi_trav.top_struct());

	const auto subC = noarr::bag(subC_structure);
	const auto subA = noarr::bag(subA_structure);
	const auto subB = noarr::bag(subB_structure);

	// initialize data
	mpi_run(mpi_trav, C, A, B, subC, subA, subB)([=](const auto inner, const auto C, const auto A, const auto B, const auto subC, const auto subA, const auto subB) {
		num_t alpha{};
		num_t beta{};

		if (rank == 0) {
			init_array(inner, alpha, beta, bag(C_structure, C.data()), bag(A_structure, A.data()), bag(B_structure, B.data()));
		}

		mpi_bcast(alpha, inner, 0);
		mpi_bcast(beta, inner, 0);


		auto start = std::chrono::high_resolution_clock::now();
		mpi_scatter(C, subC, inner, 0);
		mpi_scatter(A, subA, inner, 0);
		mpi_scatter(B, subB, inner, 0);

		// run kernel
		kernel_gemm(inner, alpha, beta, subC, subA, subB);

		mpi_gather(subC, C, inner, 0);

		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration<long double>(end - start);

		// print results
		if (rank == 0) {
			if (argc > 0 && argv[0] != ""s) {
				std::cout << std::fixed << std::setprecision(2);
				noarr::serialize_data(std::cout, C.get_ref() ^ set_length(inner.state()) ^ noarr::hoist<'I', 'i'>());
			}
			std::cerr << std::fixed << std::setprecision(6);
			std::cerr << duration.count() << std::endl;
		}

	});
}
