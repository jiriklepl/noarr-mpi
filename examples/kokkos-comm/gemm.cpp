#include <cstddef>
#include <cstdint>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <KokkosComm/KokkosComm.hpp>

#include "defines.hpp"
#include "gemm.hpp"

using num_t = DATA_TYPE;

namespace {

template<typename T, typename Layout = Kokkos::LayoutRight>
struct matrix_factory {
	auto operator()(T *data, std::size_t rows, std::size_t cols) const {
		return Kokkos::View<T **, Layout>(data, rows, cols);
	}
};

const struct tuning {
DEFINE_PROTO_STRUCT(c_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});
DEFINE_PROTO_STRUCT(a_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});
DEFINE_PROTO_STRUCT(b_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});

#ifdef C_TILE_J_MAJOR
	DEFINE_PROTO_STRUCT(c_tile_layout, matrix_factory<num_t, Kokkos::LayoutLeft>{});
#else
	DEFINE_PROTO_STRUCT(c_tile_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});
#endif

#ifdef A_TILE_K_MAJOR
	DEFINE_PROTO_STRUCT(a_tile_layout, matrix_factory<num_t, Kokkos::LayoutLeft>{});
#else
	DEFINE_PROTO_STRUCT(a_tile_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});
#endif

#ifdef B_TILE_J_MAJOR
	DEFINE_PROTO_STRUCT(b_tile_layout, matrix_factory<num_t, Kokkos::LayoutLeft>{});
#else
	DEFINE_PROTO_STRUCT(b_tile_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});
#endif
} tuning;

// initialization function
void init_array(num_t &alpha, auto C, num_t &beta, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: j x k

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	for (std::size_t i = 0; i < NI; ++i) {
		for (std::size_t j = 0; j < NJ; ++j) {
			C(i, j) = (num_t)((i * j + 1) % NI) / NI;
		}
	}

	for (std::size_t i = 0; i < NI; ++i) {
		for (std::size_t k = 0; k < NK; ++k) {
			A(i, k) = (num_t)(i * (k + 1) % NK) / NK;
		}
	}

	for (std::size_t j = 0; j < NJ; ++j) {
		for (std::size_t k = 0; k < NK; ++k) {
			B(k, j) = (num_t)(k * (j + 2) % NJ) / NJ;
		}
	}
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(num_t alpha, auto C, num_t beta, auto A, auto B,
				 std::size_t SI, std::size_t SJ, std::size_t SK) {
	// C: i x j
	// A: i x k
	// B: j x k

	for (std::size_t i = 0; i < SI; ++i) {
		for (std::size_t j = 0; j < SJ; ++j) {
			C(i, j) *= beta;
		}

		for (std::size_t j = 0; j < SJ; ++j) {
			for (std::size_t k = 0; k < SK; ++k) {
				C(i, j) += alpha * A(i, k) * B(k, j);
			}
		}
	}
}

} // namespace

void run(int argc, char *argv[]) {
	using namespace std::string_literals;
	namespace chrono = std::chrono;

	KokkosComm::Handle<> handle;

	const int rank = handle.rank();
	const int size = handle.size();
	constexpr int root = 0;

	const auto C_data =
		(rank == root) ? std::make_unique<num_t[]>(NI * NJ) : nullptr;
	const auto A_data =
		(rank == root) ? std::make_unique<num_t[]>(NI * NK) : nullptr;
	const auto B_data =
		(rank == root) ? std::make_unique<num_t[]>(NK * NJ) : nullptr;

	const auto C = tuning.c_layout(C_data.get(), NI, NJ);
	const auto A = tuning.a_layout(A_data.get(), NI, NK);
	const auto B = tuning.b_layout(B_data.get(), NK, NJ);

	const int i_tiles = (argc > 1) ? std::atoi(argv[1]) : 1;
	const int j_tiles = size / i_tiles;

	const std::size_t SI = NI / i_tiles;
	const std::size_t SJ = NJ / j_tiles;

	const auto tileC_data = std::make_unique<num_t[]>(SI * SJ);
	const auto tileA_data = std::make_unique<num_t[]>(SI * NK);
	const auto tileB_data = std::make_unique<num_t[]>(SJ * NK);

	const auto tileC = tuning.c_tile_layout(tileC_data.get(), SI, SJ);
	const auto tileA = tuning.a_tile_layout(tileA_data.get(), SI, NK);
	const auto tileB = tuning.b_tile_layout(tileB_data.get(), NK, SJ);

	num_t alpha{};
	num_t beta{};

	if (rank == root) {
		init_array(alpha, C, beta, A, B);
	}

	MPI_Bcast(&alpha, 1, KokkosComm::Impl::mpi_type_v<num_t>, root, handle.mpi_comm());
	MPI_Bcast(&beta, 1, KokkosComm::Impl::mpi_type_v<num_t>, root, handle.mpi_comm());

	const auto start = chrono::high_resolution_clock::now();

	std::vector<KokkosComm::Req<>> reqs;

	if (rank == root) {
		for (int r = 0; r < size; ++r) {
			int i = r / j_tiles;
			int j = r % j_tiles;

			const auto c_subview = Kokkos::subview(C, std::make_pair<int, int>(i * SI, (i + 1) * SI), std::make_pair<int, int>(j * SJ, (j + 1) * SJ));
			const auto a_subview = Kokkos::subview(A, std::make_pair<int, int>(i * SI, (i + 1) * SI), Kokkos::ALL);
			const auto b_subview = Kokkos::subview(B, Kokkos::ALL, std::make_pair<int, int>(j * SJ, (j + 1) * SJ));

			reqs.push_back(KokkosComm::send(handle, c_subview, r));
			reqs.push_back(KokkosComm::send(handle, a_subview, r));
			reqs.push_back(KokkosComm::send(handle, b_subview, r));
		}
	}

	reqs.push_back(KokkosComm::recv(handle, tileC, root));
	reqs.push_back(KokkosComm::recv(handle, tileA, root));
	reqs.push_back(KokkosComm::recv(handle, tileB, root));

	KokkosComm::wait_all(reqs);
	reqs.clear();

	kernel_gemm(alpha, tileC, beta, tileA, tileB, SI, SJ, NK);

	reqs.push_back(KokkosComm::send(handle, tileC, root));


	if (rank == root) {
		for (int r = 0; r < size; ++r) {
			int i = r / j_tiles;
			int j = r % j_tiles;

			const auto c_subview = Kokkos::subview(C, std::make_pair<int, int>(i * SI, (i + 1) * SI), std::make_pair<int, int>(j * SJ, (j + 1) * SJ));

			reqs.push_back(KokkosComm::recv(handle, c_subview, r));
		}
	}

	KokkosComm::wait_all(reqs);
	reqs.clear();

	const auto end = chrono::high_resolution_clock::now();

	const auto duration = chrono::duration<double>(end - start);

	// print results
	if (rank == root) {
		if (argc > 0 && argv[0] != ""s) {
			std::cout << std::fixed << std::setprecision(2);
			for (auto i = 0; i < NI; ++i) {
				for (auto j = 0; j < NJ; ++j) {
					std::cout << C(i, j) << std::endl;
				}
			}
		}

		std::cout << std::fixed << std::setprecision(6);
		std::cout << duration.count() << std::endl;
	}

	KokkosComm::barrier((KokkosComm::Handle<>)handle);
}

int main(int argc, char *argv[]) {
	int provided = 0;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE) {
		std::cerr << "MPI does not support MPI_THREAD_MULTIPLE" << std::endl;
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	try {
		Kokkos::initialize(argc, argv);
		run(argc, argv);
	} catch (const std::exception &e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	} catch (...) {
		std::cerr << "Unknown exception" << std::endl;
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	MPI_Finalize();
}
