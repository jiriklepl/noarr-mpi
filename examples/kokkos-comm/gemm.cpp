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
#include <KokkosComm/collective.hpp>
#include <KokkosComm/fwd.hpp>
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
// 	DEFINE_PROTO_STRUCT(c_layout, matrix_factory<num_t, RowMajor>{});
// 	DEFINE_PROTO_STRUCT(a_layout, matrix_factory<num_t, RowMajor>{});
// 	DEFINE_PROTO_STRUCT(b_layout, matrix_factory<num_t, RowMajor>{});
DEFINE_PROTO_STRUCT(c_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});
DEFINE_PROTO_STRUCT(a_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});
DEFINE_PROTO_STRUCT(b_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});

// #ifdef C_TILE_J_MAJOR
// 	DEFINE_PROTO_STRUCT(c_tile_layout, matrix_factory<num_t, ColMajor>{});
// #else
// 	DEFINE_PROTO_STRUCT(c_tile_layout, matrix_factory<num_t, RowMajor>{});
// #endif
#ifdef C_TILE_J_MAJOR
	DEFINE_PROTO_STRUCT(c_tile_layout, matrix_factory<num_t, Kokkos::LayoutLeft>{});
#else
	DEFINE_PROTO_STRUCT(c_tile_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});
#endif

// #ifdef A_TILE_K_MAJOR
// 	DEFINE_PROTO_STRUCT(a_tile_layout, matrix_factory<num_t, ColMajor>{});
// #else
// 	DEFINE_PROTO_STRUCT(a_tile_layout, matrix_factory<num_t, RowMajor>{});
// #endif
#ifdef A_TILE_K_MAJOR
	DEFINE_PROTO_STRUCT(a_tile_layout, matrix_factory<num_t, Kokkos::LayoutLeft>{});
#else
	DEFINE_PROTO_STRUCT(a_tile_layout, matrix_factory<num_t, Kokkos::LayoutRight>{});
#endif

// #ifdef B_TILE_J_MAJOR
// 	DEFINE_PROTO_STRUCT(b_tile_layout, matrix_factory<num_t, ColMajor>{});
// #else
// 	DEFINE_PROTO_STRUCT(b_tile_layout, matrix_factory<num_t, RowMajor>{});
// #endif
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
			B(j, k) = (num_t)(k * (j + 2) % NJ) / NJ;
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
				C(i, j) += alpha * A(i, k) * B(j, k);
			}
		}
	}
}

} // namespace

void run(int argc, char *argv[]) {
	using namespace std::string_literals;
	namespace chrono = std::chrono;

	// const mpl::communicator &comm_world{mpl::environment::comm_world()};

	// // const noarr::MPI_session mpi_session(argc, argv);
	// const int comm_rank = comm_world.rank();
	// constexpr int root = 0;

	KokkosComm::Handle<> handle;

	const int comm_rank = handle.rank();
	constexpr int root = 0;

	// const int comm_size = comm_world.size();

	const int comm_size = handle.size();

	const auto C_data =
		(comm_rank == root) ? std::make_unique<num_t[]>(NI * NJ) : nullptr;
	const auto A_data =
		(comm_rank == root) ? std::make_unique<num_t[]>(NI * NK) : nullptr;
	const auto B_data =
		(comm_rank == root) ? std::make_unique<num_t[]>(NK * NJ) : nullptr;

	const auto C = tuning.c_layout(C_data.get(), NI, NJ);
	const auto A = tuning.a_layout(A_data.get(), NI, NK);
	const auto B = tuning.b_layout(B_data.get(), NJ, NK);

	const std::size_t SI = NI / 2;
	const std::size_t SJ = NJ / (comm_size / 2);

	const auto tileC_data = std::make_unique<num_t[]>(SI * SJ);
	const auto tileA_data = std::make_unique<num_t[]>(SI * NK);
	const auto tileB_data = std::make_unique<num_t[]>(SJ * NK);

	const auto tileC = tuning.c_tile_layout(tileC_data.get(), SI, SJ);
	const auto tileA = tuning.a_tile_layout(tileA_data.get(), SI, NK);
	const auto tileB = tuning.b_tile_layout(tileB_data.get(), SJ, NK);

	num_t alpha{};
	num_t beta{};

	if (comm_rank == root) {
		init_array(alpha, C, beta, A, B);
	}

	// comm_world.bcast(root, alpha);
	// comm_world.bcast(root, beta);

	MPI_Bcast(&alpha, 1, MPI_FLOAT, root, handle.mpi_comm());
	MPI_Bcast(&beta, 1, MPI_FLOAT, root, handle.mpi_comm());

	// mpl::layouts<num_t> c_layouts;
	// mpl::layouts<num_t> a_layouts;
	// mpl::layouts<num_t> b_layouts;

	std::vector<int> c_displacements(comm_size);
	std::vector<int> a_displacements(comm_size);
	std::vector<int> b_displacements(comm_size);
	const std::vector<int> send_counts(comm_size, 1);

	const auto c_tile_type = KokkosComm::Impl::view_mpi_type(tileC);
	const auto a_tile_type = KokkosComm::Impl::view_mpi_type(tileA);
	const auto b_tile_type = KokkosComm::Impl::view_mpi_type(tileB);

	// for (int i = 0; i < comm_size; ++i) {
	// 	auto c_tile_layout_parameter = mpl::subarray_layout<num_t>::parameter{
	// 		/* first dimension */ {NI, SI, /* index of the first element */ SI * (i / (comm_size / 2))},
	// 		/* second dimension */ {NJ, SJ, /* index of the first element */ SJ * (i % (comm_size / 2))}};

	const int c_tile_displacement = &tileC((comm_rank / (comm_size / 2)) * SI, (comm_rank % (comm_size / 2)) * SJ) - tileC.data();
	const auto c_subview = Kokkos::subview(tileC, Kokkos::make_pair<int, int>(0, SI), Kokkos::make_pair<int, int>(0, SJ));
	const auto c_subview_type_pre = KokkosComm::Impl::view_mpi_type(c_subview);
	MPI_Datatype c_subview_type = MPI_DATATYPE_NULL;
	MPI_Type_create_resized(c_subview_type_pre, 0, sizeof(num_t), &c_subview_type);

	// 	auto a_tile_layout_parameter = mpl::subarray_layout<num_t>::parameter{
	// 		/* first dimension */ {NI, SI, /* index of the first element */ SI * (i / (comm_size / 2))},
	// 		/* second dimension */ {NK, NK, /* index of the first element */ 0}};

	const int a_tile_displacement = &tileA((comm_rank / (comm_size / 2)) * SI, 0) - tileA.data();
	const auto a_subview = Kokkos::subview(tileA, Kokkos::make_pair<int, int>(0, SI), Kokkos::ALL);
	const auto a_subview_type_pre = KokkosComm::Impl::view_mpi_type(a_subview);
	MPI_Datatype a_subview_type = MPI_DATATYPE_NULL;
	MPI_Type_create_resized(a_subview_type_pre, 0, sizeof(num_t), &a_subview_type);

	// 	auto b_tile_layout_parameter = mpl::subarray_layout<num_t>::parameter{
	// 		/* first dimension */ {NJ, SJ, /* index of the first element */ SJ * (i % (comm_size / 2))},
	// 		/* second dimension */ {NK, NK, /* index of the first element */ 0},
	// 	};

	const int b_tile_displacement = &tileB((comm_rank % (comm_size / 2)) * SJ, 0) - tileB.data();
	const auto b_subview = Kokkos::subview(tileB, Kokkos::make_pair<int, int>(0, SJ), Kokkos::ALL);
	const auto b_subview_type_pre = KokkosComm::Impl::view_mpi_type(b_subview);
	MPI_Datatype b_subview_type = MPI_DATATYPE_NULL;
	MPI_Type_create_resized(b_subview_type_pre, 0, sizeof(num_t), &b_subview_type);

	MPI_Type_commit(&c_subview_type);
	MPI_Type_commit(&a_subview_type);
	MPI_Type_commit(&b_subview_type);

	MPI_Allgather(&c_tile_displacement, 1, MPI_INT, c_displacements.data(), 1, MPI_INT, handle.mpi_comm());
	MPI_Allgather(&a_tile_displacement, 1, MPI_INT, a_displacements.data(), 1, MPI_INT, handle.mpi_comm());
	MPI_Allgather(&b_tile_displacement, 1, MPI_INT, b_displacements.data(), 1, MPI_INT, handle.mpi_comm());

	// 	const auto c_tile_layout = mpl::subarray_layout<num_t>{c_tile_layout_parameter};
	// 	const auto a_tile_layout = mpl::subarray_layout<num_t>{a_tile_layout_parameter};
	// 	const auto b_tile_layout = mpl::subarray_layout<num_t>{b_tile_layout_parameter};

	// 	c_layouts.push_back(c_tile_layout);
	// 	a_layouts.push_back(a_tile_layout);
	// 	b_layouts.push_back(b_tile_layout);
	// }

	const auto start = chrono::high_resolution_clock::now();

	// const auto c_layout = mpl::contiguous_layout<num_t>{NI * NJ};
	// const auto a_layout = mpl::contiguous_layout<num_t>{NI * NK};
	// const auto b_layout = mpl::contiguous_layout<num_t>{NK * NJ};

	// const auto c_tile_layout =  mpl::contiguous_layout<num_t>{SI * SJ};
	// const auto a_tile_layout =  mpl::contiguous_layout<num_t>{SI * NK};
	// const auto b_tile_layout =  mpl::contiguous_layout<num_t>{SJ * NK};

	// comm_world.scatterv(root, C_data.get(), c_layouts, tileC_data.get(), c_tile_layout);
	// comm_world.scatterv(root, A_data.get(), a_layouts, tileA_data.get(), a_tile_layout);
	// comm_world.scatterv(root, B_data.get(), b_layouts, tileB_data.get(), b_tile_layout);

	MPI_Scatterv(C.data(), send_counts.data(), c_displacements.data(), c_subview_type,
				 tileC.data(), 1, c_tile_type, root, handle.mpi_comm());
	MPI_Scatterv(A.data(), send_counts.data(), a_displacements.data(), a_subview_type,
				 tileA.data(), 1, a_tile_type, root, handle.mpi_comm());
	MPI_Scatterv(B.data(), send_counts.data(), b_displacements.data(), b_subview_type,
				 tileB.data(), 1, b_tile_type, root, handle.mpi_comm());

	// // kernel_gemm(alpha, tileC, beta, tileA, tileB, SI, SJ, NK);

	kernel_gemm(alpha, tileC, beta, tileA, tileB, SI, SJ, NK);

	// // comm_world.gatherv(root, tileC_data.get(), c_tile_layout, C_data.get(), c_layouts);
	// // comm_world.gatherv(root, tileA_data.get(), a_tile_layout, A_data.get(), a_layouts);
	// // comm_world.gatherv(root, tileB_data.get(), b_tile_layout, B_data.get(), b_layouts);

	MPI_Gatherv(tileC.data(), 1, c_tile_type, C.data(), send_counts.data(), c_displacements.data(), c_subview_type, root, handle.mpi_comm());
	MPI_Gatherv(tileA.data(), 1, a_tile_type, A.data(), send_counts.data(), a_displacements.data(), a_subview_type, root, handle.mpi_comm());
	MPI_Gatherv(tileB.data(), 1, b_tile_type, B.data(), send_counts.data(), b_displacements.data(), b_subview_type, root, handle.mpi_comm());

	const auto end = chrono::high_resolution_clock::now();

	const auto duration = chrono::duration<double>(end - start);

	// print results
	if (comm_rank == root) {
		std::cerr << std::fixed << std::setprecision(6);
		std::cerr << duration.count() << std::endl;
		if (argc > 0 && argv[0] != ""s) {
			std::cout << std::fixed << std::setprecision(2);
			for (auto i = 0; i < NI; ++i) {
				for (auto j = 0; j < NJ; ++j) {
					std::cout << C(i, j) << std::endl;
				}
			}
		}
	}

	KokkosComm::barrier(auto(handle));
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
