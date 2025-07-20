#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include <experimental/mdspan>

#include "common.hpp"
#include "defines.hpp"
#include "gemm.hpp"

namespace stdex = std::experimental;

using num_t = DATA_TYPE;

namespace {

template<typename MDSpan>
class matrix_factory {
public:
	using data_handle_type = typename MDSpan::data_handle_type;

	constexpr MDSpan operator()(data_handle_type data, std::size_t rows, std::size_t cols) const {
		return MDSpan{data, rows, cols};
	}
};

class RAII_Datatype {
public:
	explicit RAII_Datatype(MPI_Datatype type) : _type{type} { MPI_Type_commit(&_type); }

	RAII_Datatype(const RAII_Datatype &) = delete;
	RAII_Datatype &operator=(const RAII_Datatype &) = delete;

	RAII_Datatype(RAII_Datatype &&other) : _type{other._type} { other._type = MPI_DATATYPE_NULL; }

	RAII_Datatype &operator=(RAII_Datatype &&other) {
		using std::swap;
		swap(_type, other._type);
		return *this;
	}

	~RAII_Datatype() {
		if (_type != MPI_DATATYPE_NULL) {
			MPI_Type_free(&_type);
		}
	}

	MPI_Datatype get() const { return _type; }

	operator MPI_Datatype() const { return _type; }

private:
	MPI_Datatype _type;
};

template<typename T>
class mpi_type;

template<>
class mpi_type<int> {
public:
	static MPI_Datatype get() { return MPI_INT; }
};

template<>
class mpi_type<float> {
public:
	static MPI_Datatype get() { return MPI_FLOAT; }
};

template<>
class mpi_type<double> {
public:
	static MPI_Datatype get() { return MPI_DOUBLE; }
};

template<typename MDSpan>
RAII_Datatype create_mpi_datatype(const MDSpan &mdspan) {
	using value_type = typename std::remove_cvref_t<typename MDSpan::element_type>;
	using index_type = typename std::remove_cvref_t<typename MDSpan::index_type>;

	constexpr std::size_t rank = MDSpan::rank();

	MPI_Datatype type{mpi_type<value_type>::get()};

	for (int i = rank - 1; i >= 0; --i) {
		const std::size_t dim = mdspan.extent(i);
		const std::size_t stride = mdspan.stride(i);

		MPI_Datatype subarray_type = MPI_DATATYPE_NULL;
		MPI_Type_create_hvector(dim, 1, stride * sizeof(value_type), type, &subarray_type);

		if (i != rank - 1) {
			MPI_Type_free(&type);
		}

		type = subarray_type;
	}

	MPI_Datatype subarray_type = MPI_DATATYPE_NULL;
	MPI_Type_create_resized(type, 0, sizeof(value_type), &subarray_type);
	MPI_Type_free(&type);
	type = subarray_type;

	return RAII_Datatype{type};
}

const struct tuning {
private:
	using dyn_2D = stdex::dextents<std::size_t, 2>;

public:
	DEFINE_LAYOUT(c_layout, matrix_factory<stdex::mdspan<num_t, dyn_2D, stdex::layout_right>>{});
	DEFINE_LAYOUT(a_layout, matrix_factory<stdex::mdspan<num_t, dyn_2D, stdex::layout_right>>{});
	DEFINE_LAYOUT(b_layout, matrix_factory<stdex::mdspan<num_t, dyn_2D, stdex::layout_right>>{});

#ifdef C_TILE_J_MAJOR
	DEFINE_LAYOUT(c_tile_layout, matrix_factory<stdex::mdspan<num_t, dyn_2D, stdex::layout_left>>{});
#else
	DEFINE_LAYOUT(c_tile_layout, matrix_factory<stdex::mdspan<num_t, dyn_2D, stdex::layout_right>>{});
#endif

#ifdef A_TILE_K_MAJOR
	DEFINE_LAYOUT(a_tile_layout, matrix_factory<stdex::mdspan<num_t, dyn_2D, stdex::layout_left>>{});
#else
	DEFINE_LAYOUT(a_tile_layout, matrix_factory<stdex::mdspan<num_t, dyn_2D, stdex::layout_right>>{});
#endif

#ifdef B_TILE_J_MAJOR
	DEFINE_LAYOUT(b_tile_layout, matrix_factory<stdex::mdspan<num_t, dyn_2D, stdex::layout_left>>{});
#else
	DEFINE_LAYOUT(b_tile_layout, matrix_factory<stdex::mdspan<num_t, dyn_2D, stdex::layout_right>>{});
#endif
} tuning;

// initialization function
void init_array(num_t &alpha, auto C, num_t &beta, auto A, auto B) {
	// C: i x j
	// A: i x k
	// B: k x j

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	for (std::size_t i = 0; i < NI; ++i) {
		for (std::size_t j = 0; j < NJ; ++j) {
			C[i, j] = (num_t)((i * j + 1) % NI) / NI;
		}
	}

	for (std::size_t i = 0; i < NI; ++i) {
		for (std::size_t k = 0; k < NK; ++k) {
			A[i, k] = (num_t)(i * (k + 1) % NK) / NK;
		}
	}

	for (std::size_t j = 0; j < NJ; ++j) {
		for (std::size_t k = 0; k < NK; ++k) {
			B[k, j] = (num_t)(k * (j + 2) % NJ) / NJ;
		}
	}
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(num_t alpha, auto C, num_t beta, auto A, auto B, std::size_t SI, std::size_t SJ, std::size_t SK) {
	// C: i x j
	// A: i x k
	// B: k x j

	for (std::size_t i = 0; i < SI; ++i) {
		for (std::size_t j = 0; j < SJ; ++j) {
			C[i, j] *= beta;
		}

		for (std::size_t j = 0; j < SJ; ++j) {
			for (std::size_t k = 0; k < SK; ++k) {
				C[i, j] += alpha * A[i, k] * B[k, j];
			}
		}
	}
}

std::chrono::duration<double> run_experiment(num_t alpha, num_t beta, auto C, auto A, auto B, std::size_t /*i_tiles*/,
                                             std::size_t j_tiles, auto tileC, auto tileA, auto tileB, std::size_t SI,
                                             std::size_t SJ, MPI_Comm &world, int /*rank*/, int size, int root) {
	std::vector<decltype(stdex::submdspan(C, std::tuple<std::size_t, std::size_t>{},
	                                      std::tuple<std::size_t, std::size_t>{}))>
		c_layouts;
	std::vector<decltype(stdex::submdspan(A, std::tuple<std::size_t, std::size_t>{},
	                                      std::tuple<std::size_t, std::size_t>{}))>
		a_layouts;
	std::vector<decltype(stdex::submdspan(B, std::tuple<std::size_t, std::size_t>{},
	                                      std::tuple<std::size_t, std::size_t>{}))>
		b_layouts;

	c_layouts.reserve(size);
	a_layouts.reserve(size);
	b_layouts.reserve(size);

	std::vector<int> c_displacements;
	std::vector<int> a_displacements;
	std::vector<int> b_displacements;

	c_displacements.reserve(size);
	a_displacements.reserve(size);
	b_displacements.reserve(size);

	std::vector<int> send_counts(size, 1);

	for (int r = 0; r < size; ++r) {
		const int i = r / j_tiles;
		const int j = r % j_tiles;

		c_layouts.emplace_back(
			stdex::submdspan(C,
		                     /* first dimension */ std::tuple<std::size_t, std::size_t>{SI * i, SI * (i + 1)},
		                     /* second dimension */ std::tuple<std::size_t, std::size_t>{SJ * j, SJ * (j + 1)}));
		c_displacements.emplace_back(c_layouts.back().data_handle() - C.data_handle());

		a_layouts.emplace_back(
			stdex::submdspan(A,
		                     /* first dimension */ std::tuple<std::size_t, std::size_t>{SI * i, SI * (i + 1)},
		                     /* second dimension */ stdex::full_extent));
		a_displacements.emplace_back(a_layouts.back().data_handle() - A.data_handle());

		b_layouts.emplace_back(
			stdex::submdspan(B,
		                     /* first dimension */ stdex::full_extent,
		                     /* second dimension */ std::tuple<std::size_t, std::size_t>{SJ * j, SJ * (j + 1)}));
		b_displacements.emplace_back(b_layouts.back().data_handle() - B.data_handle());
	}

	auto cTileType = create_mpi_datatype(tileC);
	auto aTileType = create_mpi_datatype(tileA);
	auto bTileType = create_mpi_datatype(tileB);

	// use the fact that all tiles of a given matrix share the same layout and dimensions
	auto cType = create_mpi_datatype(c_layouts[0]);
	auto aType = create_mpi_datatype(a_layouts[0]);
	auto bType = create_mpi_datatype(b_layouts[0]);

	const auto start = std::chrono::high_resolution_clock::now();

	MPI_Scatterv(c_layouts[0].data_handle(), send_counts.data(), c_displacements.data(), cType.get(),
	             tileC.data_handle(), 1, cTileType.get(), root, world);
	MPI_Scatterv(a_layouts[0].data_handle(), send_counts.data(), a_displacements.data(), aType.get(),
	             tileA.data_handle(), 1, aTileType.get(), root, world);
	MPI_Scatterv(b_layouts[0].data_handle(), send_counts.data(), b_displacements.data(), bType.get(),
	             tileB.data_handle(), 1, bTileType.get(), root, world);

	// run kernel
	kernel_gemm(alpha, tileC, beta, tileA, tileB, SI, SJ, NK);

	MPI_Gatherv(tileC.data_handle(), 1, cTileType.get(), c_layouts[0].data_handle(), send_counts.data(),
	            c_displacements.data(), cType.get(), root, world);

	const auto end = std::chrono::high_resolution_clock::now();

	return end - start;
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	constexpr int num_runs = 20;

	MPI_Init(&argc, &argv);
	MPI_Comm world_comm = MPI_COMM_WORLD;

	int rank = -1;
	int size = -1;
	MPI_Comm_rank(world_comm, &rank);
	MPI_Comm_size(world_comm, &size);
	constexpr int root = 0;

	if (rank == root) {
		std::cerr << "Running with " << size << " processes" << '\n';
	}

	const auto C_data = (rank == root) ? std::make_unique<num_t[]>(NI * NJ) : nullptr;
	const auto A_data = (rank == root) ? std::make_unique<num_t[]>(NI * NK) : nullptr;
	const auto B_data = (rank == root) ? std::make_unique<num_t[]>(NK * NJ) : nullptr;

	const auto C = tuning.c_layout(C_data.get(), NI, NJ);
	const auto A = tuning.a_layout(A_data.get(), NI, NK);
	const auto B = tuning.b_layout(B_data.get(), NK, NJ);

	const int i_tiles = (argc > 1) ? std::atoi(argv[1]) : 1;
	const int j_tiles = size / i_tiles;

	const int SI = NI / i_tiles;
	const int SJ = NJ / j_tiles;

	const auto tileC_data = std::make_unique<num_t[]>((std::size_t)(SI * SJ));
	const auto tileA_data = std::make_unique<num_t[]>((std::size_t)(SI * NK));
	const auto tileB_data = std::make_unique<num_t[]>((std::size_t)(NK * SJ));

	const auto tileC = tuning.c_tile_layout(tileC_data.get(), SI, SJ);
	const auto tileA = tuning.a_tile_layout(tileA_data.get(), SI, NK);
	const auto tileB = tuning.b_tile_layout(tileB_data.get(), NK, SJ);

	num_t alpha{};
	num_t beta{};

	if (rank == root) {
		init_array(alpha, C, beta, A, B);
	}

	MPI_Bcast(&alpha, 1, mpi_type<num_t>::get(), root, world_comm);
	MPI_Bcast(&beta, 1, mpi_type<num_t>::get(), root, world_comm);

	// Warm up
	run_experiment(alpha, beta, C, A, B, i_tiles, j_tiles, tileC, tileA, tileB, SI, SJ, world_comm, rank, size, root);

	std::vector<double> times(num_runs);

	for (int i = 0; i < num_runs; ++i) {
		if (rank == root) {
			init_array(alpha, C, beta, A, B);
		}

		MPI_Barrier(world_comm);

		times[i] = run_experiment(alpha, beta, C, A, B, i_tiles, j_tiles, tileC, tileA, tileB, SI, SJ, world_comm, rank,
		                          size, root)
		               .count();
	}

	const auto [mean, stddev] = mean_stddev(times);
	if (rank == root) {
		std::cout << mean << " " << stddev << '\n';
	}

	int return_code = EXIT_SUCCESS;
	// print results
	if (rank == root) {
		if (argc > 0 && argv[0] != ""s) {
			if (argc > 2) {
				std::ifstream file(argv[2]);
				matrix_stream_check check(file, NI, NJ);

				for (std::size_t i = 0; i < NI; ++i) {
					for (std::size_t j = 0; j < NJ; ++j) {
						check << C[i, j] << '\n';
					}
				}

				if (!check.is_valid()) {
					std::cerr << "Validation failed!" << '\n';
					return_code = EXIT_FAILURE;
				}
			} else {
				std::cerr << std::fixed << std::setprecision(2);
				for (std::size_t i = 0; i < NI; ++i) {
					for (std::size_t j = 0; j < NJ; ++j) {
						std::cerr << C[i, j] << '\n';
					}
				}
			}
		}
	}

	MPI_Bcast(&return_code, 1, MPI_INT, root, world_comm);

	MPI_Finalize();

	return return_code;
}
