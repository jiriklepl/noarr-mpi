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

#include <mpl/mpl.hpp>

#include "common.hpp"
#include "defines.hpp"
#include "gemm.hpp"

using num_t = DATA_TYPE;

namespace {

enum MatrixOrder : std::uint8_t { COrder, FOrder };

constexpr MatrixOrder RowMajor = COrder;
constexpr MatrixOrder ColMajor = FOrder;

template<typename T, MatrixOrder Order = RowMajor>
class matrix {
public:
	using size_type = std::size_t;
	using value_type = T;
	using pointer = T *;
	using reference = T &;
	using const_pointer = const T *;
	using const_reference = const T &;
	using difference_type = std::ptrdiff_t;
	using iterator = T *;
	using const_iterator = const T *;

	constexpr static MatrixOrder order = Order;

private:
	size_type _rows, _cols;
	T *_data;

public:
	matrix(T *data, size_type rows, size_type cols) : _data(data), _rows(rows), _cols(cols) {}

	[[nodiscard]]
	constexpr size_type size() const {
		return _rows * _cols;
	}

	constexpr reference operator()(size_type i_row, size_type i_col) const
	requires (Order == RowMajor)
	{
		return _data[i_row * _cols + i_col];
	}

	constexpr reference operator()(size_type i_row, size_type i_col) const
	requires (Order == ColMajor)
	{
		return _data[i_col * _rows + i_row];
	}

	constexpr reference operator[](size_type i) const { return _data[i]; }

	constexpr pointer begin() const { return _data; }

	constexpr pointer end() const { return _data + size(); }

	constexpr const_pointer cbegin() const { return _data; }

	constexpr const_pointer cend() const { return _data + size(); }

	constexpr pointer data() const { return _data; }

	constexpr const_pointer cdata() const { return _data; }

	constexpr size_type rows() const { return _rows; }

	constexpr size_type cols() const { return _cols; }
};

template<typename T, MatrixOrder Order = RowMajor>
class matrix_factory {
public:
	constexpr static MatrixOrder order = Order;

	matrix<T, Order> operator()(T *data, std::size_t rows, std::size_t cols) const {
		return matrix<T, Order>(data, rows, cols);
	}
};

const struct tuning {
	DEFINE_LAYOUT(c_layout, matrix_factory<num_t, RowMajor>{});
	DEFINE_LAYOUT(a_layout, matrix_factory<num_t, RowMajor>{});
	DEFINE_LAYOUT(b_layout, matrix_factory<num_t, RowMajor>{});

#ifdef C_TILE_J_MAJOR
	DEFINE_LAYOUT(c_tile_layout, matrix_factory<num_t, ColMajor>{});
#else
	DEFINE_LAYOUT(c_tile_layout, matrix_factory<num_t, RowMajor>{});
#endif

#ifdef A_TILE_K_MAJOR
	DEFINE_LAYOUT(a_tile_layout, matrix_factory<num_t, ColMajor>{});
#else
	DEFINE_LAYOUT(a_tile_layout, matrix_factory<num_t, RowMajor>{});
#endif

#ifdef B_TILE_J_MAJOR
	DEFINE_LAYOUT(b_tile_layout, matrix_factory<num_t, ColMajor>{});
#else
	DEFINE_LAYOUT(b_tile_layout, matrix_factory<num_t, RowMajor>{});
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
void kernel_gemm(num_t alpha, auto C, num_t beta, auto A, auto B, std::size_t SI, std::size_t SJ, std::size_t SK) {
	// C: i x j
	// A: i x k
	// B: k x j

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

std::chrono::duration<double> run_experiment(num_t alpha, num_t beta,
                                             num_t *C_data, num_t *A_data, num_t *B_data,
                                             auto C, auto A, auto B, std::size_t /*i_tiles*/,
                                             std::size_t j_tiles,
											 num_t *tileC_data, num_t *tileA_data, num_t *tileB_data,
											 auto tileC, auto tileA, auto tileB, std::size_t SI,
                                             std::size_t SJ, const mpl::communicator & comm_world, int rank, int size,
                                             int root) {
	const auto start = std::chrono::high_resolution_clock::now();

	mpl::layouts<num_t> c_layouts;

	mpl::layouts<num_t> a_layouts;
	mpl::layouts<num_t> b_layouts;

	for (int r = 0; r < size; ++r) {
		const int i = r / j_tiles;
		const int j = r % j_tiles;

		auto c_tile_layout_parameter = mpl::subarray_layout<num_t>::parameter{
			/* first dimension */ {NI, (int)SI, /* index of the first element */ (int)(SI * i)},
			/* second dimension */ {NJ, (int)SJ, /* index of the first element */ (int)(SJ * j)}};

		auto a_tile_layout_parameter = mpl::subarray_layout<num_t>::parameter{
			/* first dimension */ {NI, (int)SI, /* index of the first element */ (int)(SI * i)},
			/* second dimension */ {NK, NK, /* index of the first element */ 0}};

		auto b_tile_layout_parameter = mpl::subarray_layout<num_t>::parameter{
			/* second dimension */ {NK, NK, /* index of the first element */ 0},
			/* first dimension */ {NJ, (int)SJ, /* index of the first element */ (int)(SJ * j)}};

		const auto c_tile_layout = mpl::subarray_layout<num_t>{c_tile_layout_parameter};
		const auto a_tile_layout = mpl::subarray_layout<num_t>{a_tile_layout_parameter};
		const auto b_tile_layout = mpl::subarray_layout<num_t>{b_tile_layout_parameter};

		c_layouts.push_back(c_tile_layout);
		a_layouts.push_back(a_tile_layout);
		b_layouts.push_back(b_tile_layout);
	}

	const auto c_layout = mpl::contiguous_layout<num_t>{NI * NJ};
	const auto a_layout = mpl::contiguous_layout<num_t>{NI * NK};
	const auto b_layout = mpl::contiguous_layout<num_t>{NK * NJ};

	const auto c_tile_layout = mpl::contiguous_layout<num_t>{(std::size_t)(SI * SJ)};
	const auto a_tile_layout = mpl::contiguous_layout<num_t>{(std::size_t)(SI * NK)};
	const auto b_tile_layout = mpl::contiguous_layout<num_t>{(std::size_t)(SJ * NK)};

	comm_world.scatterv(root, C_data, c_layouts, tileC_data, c_tile_layout);
	comm_world.scatterv(root, A_data, a_layouts, tileA_data, a_tile_layout);
	comm_world.scatterv(root, B_data, b_layouts, tileB_data, b_tile_layout);

	kernel_gemm(alpha, tileC, beta, tileA, tileB, SI, SJ, NK);

	comm_world.gatherv(root, tileC_data, c_tile_layout, C_data, c_layouts);

	const auto end = std::chrono::high_resolution_clock::now();

	return end - start;
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	const mpl::communicator &comm_world{mpl::environment::comm_world()};

	// const noarr::MPI_session mpi_session(argc, argv);
	const int rank = comm_world.rank();
	const int size = comm_world.size();
	constexpr int root = 0;

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
	const auto tileB_data = std::make_unique<num_t[]>((std::size_t)(SJ * NK));

	const auto tileC = tuning.c_tile_layout(tileC_data.get(), SI, SJ);
	const auto tileA = tuning.a_tile_layout(tileA_data.get(), SI, NK);
	const auto tileB = tuning.b_tile_layout(tileB_data.get(), NK, SJ);

	num_t alpha{};
	num_t beta{};

	if (rank == root) {
		init_array(alpha, C, beta, A, B);
	}

	comm_world.bcast(root, alpha);
	comm_world.bcast(root, beta);

	const auto duration =
		run_experiment(alpha, beta, C_data.get(), A_data.get(), B_data.get(),
		               C, A, B, i_tiles, j_tiles,
		               tileC_data.get(), tileA_data.get(), tileB_data.get(),
		               tileC, tileA, tileB, SI, SJ, comm_world, rank, size, root);

	int return_code = EXIT_SUCCESS;
	// print results
	if (rank == root) {
		if (argc > 0 && argv[0] != ""s) {
			if (argc > 2) {
				std::ifstream file(argv[2]);
				stream_check check(file);

				for (auto i = 0; i < NI; ++i) {
					for (auto j = 0; j < NJ; ++j) {
						check << C(i, j) << '\n';
					}
				}

				if (!check.is_valid()) {
					std::cerr << "Validation failed!" << std::endl;
					return_code = EXIT_FAILURE;
				}
			} else {
				std::cout << std::fixed << std::setprecision(2);
				for (auto i = 0; i < NI; ++i) {
					for (auto j = 0; j < NJ; ++j) {
						std::cout << C(i, j) << '\n';
					}
				}
			}
		}

		std::cout << std::fixed << std::setprecision(6);
		std::cout << duration.count() << std::endl;
	}

	comm_world.bcast(root, return_code);

	return return_code;
}
