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

#include <experimental/mdspan>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/mpi.hpp>

#include "common.hpp"
#include "defines.hpp"
#include "gemm.hpp"

namespace stdex = std::experimental;
namespace mpi = boost::mpi;

using num_t = DATA_TYPE;

namespace {

template<typename MDSpan>
class matrix {
public:
	using index_type = typename MDSpan::index_type;

	matrix() = default;

	explicit matrix(MDSpan data) : _data(data) {}

	const MDSpan &mdspan() const { return _data; }

	MDSpan &mdspan() { return _data; }

private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version) const {
		for (index_type i_row = 0; i_row < _data.extent(0); ++i_row) {
			for (index_type i_col = 0; i_col < _data.extent(1); ++i_col) {
				ar &_data[i_row, i_col];
			}
		}
	}

	MDSpan _data;
};

template<typename MDSpan>
class matrix_factory {
public:
	using data_handle_type = typename MDSpan::data_handle_type;

	constexpr matrix<MDSpan> operator()(data_handle_type data, std::size_t rows, std::size_t cols) const {
		return matrix<MDSpan>{MDSpan{data, rows, cols}};
	}
};

const struct tuning {
	DEFINE_LAYOUT(c_layout,
	                    matrix_factory<stdex::mdspan<num_t, stdex::dextents<std::size_t, 2>, stdex::layout_right>>{});
	DEFINE_LAYOUT(a_layout,
	                    matrix_factory<stdex::mdspan<num_t, stdex::dextents<std::size_t, 2>, stdex::layout_right>>{});
	DEFINE_LAYOUT(b_layout,
	                    matrix_factory<stdex::mdspan<num_t, stdex::dextents<std::size_t, 2>, stdex::layout_right>>{});

#ifdef C_TILE_J_MAJOR
	DEFINE_LAYOUT(c_tile_layout,
	                    matrix_factory<stdex::mdspan<num_t, stdex::dextents<std::size_t, 2>, stdex::layout_left>>{});
#else
	DEFINE_LAYOUT(c_tile_layout,
	                    matrix_factory<stdex::mdspan<num_t, stdex::dextents<std::size_t, 2>, stdex::layout_right>>{});
#endif

#ifdef A_TILE_K_MAJOR
	DEFINE_LAYOUT(a_tile_layout,
	                    matrix_factory<stdex::mdspan<num_t, stdex::dextents<std::size_t, 2>, stdex::layout_left>>{});
#else
	DEFINE_LAYOUT(a_tile_layout,
	                    matrix_factory<stdex::mdspan<num_t, stdex::dextents<std::size_t, 2>, stdex::layout_right>>{});
#endif

#ifdef B_TILE_J_MAJOR
	DEFINE_LAYOUT(b_tile_layout,
	                    matrix_factory<stdex::mdspan<num_t, stdex::dextents<std::size_t, 2>, stdex::layout_left>>{});
#else
	DEFINE_LAYOUT(b_tile_layout,
	                    matrix_factory<stdex::mdspan<num_t, stdex::dextents<std::size_t, 2>, stdex::layout_right>>{});
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

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;
	namespace chrono = std::chrono;

	mpi::environment env(argc, argv);
	mpi::communicator world;

	// const noarr::MPI_session mpi_session(argc, argv);
	const int rank = world.rank();
	const int size = world.size();
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

	auto tileC = tuning.c_tile_layout(tileC_data.get(), SI, SJ);
	auto tileA = tuning.a_tile_layout(tileA_data.get(), SI, NK);
	auto tileB = tuning.b_tile_layout(tileB_data.get(), NK, SJ);

	num_t alpha{};
	num_t beta{};

	if (rank == root) {
		init_array(alpha, C.mdspan(), beta, A.mdspan(), B.mdspan());
	}

	mpi::broadcast(world, alpha, root);
	mpi::broadcast(world, beta, root);

	std::vector<matrix<std::decay_t<decltype(C.mdspan())>>> c_layouts;
	std::vector<matrix<std::decay_t<decltype(A.mdspan())>>> a_layouts;
	std::vector<matrix<std::decay_t<decltype(B.mdspan())>>> b_layouts;

	for (int r = 0; r < size; ++r) {
		const int i = r / j_tiles;
		const int j = r % j_tiles;

		c_layouts.emplace_back(
			stdex::submdspan(C.mdspan(),
		                     /* first dimension */ std::tuple<std::size_t, std::size_t>{SI * i, SI * (i + 1)},
		                     /* second dimension */ std::tuple<std::size_t, std::size_t>{SJ * j, SJ * (j + 1)}));

		a_layouts.emplace_back(
			stdex::submdspan(A.mdspan(),
		                     /* first dimension */ std::tuple<std::size_t, std::size_t>{SI * i, SI * (i + 1)},
		                     /* second dimension */ stdex::full_extent));

		b_layouts.emplace_back(
			stdex::submdspan(B.mdspan(),
		                     /* first dimension */ stdex::full_extent,
		                     /* second dimension */ std::tuple<std::size_t, std::size_t>{SJ * j, SJ * (j + 1)}));
	}

	const auto start = chrono::high_resolution_clock::now();

	mpi::scatter(world, c_layouts, tileC, root);
	mpi::scatter(world, a_layouts, tileA, root);
	mpi::scatter(world, b_layouts, tileB, root);

	kernel_gemm(alpha, tileC.mdspan(), beta, tileA.mdspan(), tileB.mdspan(), SI, SJ, NK);

	// world.gatherv(root, tileC_data.get(), c_tile_layout, C_data.get(), c_layouts);
	mpi::gather(world, tileC, c_layouts, root);

	const auto end = chrono::high_resolution_clock::now();

	const auto duration = chrono::duration<double>(end - start);

	int return_code = EXIT_SUCCESS;
	// print results
	if (rank == root) {
		if (argc > 0 && argv[0] != ""s) {
			if (argc > 2) {
				std::ifstream file(argv[2]);
				stream_check check(file);

				for (auto i = 0; i < NI; ++i) {
					for (auto j = 0; j < NJ; ++j) {
						check << C.mdspan()[i, j] << '\n';
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
						std::cout << C.mdspan()[i, j] << '\n';
					}
				}
			}
		}

		std::cout << std::fixed << std::setprecision(6);
		std::cout << duration.count() << std::endl;
	}

	mpi::broadcast(world, return_code, root);

	return return_code;
}
