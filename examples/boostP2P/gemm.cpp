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
#include <variant>

#include <experimental/mdspan>

#include <boost/mpi.hpp>

#include "common.hpp"
#include "defines.hpp"
#include "gemm.hpp"

namespace stdex = std::experimental;
namespace mpi = boost::mpi;

using num_t = DATA_TYPE;

namespace {

template<typename MDSpan>
void serialize(mpi::packed_oarchive &ar, MDSpan m) {
	using index_type = typename std::remove_reference_t<MDSpan>::index_type;

	for (index_type i = 0; i < m.extent(0); ++i) {
		for (index_type j = 0; j < m.extent(1); ++j) {
			ar << m[i, j];
		}
	}
}

template<typename MDSpan>
void deserialize(mpi::packed_iarchive &ar, MDSpan m) {
	using index_type = typename std::remove_reference_t<MDSpan>::index_type;

	for (index_type i = 0; i < m.extent(0); ++i) {
		for (index_type j = 0; j < m.extent(1); ++j) {
			ar >> m[i, j];
		}
	}
}

template<typename MDSpan>
class matrix_factory {
public:
	using data_handle_type = typename MDSpan::data_handle_type;

	constexpr MDSpan operator()(data_handle_type data, std::size_t rows, std::size_t cols) const {
		return MDSpan{data, rows, cols};
	}
};

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
                                             std::size_t SJ, mpi::communicator &world, int rank, int size, int root) {
	std::chrono::high_resolution_clock::time_point start;

	std::vector<decltype(stdex::submdspan(C, std::tuple<std::size_t, std::size_t>(0, 0),
										std::tuple<std::size_t, std::size_t>(0, 0)))>
		c_layouts;
	std::vector<decltype(stdex::submdspan(A, std::tuple<std::size_t, std::size_t>(0, 0),
										std::tuple<std::size_t, std::size_t>(0, 0)))>
		a_layouts;
	std::vector<decltype(stdex::submdspan(B, std::tuple<std::size_t, std::size_t>(0, 0),
										std::tuple<std::size_t, std::size_t>(0, 0)))>
		b_layouts;

	{

		std::vector<std::variant<std::monostate, mpi::packed_oarchive>> coarchives(size);
		std::vector<std::variant<std::monostate, mpi::packed_oarchive>> aoarchives(size);
		std::vector<std::variant<std::monostate, mpi::packed_oarchive>> boarchives(size);

		std::vector<mpi::request> requests;
		requests.reserve(rank == root ? 3 * (size + 1) : 3);

		if (rank == root) {
			c_layouts.resize(size);
			a_layouts.resize(size);
			b_layouts.resize(size);

			for (int r = 0; r < size; ++r) {
				const int i = r / j_tiles;
				const int j = r % j_tiles;

				c_layouts[r] =
					stdex::submdspan(C,
				                     /* first dimension */ std::tuple<std::size_t, std::size_t>{SI * i, SI * (i + 1)},
				                     /* second dimension */ std::tuple<std::size_t, std::size_t>{SJ * j, SJ * (j + 1)});

				a_layouts[r] =
					stdex::submdspan(A,
				                     /* first dimension */ std::tuple<std::size_t, std::size_t>{SI * i, SI * (i + 1)},
				                     /* second dimension */ stdex::full_extent);

				b_layouts[r] =
					stdex::submdspan(B,
				                     /* first dimension */ stdex::full_extent,
				                     /* second dimension */ std::tuple<std::size_t, std::size_t>{SJ * j, SJ * (j + 1)});
			}
		}

		world.barrier();
		start = std::chrono::high_resolution_clock::now();

		if (rank == root) {
			for (int r = 0; r < size; ++r) {
				auto &coarchive =
					coarchives[r].emplace<mpi::packed_oarchive>(world, c_layouts[r].size() * sizeof(num_t));
				serialize(coarchive, c_layouts[r]);
				requests.emplace_back(world.isend(r, 3 * r + 0, coarchive));
			}
			for (int r = 0; r < size; ++r) {
				auto &aoarchive =
					aoarchives[r].emplace<mpi::packed_oarchive>(world, a_layouts[r].size() * sizeof(num_t));
				serialize(aoarchive, a_layouts[r]);
				requests.emplace_back(world.isend(r, 3 * r + 1, aoarchive));
			}
			for (int r = 0; r < size; ++r) {
				auto &boarchive =
					boarchives[r].emplace<mpi::packed_oarchive>(world, b_layouts[r].size() * sizeof(num_t));
				serialize(boarchive, b_layouts[r]);
				requests.emplace_back(world.isend(r, 3 * r + 2, boarchive));
			}
		}

		mpi::packed_iarchive ciarchive(world, tileC.size() * sizeof(num_t));
		mpi::packed_iarchive aiarchive(world, tileA.size() * sizeof(num_t));
		mpi::packed_iarchive biarchive(world, tileB.size() * sizeof(num_t));

		requests.emplace_back(world.irecv(root, 3 * rank + 0, ciarchive));
		requests.emplace_back(world.irecv(root, 3 * rank + 1, aiarchive));
		requests.emplace_back(world.irecv(root, 3 * rank + 2, biarchive));

		mpi::wait_all(requests.begin(), requests.end());

		deserialize(ciarchive, tileC);
		deserialize(aiarchive, tileA);
		deserialize(biarchive, tileB);
	}

	kernel_gemm(alpha, tileC, beta, tileA, tileB, SI, SJ, NK);

	{
		std::vector<mpi::request> requests;
		requests.reserve(rank == root ? size + 1 : 1);

		mpi::packed_oarchive coarchive(world, tileC.size() * sizeof(num_t));
		serialize(coarchive, tileC);
		requests.emplace_back(world.isend(root, 3 * size + rank, coarchive));

		std::vector<std::variant<std::monostate, mpi::packed_iarchive>> ciarchives(size);

		if (rank == root) {
			for (int r = 0; r < size; ++r) {
				auto &ciarchive =
					ciarchives[r].emplace<mpi::packed_iarchive>(world, c_layouts[r].size() * sizeof(num_t));
				requests.emplace_back(world.irecv(r, 3 * size + r, ciarchive));
			}
		}

		mpi::wait_all(requests.begin(), requests.end());

		if (rank == root) {
			for (int r = 0; r < size; ++r) {
				deserialize(std::get<mpi::packed_iarchive>(ciarchives[r]), c_layouts[r]);
			}
		}
	}

	world.barrier();

	const auto end = std::chrono::high_resolution_clock::now();

	return end - start;
}

} // namespace

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	constexpr int num_runs = 20;

	mpi::environment env(argc, argv);
	mpi::communicator world;

	const int rank = world.rank();
	const int size = world.size();
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

	auto tileC = tuning.c_tile_layout(tileC_data.get(), SI, SJ);
	auto tileA = tuning.a_tile_layout(tileA_data.get(), SI, NK);
	auto tileB = tuning.b_tile_layout(tileB_data.get(), NK, SJ);

	num_t alpha{};
	num_t beta{};

	if (rank == root) {
		init_array(alpha, C, beta, A, B);
	}

	mpi::broadcast(world, alpha, root);
	mpi::broadcast(world, beta, root);

	// Warm up
	run_experiment(alpha, beta, C, A, B, i_tiles, j_tiles, tileC, tileA, tileB, SI, SJ, world, rank, size, root);

	std::vector<double> times(num_runs);

	for (int i = 0; i < num_runs; ++i) {
		if (rank == root) {
			init_array(alpha, C, beta, A, B);
		}

		world.barrier();

		times[i] =
			run_experiment(alpha, beta, C, A, B, i_tiles, j_tiles, tileC, tileA, tileB, SI, SJ, world, rank, size, root)
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

				for (auto i = 0; i < NI; ++i) {
					for (auto j = 0; j < NJ; ++j) {
						check << C[i, j] << '\n';
					}
				}

				if (!check.is_valid()) {
					std::cerr << "Validation failed!" << '\n';
					return_code = EXIT_FAILURE;
				}
			} else {
				std::cerr << std::fixed << std::setprecision(2);
				for (auto i = 0; i < NI; ++i) {
					for (auto j = 0; j < NJ; ++j) {
						std::cerr << C[i, j] << '\n';
					}
				}
			}
		}
	}

	mpi::broadcast(world, return_code, root);

	return return_code;
}
