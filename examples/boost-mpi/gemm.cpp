#include <chrono>
#include <iomanip>
#include <iostream>

#include <boost/mpi.hpp>

#include "defines.hpp"
#include "gemm.hpp"

namespace mpi = boost::mpi;

using num_t = DATA_TYPE;

namespace {

enum MatrixOrder : std::uint8_t {
	COrder,
	FOrder
};

constexpr MatrixOrder RowMajor = COrder;
constexpr MatrixOrder ColMajor = FOrder;

template<typename T, MatrixOrder Order = RowMajor>
class matrix {
private:
	friend class boost::serialization::access;

	template<typename Archive>
	void serialize(Archive &ar, const unsigned int /*version*/) {
		for (size_type i_row = 0; i_row < _rows; ++i_row) {
			for (size_type i_col = 0; i_col < _cols; ++i_col) {
				ar & (*this)(i_row, i_col);
			}
		}
	}
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

	matrix(T *data, size_type rows, size_type cols) : _data(data), _rows(rows), _cols(cols) {}

	[[nodiscard]]
	constexpr size_type size() const {
		return _rows * _cols;
	}

	constexpr reference operator()(size_type i_row, size_type i_col) const requires(Order == RowMajor) {
		return _data[i_row * _cols + i_col];
	}

	constexpr reference operator()(size_type i_row, size_type i_col) const requires(Order == ColMajor) {
		return _data[i_col * _rows + i_row];
	}

	constexpr reference operator[](size_type i) const {
		return _data[i];
	}

	constexpr pointer begin() const {
		return _data;
	}

	constexpr pointer end() const {
		return _data + size();
	}

	constexpr const_pointer cbegin() const {
		return _data;
	}

	constexpr const_pointer cend() const {
		return _data + size();
	}

	constexpr pointer data() const {
		return _data;
	}

	constexpr const_pointer cdata() const {
		return _data;
	}

	constexpr size_type rows() const {
		return _rows;
	}

	constexpr size_type cols() const {
		return _cols;
	}

private:
	size_type _rows, _cols;
	T *_data;
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
	DEFINE_PROTO_STRUCT(c_layout, matrix_factory<num_t, RowMajor>{});
	DEFINE_PROTO_STRUCT(a_layout, matrix_factory<num_t, RowMajor>{});
	DEFINE_PROTO_STRUCT(b_layout, matrix_factory<num_t, RowMajor>{});

#ifdef C_TILE_J_MAJOR
	DEFINE_PROTO_STRUCT(c_tile_layout, matrix_factory<num_t, ColMajor>{});
#else
	DEFINE_PROTO_STRUCT(c_tile_layout, matrix_factory<num_t, RowMajor>{});
#endif

#ifdef A_TILE_K_MAJOR
	DEFINE_PROTO_STRUCT(a_tile_layout, matrix_factory<num_t, ColMajor>{});
#else
	DEFINE_PROTO_STRUCT(a_tile_layout, matrix_factory<num_t, RowMajor>{});
#endif

#ifdef B_TILE_J_MAJOR
	DEFINE_PROTO_STRUCT(b_tile_layout, matrix_factory<num_t, ColMajor>{});
#else
	DEFINE_PROTO_STRUCT(b_tile_layout, matrix_factory<num_t, RowMajor>{});
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
	// B: k x j
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

int main(int argc, char *argv[]) {
	using namespace std::string_literals;
	namespace chrono = std::chrono;

	mpi::environment env(argc, argv);
	mpi::communicator comm_world;

	// const noarr::MPI_session mpi_session(argc, argv);
	// const int rank = mpi_get_comm_rank(mpi_session);
	// constexpr int root = 0;

	const int comm_size = comm_world.size();
	const int comm_rank = comm_world.rank();
	constexpr int root = 0;

	// const auto set_lengths = noarr::set_length<'i'>(NI) ^ noarr::set_length<'j'>(NJ) ^ noarr::set_length<'k'>(NK);

	// const auto scalar = noarr::scalar<num_t>();

	// const auto C_structure = scalar ^ tuning.c_layout ^ set_lengths;
	// const auto A_structure = scalar ^ tuning.a_layout ^ set_lengths;
	// const auto B_structure = scalar ^ tuning.b_layout ^ set_lengths;

	// const auto grid_i = noarr::into_blocks<'i', 'I'>();
	// const auto grid_j = noarr::into_blocks<'j', 'J'>();
	// const auto grid = grid_i ^ grid_j;

	// const auto C_data =
	// 	(rank == root) ? std::make_unique<char[]>(C_structure | noarr::get_size()) : std::unique_ptr<char[]>{};
	// const auto A_data =
	// 	(rank == root) ? std::make_unique<char[]>(A_structure | noarr::get_size()) : std::unique_ptr<char[]>{};
	// const auto B_data =
	// 	(rank == root) ? std::make_unique<char[]>(B_structure | noarr::get_size()) : std::unique_ptr<char[]>{};
	const auto C_data = (comm_rank == root) ? std::make_unique<num_t[]>(NI * NJ) : nullptr;
	const auto A_data = (comm_rank == root) ? std::make_unique<num_t[]>(NI * NK) : nullptr;
	const auto B_data = (comm_rank == root) ? std::make_unique<num_t[]>(NK * NJ) : nullptr;

	// const auto C = noarr::bag(C_structure ^ grid, C_data.get());
	// const auto A = noarr::bag(A_structure ^ grid, A_data.get());
	// const auto B = noarr::bag(B_structure ^ grid, B_data.get());
	const auto C = tuning.c_layout(C_data.get(), NI, NJ);
	const auto A = tuning.a_layout(A_data.get(), NI, NK);
	const auto B = tuning.b_layout(B_data.get(), NJ, NK);

	const std::size_t SI = NI / 2;
	const std::size_t SJ = NJ / (comm_size / 2);

	// const auto trav = noarr::traverser(C, A, B) ^ noarr::set_length<'I'>(2) ^ noarr::merge_blocks<'I', 'J', 'r'>();
	// const auto mpi_trav = noarr::mpi_traverser<'r'>(trav, MPI_COMM_WORLD);

	// const auto tileC = noarr::bag(scalar ^ tuning.c_tile_layout ^ lengths_like<'j', 'i'>(mpi_trav));
	// const auto tileA = noarr::bag(scalar ^ tuning.a_tile_layout ^ lengths_like<'k', 'i'>(mpi_trav));
	// const auto tileB = noarr::bag(scalar ^ tuning.b_tile_layout ^ lengths_like<'j', 'k'>(mpi_trav));

	const auto tileC_data = std::make_unique<num_t[]>(SI * SJ);
	const auto tileA_data = std::make_unique<num_t[]>(SI * NK);
	const auto tileB_data = std::make_unique<num_t[]>(SJ * NK);

	const auto tileC = tuning.c_tile_layout(tileC_data.get(), SI, SJ);
	const auto tileA = tuning.a_tile_layout(tileA_data.get(), SI, NK);
	const auto tileB = tuning.b_tile_layout(tileB_data.get(), SJ, NK);

	num_t alpha{};
	num_t beta{};

	// // initialize data
	if (comm_rank == root) {
		init_array(alpha, C, beta, A, B);
	}

	// mpi_bcast(alpha, mpi_trav, root);
	// mpi_bcast(beta, mpi_trav, root);

	mpi::broadcast(comm_world, alpha, root);
	mpi::broadcast(comm_world, beta, root);

	// const auto start = chrono::high_resolution_clock::now();
	// {
	// 	mpi_scatter(C, tileC, mpi_trav, root);
	// 	mpi_scatter(A, tileA, mpi_trav, root);
	// 	mpi_scatter(B, tileB, mpi_trav, root);

	const auto start = chrono::high_resolution_clock::now();
	{

		// TODO
	}

	// 	// run kernel
	// 	kernel_gemm(mpi_trav, alpha, tileC.get_ref(), beta, tileA.get_ref(), tileB.get_ref());

	// 	mpi_gather(tileC, C, mpi_trav, root);
	// }
	// const auto end = chrono::high_resolution_clock::now();

	// const auto duration = chrono::duration<double>(end - start);

	// // print results
	// if (rank == root) {
	// 	std::cerr << std::fixed << std::setprecision(6);
	// 	std::cerr << duration.count() << std::endl;
	// 	if (argc > 0 && argv[0] != ""s) {
	// 		std::cout << std::fixed << std::setprecision(2);
	// 		noarr::serialize_data(std::cout, C.get_ref() ^ set_length(mpi_trav) ^ noarr::hoist<'I', 'i', 'J', 'j'>());
	// 	}
	// }

	// mpi_barrier(mpi_trav);
}
