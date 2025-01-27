#include <chrono>
#include <iomanip>
#include <iostream>

#include <noarr/traversers.hpp>
#include <noarr/introspection.hpp>

#include "noarr/structures/interop/mpi_algorithms.hpp"
#include "noarr/structures/interop/mpi_traverser.hpp"
#include "noarr/structures/interop/mpi_utility.hpp"

#define EXTRALARGE_DATASET
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
void init_array(auto inner, num_t &alpha, num_t &beta, const auto& C, const auto& A, const auto& B) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	alpha = (num_t)1.5;
	beta = (num_t)1.2;

	traverser(C) ^ set_length(inner) | [&](auto state) {
		auto [i, j] = get_indices<'i', 'j'>(state);
		C[state] = (num_t)((i * j + 1) % (C | get_length<'i'>(inner.state()))) / (C | get_length<'i'>(inner.state()));
	};

	traverser(A) ^ set_length(inner) | [&](auto state) {
		auto [i, k] = get_indices<'i', 'k'>(state);
		A[state] = (num_t)(i * (k + 1) % (A | get_length<'k'>(inner.state()))) / (A | get_length<'k'>(inner.state()));
	};

	traverser(B) ^ set_length(inner) | [&](auto state) {
		auto [k, j] = get_indices<'k', 'j'>(state);
		B[state] = (num_t)(k * (j + 2) % (B | get_length<'j'>(inner.state()))) / (B | get_length<'j'>(inner.state()));
	};
}

// computation kernel
[[gnu::flatten, gnu::noinline]]
void kernel_gemm(auto inner, num_t alpha, num_t beta, const auto& subC, const auto& subA, const auto& subB) {
	// C: i x j
	// A: i x k
	// B: k x j
	using namespace noarr;

	inner | for_each<'i', 'j'>([&](auto state) {
		subC[state] *= beta;
	});

	inner | for_each<'i', 'j', 'k'>([&](auto state) {
		subC[state] += alpha * subA[state] * subB[state];
	});
}

} // namespace

// TODO: fails if NJ % 4 != 0 || NI % 2 != 0

namespace noarr {

template<class Structure, IsState State>
auto new_transform_impl(const Structure& structure, const dim_sequence<> &/*unused*/, State state) -> MPI_custom_type {
	// TODO: implement
	using scalar = scalar_t<Structure, State>;
	const auto datatype = choose_mpi_type_v<scalar>();
	std::cerr << "scalar: " << typeid(scalar).name() << " -> " << datatype << std::endl;
	MPI_Datatype new_Datatype = MPI_DATATYPE_NULL;
	MPICHK(MPI_Type_dup(datatype, &new_Datatype));
	std::cerr << "MPI_Type_dup(base: " << datatype << ", derived: " << new_Datatype << ")" << std::endl;
	return MPI_custom_type{new_Datatype};
}

template<auto Dim, class Branches, class Structure, IsState State>
auto new_transform_impl(const Structure& structure, const dim_tree<Dim, Branches> &/*unused*/, State state) -> MPI_custom_type {

	// TODO: constexpr bool contiguous = IsContiguous<Structure, State>;
	constexpr bool has_lower_bound = HasLowerBoundAlong<Structure, Dim, State>;
	constexpr bool has_stride_along = HasStrideAlong<Structure, Dim, State>;
	constexpr bool is_uniform_along = IsUniformAlong<Structure, Dim, State>;
	constexpr bool has_length = Structure::template has_length<Dim, State>();

	if constexpr (has_lower_bound && has_stride_along && is_uniform_along && has_length) {
		const auto lower_bound = lower_bound_along<Dim>(structure, state);
		const auto stride = stride_along<Dim>(structure, state);
		const auto length = structure.template length<Dim>(state);

		// TODO: probably wanna add { lower_bound_assumption(structure, state) } -> IsState
		const MPI_custom_type sub_transformed = new_transform_impl(structure, Branches{}, state);

		if (lower_bound != 0) {
			throw std::runtime_error("Unsupported: lower bound is not zero");
		}

		MPI_Datatype new_Datatype = MPI_DATATYPE_NULL;

		if (stride == 0 || length == 1) {
			MPICHK(MPI_Type_dup((MPI_Datatype)sub_transformed, &new_Datatype));
			std::cerr << "MPI_Type_dup(base: " << (MPI_Datatype)sub_transformed << ", derived: " << new_Datatype << ")" << std::endl;
		} else {
			MPICHK(MPI_Type_create_hvector(length, 1, stride, (MPI_Datatype)sub_transformed, &new_Datatype));
			std::cerr << "MPI_Type_create_hvector(" << length << ", 1, " << stride << ", base: " << (MPI_Datatype)sub_transformed
					  << ", derived: " << new_Datatype << ")" << std::endl;
		}

		return MPI_custom_type{new_Datatype};
	} else {
		if constexpr (!has_lower_bound) {
			throw std::runtime_error("Unsupported: lower bound is not set");
		} else if constexpr (!has_stride_along) {
			throw std::runtime_error("Unsupported: stride is not set");
		} else if constexpr (!is_uniform_along) {
			throw std::runtime_error("Unsupported: non-uniform stride");
		} else if constexpr (!has_length) {
			throw std::runtime_error("Unsupported: length is not set for dimension " + std::to_string(Dim));
		} else {
			throw std::runtime_error("Unsupported transformation");
		}
	}
}

template<class Structure, IsTraverser Trav>
auto new_transform(const Trav& trav, const Structure& structure) {
	using dim_tree = sig_dim_tree<typename decltype(trav.top_struct())::signature>;

	return new_transform_impl(structure, dim_tree{}, trav.state());
}

template<class... Structures, IsTraverser Trav>
auto new_transform(const Trav& trav, const Structures &... structs) {
	return std::make_tuple(new_transform(trav, structs)...);
}

void new_scatter(const auto& from, const auto& to, const IsMpiTraverser auto& trav, int root) {
	const auto from_struct = convert_to_struct(from);
	const auto to_struct = convert_to_struct(to);
	const auto comm = convert_to_MPI_Comm(trav);

	using from_dim_tree = sig_dim_tree<typename decltype(from_struct ^ set_length(trav))::signature>;
	using to_dim_tree = sig_dim_tree<typename decltype(to_struct ^ set_length(trav))::signature>;

	using to_dim_filtered = dim_tree_filter<to_dim_tree, in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>;
	using to_dim_removed =
		dim_tree_filter<to_dim_tree, dim_pred_not<in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>>;

	using from_dim_filtered = dim_tree_filter<from_dim_tree, in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>;
	using from_dim_removed =
		dim_tree_filter<from_dim_tree, dim_pred_not<in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>>;

	// to must be a subset of from
	static_assert(std::is_same_v<to_dim_filtered, to_dim_tree> && std::is_same_v<to_dim_removed, dim_sequence<>>,
	              R"(The "from" structure must be a subset of the "to" structure)");

	// TODO: this is incomplete
	const auto from_rep = new_transform_impl(from_struct, to_dim_filtered{}, trav.state());
	const auto to_rep = new_transform_impl(to_struct, to_dim_filtered{}, trav.state());

	// TODO: the following may be incorrect
	const auto difference_size = mpi_get_comm_size(comm);

	std::vector<int> displacements(difference_size);
	const std::vector<int> sendcounts(difference_size, 1);

	const int offset = (from_struct ^ set_length(trav) ^ fix(trav)) | noarr::offset(fix_zeros(from_dim_tree{}));

	MPICHK(MPI_Allgather(&offset, 1, MPI_INT, displacements.data(), 1, MPI_INT, comm));

	// compute the second smallest displacement
	int min_displacement = std::numeric_limits<int>::max();
	int second_min_displacement = std::numeric_limits<int>::max();
	for (const auto displacement : displacements) {
		if (displacement < min_displacement) {
			second_min_displacement = min_displacement;
			min_displacement = displacement;
		} else if (displacement < second_min_displacement && displacement != min_displacement) {
			second_min_displacement = displacement;
		}
	}

	MPI_Datatype from_rep_resized = MPI_DATATYPE_NULL;
	MPICHK(MPI_Type_create_resized(convert_to_MPI_Datatype(from_rep), 0, second_min_displacement, &from_rep_resized));
	const MPI_custom_type from_rep_resized_custom(from_rep_resized);

	for (auto &displacement : displacements) {
		displacement /= second_min_displacement;
	}

	MPICHK(MPI_Scatterv(from.data(), sendcounts.data(), displacements.data(), convert_to_MPI_Datatype(from_rep_resized),
	                    to.data(), 1, convert_to_MPI_Datatype(to), root, comm));
}

void new_gather(const auto& from, const auto& to, const IsMpiTraverser auto& trav, int root) {
	const auto from_struct = convert_to_struct(from);
	const auto to_struct = convert_to_struct(to);
	const auto comm = convert_to_MPI_Comm(trav);

	using from_dim_tree = sig_dim_tree<typename decltype(from_struct ^ set_length(trav))::signature>;
	using to_dim_tree = sig_dim_tree<typename decltype(to_struct ^ set_length(trav))::signature>;

	using to_dim_filtered = dim_tree_filter<to_dim_tree, in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>;
	using to_dim_removed =
		dim_tree_filter<to_dim_tree, dim_pred_not<in_signature<typename decltype(from_struct ^ set_length(trav))::signature>>>;

	using from_dim_filtered = dim_tree_filter<from_dim_tree, in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>;
	using from_dim_removed =
		dim_tree_filter<from_dim_tree, dim_pred_not<in_signature<typename decltype(to_struct ^ set_length(trav))::signature>>>;

	// from must be a subset of to
	static_assert(std::is_same_v<from_dim_filtered, from_dim_tree> && std::is_same_v<from_dim_removed, dim_sequence<>>,
	              R"(The "to" structure must be a subset of the "from" structure)");

	// TODO: this is incomplete
	const auto from_rep = new_transform_impl(from_struct, from_dim_filtered{}, trav.state());
	const auto to_rep = new_transform_impl(to_struct, from_dim_filtered{}, trav.state());

	// TODO: the following may be incorrect
	const auto difference_size = mpi_get_comm_size(comm);

	std::vector<int> displacements(difference_size);
	const std::vector<int> sendcounts(difference_size, 1);

	const int offset = (to_struct ^ set_length(trav) ^ fix(trav)) | noarr::offset(fix_zeros(to_dim_tree{}));

	MPICHK(MPI_Allgather(&offset, 1, MPI_INT, displacements.data(), 1, MPI_INT, comm));

	// compute the second smallest displacement
	int min_displacement = std::numeric_limits<int>::max();
	int second_min_displacement = std::numeric_limits<int>::max();
	for (const auto displacement : displacements) {
		if (displacement < min_displacement) {
			second_min_displacement = min_displacement;
			min_displacement = displacement;
		} else if (displacement < second_min_displacement && displacement != min_displacement) {
			second_min_displacement = displacement;
		}
	}

	MPI_Datatype to_rep_resized = MPI_DATATYPE_NULL;
	MPICHK(MPI_Type_create_resized(convert_to_MPI_Datatype(to_rep), 0, second_min_displacement, &to_rep_resized));
	const MPI_custom_type to_rep_resized_custom(to_rep_resized);

	for (auto &displacement : displacements) {
		displacement /= second_min_displacement;
	}

	MPICHK(MPI_Gatherv(from.data(), 1, convert_to_MPI_Datatype(from), to.data(), sendcounts.data(),
	                   displacements.data(), convert_to_MPI_Datatype(to_rep_resized), root, comm));
}

} // namespace noarr

int main(int argc, char *argv[]) {
	using namespace std::string_literals;

	const noarr::MPI_session mpi_session(argc, argv);
	const int rank = mpi_get_comm_rank(mpi_session);

	const auto set_lengths = noarr::set_length<'i'>(NI) ^ noarr::set_length<'j'>(NJ) ^ noarr::set_length<'k'>(NK);

	const auto scalar = noarr::scalar<num_t>();

	const auto C_structure = scalar ^ tuning.c_layout ^ set_lengths;
	const auto A_structure = scalar ^ tuning.a_layout ^ set_lengths;
	const auto B_structure = scalar ^ tuning.b_layout ^ set_lengths;

	const auto grid_i = noarr::into_blocks<'i', 'I'>();
	const auto grid_j = noarr::into_blocks<'j', 'J'>();

	const auto C_data = (rank == 0) ? std::make_unique<char[]>(C_structure | noarr::get_size()) : std::unique_ptr<char[]>{};
	const auto A_data = (rank == 0) ? std::make_unique<char[]>(A_structure | noarr::get_size()) : std::unique_ptr<char[]>{};
	const auto B_data = (rank == 0) ? std::make_unique<char[]>(B_structure | noarr::get_size()) : std::unique_ptr<char[]>{};

	const auto C = noarr::bag(C_structure ^ grid_i ^ grid_j, C_data.get());
	const auto A = noarr::bag(A_structure ^ grid_i, A_data.get());
	const auto B = noarr::bag(B_structure ^ grid_j, B_data.get());

	const auto trav = noarr::traverser(C, A, B) ^ noarr::set_length<'I'>(2) ^ noarr::merge_blocks<'I', 'J', 'r'>();
	const auto mpi_trav = noarr::mpi_traverser<'r'>(trav, MPI_COMM_WORLD);

	const auto subC = noarr::bag(scalar ^ vectors_like<'j', 'i'>(mpi_trav));
	const auto subA = noarr::bag(scalar ^ vectors_like<'i', 'k'>(mpi_trav));
	const auto subB = noarr::bag(scalar ^ vectors_like<'k', 'j'>(mpi_trav));

	mpi_run(mpi_trav, C, A, B, subC, subA, subB) | [=](const auto inner, const auto C, const auto A, const auto B, const auto subC, const auto subA, const auto subB) {
		num_t alpha{};
		num_t beta{};

		// initialize data
		if (rank == 0) {
			init_array(inner, alpha, beta, bag(C_structure, C.data()), bag(A_structure, A.data()), bag(B_structure, B.data()));
		}

		mpi_bcast(alpha, inner, 0);
		mpi_bcast(beta, inner, 0);


		auto start = std::chrono::high_resolution_clock::now();
		new_scatter(C, subC, inner, 0);
		new_scatter(A, subA, inner, 0);
		new_scatter(B, subB, inner, 0);

		// run kernel
		kernel_gemm(inner, alpha, beta, subC, subA, subB);

		new_gather(subC, C, inner, 0);

		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration<long double>(end - start);



		// print results
		if (rank == 0) {
			std::cerr << std::fixed << std::setprecision(6);
			std::cerr << duration.count() << std::endl;
			if (argc > 0 && argv[0] != ""s) {
				std::cout << std::fixed << std::setprecision(2);
				noarr::serialize_data(std::cout, C.get_ref() ^ set_length(inner) ^ noarr::hoist<'I', 'i', 'J', 'j'>());
			}
		}
	};
}
