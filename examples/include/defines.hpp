#ifndef NOARR_POLYBENCH_DEFINES_HPP
#define NOARR_POLYBENCH_DEFINES_HPP

#define STRINGIFY_DETAIL(x) #x
#define STRINGIFY(x) STRINGIFY_DETAIL(x)

#define AUTO_FIELD(name, ...) decltype(__VA_ARGS__) name = __VA_ARGS__

#define DEFINE_LAYOUT(name, ...) AUTO_FIELD(name, __VA_ARGS__)

#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) &&        \
	!defined(EXTRALARGE_DATASET)
#	error "Please define one of MINI_DATASET, SMALL_DATASET, MEDIUM_DATASET, LARGE_DATASET, EXTRALARGE_DATASET"
#	define MINI_DATASET
#endif

#ifdef MINI_DATASET
#	define DATASET_SIZE "MINI_DATASET"
#elif defined(SMALL_DATASET)
#	define DATASET_SIZE "SMALL_DATASET"
#elif defined(MEDIUM_DATASET)
#	define DATASET_SIZE "MEDIUM_DATASET"
#elif defined(LARGE_DATASET)
#	define DATASET_SIZE "LARGE_DATASET"
#elif defined(EXTRALARGE_DATASET)
#	define DATASET_SIZE "EXTRALARGE_DATASET"
#endif

#if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_LONG) && !defined(DATA_TYPE_IS_FLOAT) &&                       \
	!defined(DATA_TYPE_IS_DOUBLE) && !defined(DATA_TYPE)
#	error                                                                                                              \
		"Please define one of DATA_TYPE_IS_INT, DATA_TYPE_IS_LONG, DATA_TYPE_IS_FLOAT, or DATA_TYPE_IS_DOUBLE; or define directly DATA_TYPE to desired type"
#	define DATA_TYPE_IS_FLOAT
#endif

#ifdef DATA_TYPE_IS_INT
#	define DATA_TYPE int
#	define DATA_TYPE_CHOICE "DATA_TYPE_IS_INT"
#elif defined(DATA_TYPE_IS_LONG)
#	define DATA_TYPE long
#	define DATA_TYPE_CHOICE "DATA_TYPE_IS_LONG"
#elif defined(DATA_TYPE_IS_FLOAT)
#	define DATA_TYPE float
#	define DATA_TYPE_CHOICE "DATA_TYPE_IS_FLOAT"
#elif defined(DATA_TYPE_IS_DOUBLE)
#	define DATA_TYPE double
#	define DATA_TYPE_CHOICE "DATA_TYPE_IS_DOUBLE"
#endif

#if !defined(C_TILE_J_MAJOR) && !defined(C_TILE_I_MAJOR)
#	error "Please define one of C_TILE_J_MAJOR or C_TILE_I_MAJOR"
#	define C_TILE_I_MAJOR
#endif

#if !defined(A_TILE_K_MAJOR) && !defined(A_TILE_I_MAJOR)
#	error "Please define one of A_TILE_K_MAJOR or A_TILE_I_MAJOR"
#	define A_TILE_I_MAJOR
#endif

#if !defined(B_TILE_J_MAJOR) && !defined(B_TILE_K_MAJOR)
#	error "Please define one of B_TILE_J_MAJOR or B_TILE_K_MAJOR"
#	define B_TILE_K_MAJOR
#endif

#endif // NOARR_POLYBENCH_DEFINES_HPP
