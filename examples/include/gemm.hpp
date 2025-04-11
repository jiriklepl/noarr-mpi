#ifndef GEMM_HPP
#define GEMM_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
#	define NI 64
#	define NJ 64
#	define NK 64
#elif defined(SMALL_DATASET)
#	define NI 64
#	define NJ 192
#	define NK 128
#elif defined(MEDIUM_DATASET)
#	define NI 320
#	define NJ 384
#	define NK 256
#elif defined(LARGE_DATASET)
#	define NI 1024
#	define NJ 1088
#	define NK 1408
#elif defined(EXTRALARGE_DATASET)
#	define NI 2048
#	define NJ 2560
#	define NK 1408
#endif

#endif // GEMM_HPP
