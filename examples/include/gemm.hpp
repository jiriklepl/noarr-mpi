#ifndef GEMM_HPP
#define GEMM_HPP

#include "defines.hpp"

#ifdef MINI_DATASET
#	define NI 20
#	define NJ 28
#	define NK 30
#elif defined(SMALL_DATASET)
#	define NI 64
#	define NJ 192
#	define NK 128
#elif defined(MEDIUM_DATASET)
#	define NI 200
#	define NJ 220
#	define NK 240
#elif defined(LARGE_DATASET)
#	define NI 1000
#	define NJ 1100
#	define NK 1200
#elif defined(EXTRALARGE_DATASET)
#	define NI 2048
#	define NJ 2560
#	define NK 1408
#endif

#endif // GEMM_HPP
