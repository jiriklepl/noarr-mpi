#ifndef NOARR_POLYBENCH_COMMON_HPP
#define NOARR_POLYBENCH_COMMON_HPP

#include <format>
#include <iomanip>
#include <iostream>
#include <type_traits>

class matrix_stream_check {
public:
	static constexpr std::size_t max_invalid = 10;

	explicit matrix_stream_check(std::istream &f, std::size_t ni, std::size_t nj) : _file(&f), _ni(ni), _nj(nj) {}

	matrix_stream_check &operator<<(char /* v */) {
		// ignore
		return *this;
	}

	template<typename T>
	matrix_stream_check &operator<<(T v) {
		std::string result;

		if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
			result = std::format("{:.2f}", v);
		} else if constexpr (std::is_same_v<T, int>) {
			result = std::format("{}", v);
		}

		std::string expected;
		(*_file) >> expected;

		if (expected != result) {
			if (_invalid_count++ < max_invalid) {
				std::cerr << "Invalid value at C[" << _i << ", " << _j << "]: expected " << expected << ", got "
						  << result << std::endl;
			}
		}

		++_j;
		if (_j >= _nj) {
			_j = 0;
			++_i;
		}

		return *this;
	}

	bool is_valid() const { return _invalid_count == 0; }

private:
	std::istream *_file;

	std::size_t _invalid_count = 0;

	std::size_t _ni;
	std::size_t _nj;

	std::size_t _i = 0;
	std::size_t _j = 0;
};

#endif // NOARR_POLYBENCH_COMMON_HPP
