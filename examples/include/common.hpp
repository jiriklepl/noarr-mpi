#ifndef NOARR_POLYBENCH_COMMON_HPP
#define NOARR_POLYBENCH_COMMON_HPP

#include <cmath>

#include <format>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

class matrix_stream_check {
public:
	static constexpr std::size_t max_invalid = 10;
	static constexpr double epsilon = 2e-2;

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
			auto expected_value = std::stod(expected);
			auto result_value = std::stod(result);

			if (std::abs(expected_value - result_value) >= epsilon && _invalid_count < max_invalid) {
				std::cerr << "Invalid value at C[" << _i << ", " << _j << "]: expected " << expected << ", got "
						  << result << '\n';
				++_invalid_count;
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

inline std::pair<double, double> mean_stddev(const std::vector<double> &v) {
	if (v.empty()) {
		return {0.0, 0.0};
	}

	const double mean = std::accumulate(v.cbegin(), v.cend(), 0.0) / (double)v.size();
	const double stddev =
		std::sqrt(std::accumulate(v.cbegin(), v.cend(), 0.0,
	                              [mean](double acc, double val) { return acc + ((val - mean) * (val - mean)); }) /
	              (double)v.size());

	return {mean, stddev};
}

#endif // NOARR_POLYBENCH_COMMON_HPP
