#ifndef NOARR_POLYBENCH_COMMON_HPP
#define NOARR_POLYBENCH_COMMON_HPP

#include <iomanip>
#include <iostream>
#include <sstream>

class stream_check {
public:
	explicit stream_check(std::istream &f) : file(&f) { ss << std::fixed << std::setprecision(2); }

	stream_check &operator<<(char /* v */) {
		// ignore
		return *this;
	}

	template<typename T>
	stream_check &operator<<(T v) {
		ss.str("");
		ss.clear();

		ss << v;

		std::string compare_with;
		(*file) >> compare_with;
		valid &= (compare_with == ss.str());
		return *this;
	}

	bool is_valid() const { return valid; }

private:
	std::ostringstream ss;
	std::istream *file;
	bool valid = true;
};

#endif // NOARR_POLYBENCH_COMMON_HPP
