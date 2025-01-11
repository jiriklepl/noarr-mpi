#ifndef NOARR_STRUCTURES_INTEROP_MPI_UTILITY_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_UTILITY_HPP

#include <cstdlib>

#include <array>
#include <complex>
#include <iostream>
#include <string_view>
#include <utility>

#include <noarr/structures/base/utility.hpp>

#include <mpi.h>

#define MPICHK(...)                                                                                                    \
	do {                                                                                                               \
		const decltype(MPI_SUCCESS) mpi_result = (__VA_ARGS__);                                                        \
		if (mpi_result != MPI_SUCCESS) {                                                                               \
			std::array<char, MPI_MAX_ERROR_STRING> error_string{};                                                     \
			int error_string_length;                                                                                   \
			MPI_Error_string(mpi_result, error_string.data(), &error_string_length);                                   \
			std::cerr << __FILE__ << ':' << __LINE__                                                                   \
					  << ": MPI error: " << std::string_view(error_string.data(), error_string_length) << '\n';        \
			std::cerr << "\t in " << __func__ << ": " #__VA_ARGS__ << '\n';                                            \
			std::exit(mpi_result);                                                                                     \
		}                                                                                                              \
	} while (0)

namespace noarr {

class MPI_custom_type {
	MPI_Datatype value;

public:
	constexpr MPI_custom_type() noexcept : value(MPI_DATATYPE_NULL) {}

	explicit MPI_custom_type(MPI_Datatype value) : value(value) {
		MPICHK(MPI_Type_commit(&value));
		std::cerr << "MPI_Type_commit(" << value << ")" << '\n';
	}

	MPI_custom_type(const MPI_custom_type &) = delete;
	auto operator=(const MPI_custom_type &) -> MPI_custom_type & = delete;

	// leaves the other in a valid but unspecified state
	MPI_custom_type(MPI_custom_type &&other) noexcept : value(other.value) { other.value = MPI_DATATYPE_NULL; }

	// leaves the other in a valid but unspecified state
	auto operator=(MPI_custom_type &&other) noexcept -> MPI_custom_type & {
		std::swap(value, other.value);

		return *this;
	}

	void reset(MPI_Datatype value) {
		if (this->value != MPI_DATATYPE_NULL) {
			MPICHK(MPI_Type_free(&this->value));
		}

		if (value != MPI_DATATYPE_NULL) {
			MPICHK(MPI_Type_commit(&value));
		}

		this->value = value;
	}

	~MPI_custom_type() { reset(MPI_DATATYPE_NULL); }

	explicit operator MPI_Datatype() const { return value; }
};

class MPI_session {
public:
	MPI_session() { MPICHK(MPI_Init(nullptr, nullptr)); }

	~MPI_session() { MPICHK(MPI_Finalize()); }

	MPI_session(const MPI_session &) = delete;
	auto operator=(const MPI_session &) -> MPI_session & = delete;

	MPI_session(MPI_session &&) = delete;
	auto operator=(MPI_session &&) -> MPI_session & = delete;
};

template<class T>
struct choose_mpi_type {
	static_assert(always_false<T>, "Unsupported type");
};

template<class T>
constexpr auto choose_mpi_type_v() -> MPI_Datatype {
	return choose_mpi_type<T>::value();
}

template<>
struct choose_mpi_type<char> {
	static constexpr auto value() -> MPI_Datatype { return MPI_CHAR; }
};

template<>
struct choose_mpi_type<signed char> {
	static constexpr auto value() -> MPI_Datatype { return MPI_SIGNED_CHAR; }
};

template<>
struct choose_mpi_type<unsigned char> {
	static constexpr auto value() -> MPI_Datatype { return MPI_UNSIGNED_CHAR; }
};

template<>
struct choose_mpi_type<wchar_t> {
	static constexpr auto value() -> MPI_Datatype { return MPI_WCHAR; }
};

template<>
struct choose_mpi_type<short> {
	static constexpr auto value() -> MPI_Datatype { return MPI_SHORT; }
};

template<>
struct choose_mpi_type<unsigned short> {
	static constexpr auto value() -> MPI_Datatype { return MPI_UNSIGNED_SHORT; }
};

template<>
struct choose_mpi_type<int> {
	static constexpr auto value() -> MPI_Datatype { return MPI_INT; }
};

template<>
struct choose_mpi_type<unsigned int> {
	static constexpr auto value() -> MPI_Datatype { return MPI_UNSIGNED; }
};

template<>
struct choose_mpi_type<long> {
	static constexpr auto value() -> MPI_Datatype { return MPI_LONG; }
};

template<>
struct choose_mpi_type<unsigned long> {
	static constexpr auto value() -> MPI_Datatype { return MPI_UNSIGNED_LONG; }
};

template<>
struct choose_mpi_type<long long> {
	static constexpr auto value() -> MPI_Datatype { return MPI_LONG_LONG; }
};

template<>
struct choose_mpi_type<unsigned long long> {
	static constexpr auto value() -> MPI_Datatype { return MPI_UNSIGNED_LONG_LONG; }
};

template<>
struct choose_mpi_type<float> {
	static constexpr auto value() -> MPI_Datatype { return MPI_FLOAT; }
};

template<>
struct choose_mpi_type<double> {
	static constexpr auto value() -> MPI_Datatype { return MPI_DOUBLE; }
};

template<>
struct choose_mpi_type<long double> {
	static constexpr auto value() -> MPI_Datatype { return MPI_LONG_DOUBLE; }
};

template<class T>
requires std::same_as<T, int8_t>
struct choose_mpi_type<T> {
	static constexpr auto value() -> MPI_Datatype { return MPI_INT8_T; }
};

template<class T>
requires std::same_as<T, int16_t>
struct choose_mpi_type<T> {
	static constexpr auto value() -> MPI_Datatype { return MPI_INT16_T; }
};

template<class T>
requires std::same_as<T, int32_t>
struct choose_mpi_type<T> {
	static constexpr auto value() -> MPI_Datatype { return MPI_INT32_T; }
};

template<class T>
requires std::same_as<T, int64_t>
struct choose_mpi_type<T> {
	static constexpr auto value() -> MPI_Datatype { return MPI_INT64_T; }
};

template<class T>
requires std::same_as<T, uint8_t>
struct choose_mpi_type<T> {
	static constexpr auto value() -> MPI_Datatype { return MPI_UINT8_T; }
};

template<class T>
requires std::same_as<T, uint16_t>
struct choose_mpi_type<T> {
	static constexpr auto value() -> MPI_Datatype { return MPI_UINT16_T; }
};

template<class T>
requires std::same_as<T, uint32_t>
struct choose_mpi_type<T> {
	static constexpr auto value() -> MPI_Datatype { return MPI_UINT32_T; }
};

template<class T>
requires std::same_as<T, uint64_t>
struct choose_mpi_type<T> {
	static constexpr auto value() -> MPI_Datatype { return MPI_UINT64_T; }
};

// TODO: not sure about the following types

template<>
struct choose_mpi_type<bool> {
	static constexpr auto value() -> MPI_Datatype { return MPI_C_BOOL; }
};

template<>
struct choose_mpi_type<std::complex<float>> {
	static constexpr auto value() -> MPI_Datatype { return MPI_COMPLEX; }
};

template<>
struct choose_mpi_type<std::complex<double>> {
	static constexpr auto value() -> MPI_Datatype { return MPI_DOUBLE_COMPLEX; }
};

template<>
struct choose_mpi_type<std::complex<long double>> {
	static constexpr auto value() -> MPI_Datatype { return MPI_C_LONG_DOUBLE_COMPLEX; }
};

template<>
struct choose_mpi_type<std::pair<long, int>> {
	static constexpr auto value() -> MPI_Datatype {
		using pair_t = struct {
			long first;
			int second;
		};

		static_assert(sizeof(std::pair<long, int>) == sizeof(pair_t));
		static_assert(alignof(std::pair<long, int>) == alignof(pair_t));
		static_assert(std::is_standard_layout_v<std::pair<long, int>>);
		return MPI_LONG_INT;
	}
};

template<>
struct choose_mpi_type<std::pair<short, int>> {
	static constexpr auto value() -> MPI_Datatype {
		using pair_t = struct {
			short first;
			int second;
		};

		static_assert(sizeof(std::pair<short, int>) == sizeof(pair_t));
		static_assert(alignof(std::pair<short, int>) == alignof(pair_t));
		static_assert(std::is_standard_layout_v<std::pair<short, int>>);
		return MPI_SHORT_INT;
	}
};

template<>
struct choose_mpi_type<std::pair<int, int>> {
	static constexpr auto value() -> MPI_Datatype {
		using pair_t = struct {
			int first;
			int second;
		};

		static_assert(sizeof(std::pair<int, int>) == sizeof(pair_t));
		static_assert(alignof(std::pair<int, int>) == alignof(pair_t));
		static_assert(std::is_standard_layout_v<std::pair<int, int>>);
		return MPI_2INT;
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_UTILITY_HPP
