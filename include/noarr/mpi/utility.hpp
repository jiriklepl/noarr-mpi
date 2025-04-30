#ifndef NOARR_MPI_UTILITY_HPP
#define NOARR_MPI_UTILITY_HPP

#include <cstdlib>

#include <array>
#include <complex>
#include <concepts>
#include <iostream>
#include <string_view>
#include <type_traits>
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

namespace noarr::mpi {

class MPI_session {
	MPI_Comm comm = MPI_COMM_WORLD;

public:
	explicit MPI_session(MPI_Comm comm = MPI_COMM_WORLD) : comm(comm) { MPICHK(MPI_Init(nullptr, nullptr)); }

	explicit MPI_session(int &argc, char **&argv, MPI_Comm comm = MPI_COMM_WORLD) : comm(comm) {
		MPICHK(MPI_Init(&argc, &argv));
	}

	~MPI_session() noexcept(false) { MPICHK(MPI_Finalize()); }

	MPI_session(const MPI_session &) = delete;
	MPI_session &operator=(const MPI_session &) = delete;

	MPI_session(MPI_session &&) = delete;
	MPI_session &operator=(MPI_session &&) = delete;

	[[nodiscard]]
	constexpr auto get_comm() const noexcept {
		return comm;
	}
};

template<class T>
struct to_MPI_Datatype : std::false_type {};

template<class T>
constexpr bool to_MPI_Datatype_v = to_MPI_Datatype<T>::value;

template<class T>
using to_MPI_Datatype_t = typename to_MPI_Datatype<T>::type;

template<class T>
concept ToMPIDatatype = to_MPI_Datatype_v<std::remove_cvref_t<T>>;

template<class T>
requires ToMPIDatatype<T>
constexpr decltype(auto) convert_to_MPI_Datatype(T &&t) noexcept {
	return to_MPI_Datatype<std::remove_cvref_t<T>>::convert(std::forward<T>(t));
}

template<>
struct to_MPI_Datatype<MPI_Datatype> : std::true_type {
	using type = MPI_Datatype;

	[[nodiscard]]
	static constexpr type convert(MPI_Datatype mpi_type) noexcept {
		return mpi_type;
	}
};

class MPI_custom_type {
	MPI_Datatype value;
	bool free_on_destruction = true;

public:
	constexpr MPI_custom_type() noexcept : value(MPI_DATATYPE_NULL) {}

	explicit MPI_custom_type(MPI_Datatype value, bool free_on_destruction = true) noexcept
		: value(value), free_on_destruction(free_on_destruction) {}

	MPI_custom_type(const MPI_custom_type &) = delete;
	MPI_custom_type &operator=(const MPI_custom_type &) = delete;

	// leaves the other in a valid but unspecified state
	MPI_custom_type(MPI_custom_type &&other) noexcept
		: value(other.value), free_on_destruction(other.free_on_destruction) {
		other.value = MPI_DATATYPE_NULL;
		other.free_on_destruction = false;
	}

	// leaves the other in a valid but unspecified state
	MPI_custom_type &operator=(MPI_custom_type &&other) noexcept {
		using std::swap;
		swap(value, other.value);

		return *this;
	}

	void commit() {
		if (value != MPI_DATATYPE_NULL) {
			MPICHK(MPI_Type_commit(&value));
		}
	}

	void reset(MPI_Datatype value = MPI_DATATYPE_NULL, bool free_on_destruction = true) {
		if (this->value != MPI_DATATYPE_NULL && free_on_destruction) {
			MPICHK(MPI_Type_free(&this->value));
		}

		this->value = value;
		this->free_on_destruction = value != MPI_DATATYPE_NULL && free_on_destruction;
	}

	~MPI_custom_type() noexcept(false) { reset(); }

	constexpr explicit operator MPI_Datatype() const { return value; }
};

template<>
struct to_MPI_Datatype<MPI_custom_type> : std::true_type {
	using type = MPI_Datatype;

	[[nodiscard]]
	static constexpr type convert(const MPI_custom_type &mpi_type) noexcept {
		return static_cast<MPI_Datatype>(mpi_type);
	}
};

template<class T>
struct to_MPI_Comm : std::false_type {};

template<class T>
constexpr bool to_MPI_Comm_v = to_MPI_Comm<T>::value;

template<class T>
using to_MPI_Comm_t = typename to_MPI_Comm<T>::type;

template<class T>
concept ToMPIComm = to_MPI_Comm_v<std::remove_cvref_t<T>>;

template<class T>
requires ToMPIComm<T>
constexpr decltype(auto) convert_to_MPI_Comm(T &&t) noexcept {
	return to_MPI_Comm<std::remove_cvref_t<T>>::convert(std::forward<T>(t));
}

template<>
struct to_MPI_Comm<MPI_Comm> : std::true_type {
	using type = MPI_Comm;

	[[nodiscard]]
	static constexpr type convert(MPI_Comm comm) noexcept {
		return comm;
	}
};

template<>
struct to_MPI_Comm<MPI_session> : std::true_type {
	using type = MPI_Comm;

	[[nodiscard]]
	static constexpr type convert(const MPI_session &session) noexcept {
		return session.get_comm();
	}
};

class mpi_comm_guard {
	MPI_Comm value;

public:
	explicit mpi_comm_guard(MPI_Comm value) : value(value) {}

	mpi_comm_guard(const mpi_comm_guard &) = delete;
	mpi_comm_guard &operator=(const mpi_comm_guard &) = delete;

	mpi_comm_guard(mpi_comm_guard &&) = delete;
	mpi_comm_guard &operator=(mpi_comm_guard &&) = delete;

	~mpi_comm_guard() noexcept(false) { MPICHK(MPI_Comm_free(&value)); }

	constexpr explicit operator MPI_Comm() const { return value; }
};

template<>
struct to_MPI_Comm<mpi_comm_guard> : std::true_type {
	using type = MPI_Comm;

	[[nodiscard]]
	static constexpr type convert(const mpi_comm_guard &comm) noexcept {
		return static_cast<MPI_Comm>(comm);
	}
};

template<class T>
struct choose_mpi_type : std::false_type {};

template<class T>
constexpr MPI_Datatype choose_mpi_type_v() noexcept
requires choose_mpi_type<T>::value
{
	return choose_mpi_type<std::remove_cvref_t<T>>::get();
}

template<>
struct choose_mpi_type<char> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_CHAR; }
};

template<>
struct choose_mpi_type<signed char> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_SIGNED_CHAR; }
};

template<>
struct choose_mpi_type<unsigned char> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_UNSIGNED_CHAR; }
};

template<>
struct choose_mpi_type<wchar_t> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_WCHAR; }
};

template<>
struct choose_mpi_type<short> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_SHORT; }
};

template<>
struct choose_mpi_type<unsigned short> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_UNSIGNED_SHORT; }
};

template<>
struct choose_mpi_type<int> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_INT; }
};

template<>
struct choose_mpi_type<unsigned int> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_UNSIGNED; }
};

template<>
struct choose_mpi_type<long> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_LONG; }
};

template<>
struct choose_mpi_type<unsigned long> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_UNSIGNED_LONG; }
};

template<>
struct choose_mpi_type<long long> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_LONG_LONG; }
};

template<>
struct choose_mpi_type<unsigned long long> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_UNSIGNED_LONG_LONG; }
};

template<>
struct choose_mpi_type<float> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_FLOAT; }
};

template<>
struct choose_mpi_type<double> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_DOUBLE; }
};

template<>
struct choose_mpi_type<long double> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_LONG_DOUBLE; }
};

template<class T>
requires std::same_as<T, int8_t>
struct choose_mpi_type<T> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_INT8_T; }
};

template<class T>
requires std::same_as<T, int16_t>
struct choose_mpi_type<T> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_INT16_T; }
};

template<class T>
requires std::same_as<T, int32_t>
struct choose_mpi_type<T> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_INT32_T; }
};

template<class T>
requires std::same_as<T, int64_t>
struct choose_mpi_type<T> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_INT64_T; }
};

template<class T>
requires std::same_as<T, uint8_t>
struct choose_mpi_type<T> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_UINT8_T; }
};

template<class T>
requires std::same_as<T, uint16_t>
struct choose_mpi_type<T> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_UINT16_T; }
};

template<class T>
requires std::same_as<T, uint32_t>
struct choose_mpi_type<T> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_UINT32_T; }
};

template<class T>
requires std::same_as<T, uint64_t>
struct choose_mpi_type<T> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_UINT64_T; }
};

template<>
struct choose_mpi_type<bool> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_C_BOOL; }
};

template<>
struct choose_mpi_type<std::complex<float>> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_COMPLEX; }
};

template<>
struct choose_mpi_type<std::complex<double>> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_DOUBLE_COMPLEX; }
};

template<>
struct choose_mpi_type<std::complex<long double>> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_C_LONG_DOUBLE_COMPLEX; }
};

template<>
struct choose_mpi_type<std::pair<long, int>> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_LONG_INT; }
};

template<>
struct choose_mpi_type<std::pair<short, int>> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_SHORT_INT; }
};

template<>
struct choose_mpi_type<std::pair<int, int>> : std::true_type {
	static constexpr MPI_Datatype get() noexcept { return MPI_2INT; }
};

inline std::pair<MPI_Aint, MPI_Aint> mpi_type_get_extent(const ToMPIDatatype auto &has_type) {
	std::pair<MPI_Aint, MPI_Aint> result;

	const auto type = convert_to_MPI_Datatype(has_type);

	auto &[lb, extent] = result;
	MPICHK(MPI_Type_get_extent(type, &lb, &extent));

	return result;
}

} // namespace noarr::mpi

#endif // NOARR_MPI_UTILITY_HPP
