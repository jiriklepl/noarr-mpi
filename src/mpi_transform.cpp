#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <expected>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <variant>
#include <vector>

#include <mpi.h>

#include <noarr/structures/extra/shortcuts.hpp>
#include <noarr/structures/extra/struct_concepts.hpp>
#include <noarr/traversers.hpp>

#include "noarr/structures/base/structs_common.hpp"
#include "noarr/structures/base/utility.hpp"
#include "noarr/tokenizer.hpp"

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
	[[maybe_unused]]
	MPI_session() {
		MPICHK(MPI_Init(nullptr, nullptr));
	}

	~MPI_session() { MPICHK(MPI_Finalize()); }

	MPI_session(const MPI_session &) = delete;
	auto operator=(const MPI_session &) -> MPI_session & = delete;

	MPI_session(MPI_session &&) = delete;
	auto operator=(MPI_session &&) -> MPI_session & = delete;
};

template<class T>
struct choose_mpi_type {
	static_assert(noarr::always_false<T>, "Unsupported type");
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

// TODO: review the types and implement the missing ones

// ---------------------------------------------------------------------------

class erasure {
	class abstract_type {
	public:
		constexpr abstract_type() = default;
		virtual ~abstract_type() = default;

		abstract_type(const abstract_type &) = delete;
		auto operator=(const abstract_type &) -> abstract_type & = delete;

		abstract_type(abstract_type &&) = delete;
		auto operator=(abstract_type &&) -> abstract_type & = delete;
	};

	// singleton
	template<class T>
	class concrete_type : public abstract_type {
		concrete_type() = default;

	public:
		[[nodiscard]]
		static auto get() -> abstract_type * {
			if (m_instance == nullptr) {
				m_instance = std::unique_ptr<concrete_type>(new concrete_type());
			}

			return m_instance.get();
		}

	private:
		thread_local static std::unique_ptr<abstract_type> m_instance;
	};

	explicit erasure(abstract_type *dim) : m_dim(dim) {}

public:
	template<class T>
	explicit erasure(T /*unused*/) : m_dim(concrete_type<T>::get()) {}

	template<class T>
	static auto get() -> erasure;

	friend constexpr auto operator<=>(const erasure &lhs, const erasure &rhs) = default;

private:
	abstract_type *m_dim;
};

template<class T>
auto erasure::get() -> erasure {
	return erasure(concrete_type<T>::get());
}

template<class T>
thread_local std::unique_ptr<erasure::abstract_type> erasure::concrete_type<T>::m_instance;

template<class T>
class translate {};

template<class T>
class translate<noarr::scalar<T>> {
public:
	static auto get() -> erasure { return erasure::get<noarr::scalar<T>>(); }
};

template<auto Dim, class T>
class translate<noarr::vector_t<Dim, T>> {
public:
	static auto get() -> erasure { return erasure::get<noarr::dim<Dim>>(); }
};

template<auto Dim, class T, class LenT>
class translate<noarr::set_length_t<Dim, T, LenT>> {
public:
	static auto get() -> erasure { return erasure::get<noarr::dim<Dim>>(); }
};

template<auto OldDim, auto DimMajor, auto DimMinor, class T>
class translate<noarr::into_blocks_t<OldDim, DimMajor, DimMinor, T>> {
public:
	static auto get() -> erasure {
		// get the major dimension, as it is the outermost one
		return erasure::get<noarr::dim<DimMajor>>();
	}
};

template<auto Dim, class T>
class translate<noarr::hoist_t<Dim, T>> {
public:
	static auto get() -> erasure { return erasure::get<noarr::dim<Dim>>(); }
};

template<auto Dim, class LenT>
class translate<noarr::set_length_proto<Dim, LenT>> {
public:
	static auto get() -> erasure { return erasure::get<noarr::dim<Dim>>(); }
};

template<auto Dim, class T, class LenT>
class translate<noarr::shift_t<Dim, T, LenT>> {
public:
	static auto get() -> erasure { return erasure::get<noarr::dim<Dim>>(); }
};

template<auto Dim, class T, class StartT, class LenT>
class translate<noarr::slice_t<Dim, T, StartT, LenT>> {
public:
	static auto get() -> erasure { return erasure::get<noarr::dim<Dim>>(); }
};

template<auto Dim, class T, class StartT, class EndT>
class translate<noarr::span_t<Dim, T, StartT, EndT>> {
public:
	static auto get() -> erasure { return erasure::get<noarr::dim<Dim>>(); }
};

template<auto Dim, class T>
class translate<noarr::bcast_t<Dim, T>> {
public:
	static auto get() -> erasure { return erasure::get<noarr::dim<Dim>>(); }
};

// TODO: add support for step_t
// TODO: and reverse_t... omg
// TODO: a support for general rename is kinda annoying (imagine rename('x' -> 'y', 'y' -> 'x'); and it can have an
// arbitrary number of renames that happen 'simultaneously')

struct dimension_data;

class size_expression {
public:
	using ptr = std::unique_ptr<size_expression>;

	constexpr size_expression() = default;
	virtual ~size_expression() = default;

	size_expression(const size_expression &) = delete;
	auto operator=(const size_expression &) -> size_expression & = delete;

	size_expression(size_expression &&) = default;
	auto operator=(size_expression &&) -> size_expression & = default;

	[[nodiscard]]
	virtual auto get(std::map<erasure, dimension_data> &dimensions) const -> std::size_t = 0;

	[[nodiscard]]
	virtual auto clone() const -> ptr = 0;
};

struct dimension_data {
	enum class param : std::uint8_t {
		extent,
		start,
		end,
	};

	size_expression::ptr start;  // the first element
	size_expression::ptr end;    // the last element + 1
	size_expression::ptr extent; // real size
	size_expression::ptr stride; // step
	std::optional<erasure> parent;
	std::set<erasure> children;
	std::variant<std::monostate, MPI_Datatype, MPI_custom_type> type;
};

class constant_size_expression : public size_expression {
public:
	explicit constant_size_expression(std::size_t value) : m_value(value) {}

	auto get(std::map<erasure, dimension_data> & /*dimensions*/) const -> std::size_t override { return m_value; }

	[[nodiscard]]
	auto clone() const -> size_expression::ptr override {
		return std::make_unique<constant_size_expression>(m_value);
	}

private:
	std::size_t m_value;
};

auto make_size_expression(std::size_t value) -> size_expression::ptr {
	return std::make_unique<constant_size_expression>(value);
}

class unknown_size_expression : public size_expression {
public:
	[[nodiscard]]
	auto get(std::map<erasure, dimension_data> & /*unused*/) const -> std::size_t override {
		throw std::runtime_error("Unknown size expression");
	}

	[[nodiscard]]
	auto clone() const -> size_expression::ptr override {
		return std::make_unique<unknown_size_expression>();
	}
};

auto make_size_expression() -> size_expression::ptr { return std::make_unique<unknown_size_expression>(); }

template<dimension_data::param Param>
class param_size_expression : public size_expression {
	using param = dimension_data::param;

public:
	explicit param_size_expression(erasure e) : m_erasure(e) {}

	[[nodiscard]]
	auto get(std::map<erasure, dimension_data> &dimensions) const -> std::size_t override {
		const auto it = dimensions.find(m_erasure);
		if (it == dimensions.end()) {
			throw std::runtime_error("Dimension not found");
		}

		const auto &data = it->second;

		switch (Param) {
		case param::extent:
			return data.extent->get(dimensions);
		case param::start:
			return data.start->get(dimensions);
		case param::end:
			return data.end->get(dimensions);
		}

		// should be unreachable
		assert(((void)"Invalid param", false));
		return 0;
	}

	[[nodiscard]]
	auto clone() const -> size_expression::ptr override {
		return std::make_unique<param_size_expression>(m_erasure);
	}

private:
	erasure m_erasure;
};

template<dimension_data::param p>
auto make_size_expression(erasure e) -> size_expression::ptr {
	return std::make_unique<param_size_expression<p>>(e);
}

template<dimension_data::param p>
auto make_size_expression(dimension_data &data) -> size_expression::ptr {
	using param = dimension_data::param;

	switch (p) {
	case param::extent:
		return data.extent->clone();
	case param::start:
		return data.start->clone();
	case param::end:
		return data.end->clone();
	}

	// should be unreachable
	assert(((void)"Invalid param", false));
	return nullptr;
}

template<class Op>
requires std::is_invocable_v<Op, std::size_t, std::size_t>
class binary_size_expression : public size_expression {
public:
	binary_size_expression(size_expression::ptr left, size_expression::ptr right)
		: m_left(std::move(left)), m_right(std::move(right)), m_op() {}

	[[nodiscard]]
	auto get(std::map<erasure, dimension_data> &dimensions) const -> std::size_t override {
		return m_op(m_left->get(dimensions), m_right->get(dimensions));
	}

	[[nodiscard]]
	auto clone() const -> size_expression::ptr override {
		return std::make_unique<binary_size_expression>(m_left->clone(), m_right->clone());
	}

private:
	size_expression::ptr m_left;
	size_expression::ptr m_right;
	Op m_op;
};

template<class Op>
requires std::is_invocable_v<Op, std::size_t, std::size_t>
auto make_size_expression(size_expression::ptr left, size_expression::ptr right) -> size_expression::ptr {
	return std::make_unique<binary_size_expression<Op>>(std::move(left), std::move(right));
}

struct range_size_t {};

constexpr range_size_t range_size;

auto make_size_expression(range_size_t /*unused*/, erasure e) -> size_expression::ptr {
	using param = dimension_data::param;
	return make_size_expression<param::extent>(e);
}

auto make_size_expression(range_size_t /*unused*/, dimension_data &e) -> size_expression::ptr {
	return e.extent->clone(); // TODO: check whether clone is a good idea
}

class mpi_transform_builder {
public:
	template<class T>
	void operator()(noarr::scalar<T> /*unused*/) {
		const auto e = erasure::get<noarr::scalar<T>>();
		MPI_Datatype mpi_type = choose_mpi_type_v<T>();

		m_dimensions.try_emplace(e, dimension_data{
										.start = make_size_expression(0),
										.end = make_size_expression(1), // TODO: reassess this
										.extent = make_size_expression(1),
										.stride = make_size_expression(0), // TODO: can be basically anything
										.parent = std::nullopt,
										.children = {},
										.type = mpi_type,
									});
	}

	template<auto Dim, class T>
	requires noarr::IsDim<decltype(Dim)>
	void operator()(noarr::vector_t<Dim, T> /*unused*/) {
		const auto e = erasure::get<noarr::dim<Dim>>();
		const auto parent = translate<T>::get();

		const auto result = m_dimensions.try_emplace(e, dimension_data{
															.start = make_size_expression(),
															.end = make_size_expression(),
															.extent = make_size_expression(),
															.stride = make_size_expression(1),
															.parent = parent,
															.children = {},
															.type = {},
														});

		if (!result.second) {
			throw std::runtime_error("Dimension already exists");
		}

		if (auto it = m_dimensions.find(parent); it != m_dimensions.end()) {
			it->second.children.emplace(e);
		} else {
			throw std::runtime_error("Parent not found");
		}
	}

	template<auto Dim, auto DimMajor, auto DimMinor, class T>
	requires noarr::IsDim<decltype(Dim)> && noarr::IsDim<decltype(DimMajor)> && noarr::IsDim<decltype(DimMinor)>
	void operator()(noarr::into_blocks_t<Dim, DimMajor, DimMinor, T> /*unused*/) {
		// take the dimension at Dim
		const auto old = erasure::get<noarr::dim<Dim>>();
		const auto old_it = m_dimensions.find(old);
		if (old_it == m_dimensions.end()) {
			throw std::runtime_error("Dimension not found");
		}

		// kill the old dimension
		m_graveyard.emplace_back(std::move(*old_it));
		m_dimensions.erase(old_it);

		auto &moved_data = m_graveyard.back().second;

		moved_data.type = MPI_DATATYPE_NULL;

		const auto e_minor = erasure::get<noarr::dim<DimMinor>>();
		const auto e_major = erasure::get<noarr::dim<DimMajor>>();

		// create new dimensions
		m_dimensions.try_emplace(
			e_major, dimension_data{
						 .start = make_size_expression(0),
						 .end = make_size_expression<std::divides<>>(make_size_expression(range_size, moved_data),
		                                                             make_size_expression(range_size, e_minor)),
						 .extent = make_size_expression<std::divides<>>(make_size_expression(range_size, moved_data),
		                                                                make_size_expression(range_size, e_minor)),
						 .stride = make_size_expression(1),
						 .parent = e_minor,
						 .type = {},
					 });

		m_dimensions.try_emplace(e_minor,
		                         dimension_data{
									 .start = make_size_expression(),
									 .end = make_size_expression(),
									 .extent = make_size_expression(),
									 .stride = moved_data.stride->clone(), // TODO: Check whether clone is a good idea
									 .parent = moved_data.parent,
									 .children = {e_major},
									 .type = {},
								 });

		for (const auto &e : moved_data.children) {
			if (auto it = m_dimensions.find(e); it != m_dimensions.end()) {
				it->second.parent = e_major;
			} else {
				throw std::runtime_error("Child not found");
			}
		}

		moved_data.children.clear();

		if (moved_data.parent.has_value()) {
			if (auto it = m_dimensions.find(moved_data.parent.value()); it != m_dimensions.end()) {
				if (old != e_minor) {
					it->second.children.erase(old);
					it->second.children.emplace(e_minor);
				}
			} else {
				throw std::runtime_error("Parent not found");
			}
		}

		moved_data.parent = std::nullopt;
	}

	template<auto Dim, class T>
	requires noarr::IsDim<decltype(Dim)>
	void operator()(noarr::hoist_t<Dim, T> /*unused*/) {
		/* do nothing */
	}

	template<auto Dim, class T, class LenT>
	requires noarr::IsDim<decltype(Dim)>
	void operator()(noarr::set_length_t<Dim, T, LenT> sl) {
		const auto e = erasure::get<noarr::dim<Dim>>();

		const auto it = m_dimensions.find(e);
		if (it == m_dimensions.end()) {
			throw std::runtime_error("Dimension not found");
		}

		if (!dynamic_cast<unknown_size_expression *>(it->second.start.get())) {
			throw std::runtime_error("Start already set");
		}

		if (!dynamic_cast<unknown_size_expression *>(it->second.extent.get())) {
			throw std::runtime_error("Extent already set");
		}

		if (!dynamic_cast<unknown_size_expression *>(it->second.end.get())) {
			throw std::runtime_error("End already set");
		}

		it->second.start = make_size_expression(0);
		it->second.extent = make_size_expression(sl.len());
		it->second.end = make_size_expression(sl.len());
	}

	template<auto Dim, class T, class LenT>
	requires noarr::IsDim<decltype(Dim)>
	[[deprecated("Not implemented")]]
	void operator()(noarr::shift_t<Dim, T, LenT> shift) {
		const auto e = erasure::get<noarr::dim<Dim>>();
		const auto it = m_dimensions.find(e);

		if (it == m_dimensions.end()) {
			throw std::runtime_error("Dimension not found");
		}

		if (dynamic_cast<unknown_size_expression *>(it->second.start.get())) {
			// TODO: in noarr (or mu), it is technically possible to first shift the dimension and then set the length
			throw std::runtime_error("Start not set");
		}

		it->second.start =
			make_size_expression<std::plus<>>(std::move(it->second.start), make_size_expression(shift.start()));
	}

	template<auto Dim, class T, class StartT, class LenT>
	requires noarr::IsDim<decltype(Dim)>
	void operator()(noarr::slice_t<Dim, T, StartT, LenT> slice) {
		const auto e = erasure::get<noarr::dim<Dim>>();
		const auto it = m_dimensions.find(e);

		if (it == m_dimensions.end()) {
			throw std::runtime_error("Dimension not found");
		}

		if (dynamic_cast<unknown_size_expression *>(it->second.start.get())) {
			throw std::runtime_error("Start not set");
		}

		if (dynamic_cast<unknown_size_expression *>(it->second.end.get())) {
			throw std::runtime_error("End not set");
		}

		it->second.start =
			make_size_expression<std::plus<>>(std::move(it->second.start), make_size_expression(slice.start()));

		it->second.end = make_size_expression(slice.start() + slice.len());
	}

	template<auto Dim, class T, class StartT, class EndT>
	requires noarr::IsDim<decltype(Dim)>
	void operator()(noarr::span_t<Dim, T, StartT, EndT> span) {
		const auto e = erasure::get<noarr::dim<Dim>>();
		const auto it = m_dimensions.find(e);

		if (it == m_dimensions.end()) {
			throw std::runtime_error("Dimension not found");
		}

		if (dynamic_cast<unknown_size_expression *>(it->second.start.get())) {
			throw std::runtime_error("Start not set");
		}

		if (dynamic_cast<unknown_size_expression *>(it->second.end.get())) {
			throw std::runtime_error("End not set");
		}

		it->second.start =
			make_size_expression<std::plus<>>(std::move(it->second.start), make_size_expression(span.start()));

		it->second.end = make_size_expression(span.end());
	}

	template<auto Dim, class T>
	void operator()(noarr::bcast_t<Dim, T> /*unused*/) {
		const auto e = erasure::get<noarr::dim<Dim>>();
		const auto parent = translate<T>::get();

		const auto result = m_dimensions.try_emplace(e, dimension_data{
															.start = make_size_expression(),
															.end = make_size_expression(),
															.extent = make_size_expression(),
															.stride = make_size_expression(0),
															.parent = parent,
															.children = {},
															.type = {},
														});

		if (!result.second) {
			throw std::runtime_error("Dimension already exists");
		}

		if (auto it = m_dimensions.find(parent); it != m_dimensions.end()) {
			it->second.children.emplace(e);
		} else {
			throw std::runtime_error("Parent not found");
		}
	}

	template<class T>
	[[deprecated("Unsupported type")]]
	void operator()(T /*unused*/) {
// check for gnu extensions (PRETTY_FUNCTION)
#if defined(__GNUC__) || defined(__clang__)
		throw std::runtime_error(__FILE__ ":" + std::to_string(__LINE__) +
		                         ": Error: unsupported type: " + __PRETTY_FUNCTION__);
#else
		throw std::runtime_error(__FILE__ ":" + std::to_string(__LINE__) + ": Error: unsupported type");
#endif
	}

	void operator()(noarr::IsProtoStruct auto /*unused*/) {
		throw std::runtime_error("A proto structure is not allowed here; we transform only full structures");
	}

	auto finalize() -> MPI_custom_type {
		std::vector<decltype(m_dimensions)::pointer> stack;
		std::set<decltype(m_dimensions)::pointer> visited;

		std::optional<erasure> root;

		std::ranges::transform(m_dimensions, std::back_inserter(stack), [](auto &item) { return &item; });
		std::ranges::transform(m_graveyard, std::back_inserter(stack), [](auto &item) { return &item; });

		while (!stack.empty()) {
			auto *const item = stack.back();
			auto &[e, data] = *item;
			stack.pop_back();

			const auto it = visited.find(item);
			if (it != visited.end()) {
				continue;
			}

			if (data.parent.has_value()) {
				const auto parent = data.parent.value();
				if (auto it = m_dimensions.find(parent); it != m_dimensions.end()) {
					const auto parent_it = visited.find(&*it);
					if (parent_it == visited.end()) {
						stack.push_back(item);
						stack.push_back(&*it);
						continue;
					}
				}

				const auto &parent_data = m_dimensions.at(parent);

				if (std::holds_alternative<MPI_Datatype>(data.type)) {
					continue;
				}

				if (std::holds_alternative<MPI_custom_type>(data.type)) {
					throw std::runtime_error("Type already set");
				}

				MPI_Datatype parent_mpi_type = MPI_DATATYPE_NULL;

				if (std::holds_alternative<MPI_Datatype>(parent_data.type)) {
					parent_mpi_type = std::get<MPI_Datatype>(parent_data.type);
				} else if (std::holds_alternative<MPI_custom_type>(parent_data.type)) {
					parent_mpi_type = MPI_Datatype(std::get<MPI_custom_type>(parent_data.type));
				} else {
					throw std::runtime_error("Parent type not set");
				}

				const auto start = data.start->get(m_dimensions);
				const auto end = data.end->get(m_dimensions);
				const auto extent = data.extent->get(m_dimensions);
				const auto stride = data.stride->get(m_dimensions);

				assert(start <= end && end <= extent);

				if (start == 0 && end == extent && stride == 1) {
					MPI_Datatype new_type = MPI_DATATYPE_NULL;
					MPICHK(MPI_Type_contiguous(end, parent_mpi_type, &new_type));
					std::cerr << "MPI_Type_contiguous(" << extent << ", " << parent_mpi_type << ", " << new_type << ")"
							  << '\n';
					data.type = MPI_custom_type(new_type);
				} else if (stride == 1) {
					MPI_Datatype new_type = MPI_DATATYPE_NULL;
					const auto displacements = static_cast<int>(start);
					MPICHK(MPI_Type_create_indexed_block(1, end - start, &displacements, parent_mpi_type, &new_type));
					std::cerr << "MPI_Type_create_indexed_block(1, " << end - start << ", {" << displacements << "}, "
							  << parent_mpi_type << ", " << new_type << ")" << '\n';
					m_graveyard.emplace_back(nullptr, dimension_data{
														  .start = {},
														  .end = {},
														  .extent = {},
														  .stride = {},
														  .parent = {},
														  .children = {},
														  .type = MPI_custom_type(new_type),
													  }); // TODO: we store a lot of data on top of the MPI_Datatype

					MPI_Datatype padded_type = MPI_DATATYPE_NULL;
					MPI_Aint old_lb = 0;
					MPI_Aint old_extent = 0;

					MPICHK(MPI_Type_get_extent(parent_mpi_type, &old_lb, &old_extent));

					MPICHK(MPI_Type_create_resized(new_type, 0, old_extent * extent, &padded_type));
					std::cerr << "MPI_Type_create_resized(" << new_type << ", 0, " << old_extent * extent << ", "
							  << padded_type << ")" << '\n';

					data.type = MPI_custom_type(padded_type);
				} else if (stride == 0) {
					// it is essentially a union/broadcast; it doesn't do anything
					MPI_Datatype new_type = MPI_DATATYPE_NULL;

					MPICHK(MPI_Type_dup(parent_mpi_type, &new_type));
					std::cerr << "MPI_Type_dup(" << parent_mpi_type << ", " << new_type << ")" << '\n';

					data.type = MPI_custom_type(new_type);
				} else {
					throw std::runtime_error("Unsupported transformation");
				}
			}

			visited.insert(item);
			root = e;
		}

		if (!root.has_value()) {
			return {};
		}

		// cleanup
		m_graveyard.clear();

		auto item = std::move(*m_dimensions.find(root.value()));

		m_dimensions.clear();
		m_dimensions.emplace(std::move(item));

		// return the root type
		auto &&root_type = m_dimensions.begin()->second.type;
		if (std::holds_alternative<MPI_Datatype>(root_type)) {
			MPI_Datatype new_type = MPI_DATATYPE_NULL;
			MPICHK(MPI_Type_dup(std::get<MPI_Datatype>(root_type), &new_type));
			std::cerr << "MPI_Type_dup(" << std::get<MPI_Datatype>(root_type) << ", " << new_type << ")" << '\n';

			return MPI_custom_type(new_type);
		}

		if (std::holds_alternative<MPI_custom_type>(root_type)) {
			return std::move(std::get<MPI_custom_type>(root_type));
		}

		throw std::runtime_error("Root type not set");
	}

	template<class T>
	auto process(T arg) -> MPI_custom_type {
		constexpr auto primitive_factory = noarr::generic_token_factory([](auto arg) { return token_list(arg); });

		noarr::print_tokens(*this, tokenizer(arg).tokenize(primitive_factory));

		return finalize();
	}

private:
	std::map<erasure, dimension_data> m_dimensions;
	std::vector<decltype(m_dimensions)::value_type> m_graveyard;
};

using num_t = int;

namespace noarr {

template<IsDim auto Dim>
inline auto mpi_bind(MPI_Comm comm) {
	int rank = 0;
	// int size;

	MPICHK(MPI_Comm_rank(comm, &rank));
	// MPI_Comm_size(comm, &size);

	return fix<Dim>(rank);
}

template<IsDim auto Dim, auto MajorDim = dim<[]() {}>{}>
inline auto mpi_block(MPI_Comm comm) {
	int rank = 0;
	int size = 0;

	MPICHK(MPI_Comm_rank(comm, &rank));
	MPICHK(MPI_Comm_size(comm, &size));

	return into_blocks<Dim, MajorDim>(size) ^ fix<MajorDim>(rank);
}

template<class Bag>
class mpi_bag : public Bag {
public:
	mpi_bag(const Bag &bag, MPI_Datatype mpi_type) : Bag(bag), mpi_type_(mpi_type) {}

	[[nodiscard]]
	auto get_mpi_type() const -> MPI_Datatype {
		return mpi_type_;
	}

	[[nodiscard]]
	auto get_bag() const {
		return *this;
	}

private:
	MPI_Datatype mpi_type_;
};

// for each of the functions, we want `fun(mpi_traverser, ...)`, `mpi_traverser.fun(...)` and `mpi_traverser | fun(...)`
// variants

// TODO: mpi_traverser_t <- this is a primitive that holds a traverser and a communicator

// 1. the traverser has a partitioned structure as a parameter and is merged into one tiled structure + a communicator
// dimension

// TODO: mpi_for(mpi_traverser, structure : noarr_bag, init, for_each, finalize): distributes the structure across the
// ranks, calls the function, gathers the results

// - init(strucure) on root structure (according to the rank dimension)
// - *scatter(structure, privatized_structure)*
// - for_each(privatized_structure) on privatized structures (according to the rank dimension)
// - *gather(privatized_structure, structure)*
// - finalize(structure) on root structure (according to the rank dimension)

// - new_structure(mpi_traverser, structure : noarr_structure) -> mpi_noarr_structure
// - new_structure(mpi_traverser, structure : noarr_bag) -> mpi_noarr_bag

// - communicate<dims...>(mpi_traverser) -> mpi_communicator<dims...>: creates a communicator for the given dimensions

// TODO: mpi_joinreduce(mpi_traverser, structure : noarr_bag, init, for_each, join) like tbb::parallel_reduce;
// automatically performs a scatter, gather is done by 1-1 sending

// - *allocate(privatized_structure)* or *scatter(structure, privatized_structure)* (depending on the bound dimension)
// - init(privatized_structure) on privatized structure
// - for_each(privatized_structure) on privatized structures
// - *send/receive(received_structure)*
// - join(privatized_structure, received_structure) on two privatized structures

// TODO: parallel_scan?

// EXAMPLES of simple use cases:

// - A * x = b (matrix-vector multiplication)
//   - A is a matrix (in)
//   - x is a vector (in)
//   - b is a vector (out)
//   - we want to distribute the matrix across the ranks and then gather the results
//   - A: (i, j) ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
//   - x: ('j')
//   - b: ('i') ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
// - x . y = z (dot product)
//   - x is a vector (in)
//   - y is a vector (in)
//   - z is a scalar (out)
//   - we want to distribute chunks of x and y across the ranks and then reduce the results
//   - x: ('i') ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
//   - y: ('i') ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
//   - z: scalar<int>() ^ bcast<'I'>() ^ mpi_bind<'I'>(comm)
//   - we wanna use the reduce collective (or the join phase)
// - histogram (counting the number of elements in each bin)
//   - histogram is a vector (out)
//   - data is a vector (in)
//   - we want to distribute the data across the ranks and then reduce the results
//   - data: ('i') ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
//   - histogram: ('h') ^ bcast<'I'>() ^ mpi_bind<'I'>(comm)
//   - we wanna use (the join phase or) the reduce collective
// - parallel sort
//   - data is a vector (in-out)
//   - we want to distribute the data across the ranks and then merge the sorted results
//   - data: ('i') ^ (block<'i', 'I'>() ^ mpi_bind<'I'>(comm))
//   - we wanna use the join phase; it merges the sorted results
// - parallel pi calculation; see https://github.com/pmodels/mpich/blob/main/examples/cpi.c
// - spectrogram calculation (essentially a 2D FFT; similar to histogram, but not exactly); see
// https://github.com/jbornschein/mpi4py-examples/blob/master/04-image-spectrogram

// EXAMPLES of more complex use cases:

// - Mandelbrot set calculation

// types of structures used in the abstraction:
// - (in) bags with MPI_Datatype (custom types)
// - (out) bags with MPI_Datatype (custom types)
// - (in-out) bags with MPI_Datatype (custom types)

// related work:

// - Boost.MPI: https://www.boost.org/doc/libs/1_87_0/doc/html/mpi/tutorial.html
// - EMPI: https://cosenza.eu/papers/SalimiBeniCCGRID23.pdf
// - A lightweight C++ MPI library:
// - Towards Modern C++ Language support for MPI

template<class Traverser>
requires IsTraverser<Traverser>
struct mpi_traverser_t : strict_contain<Traverser, MPI_Comm> {
	using base = strict_contain<Traverser, MPI_Comm>;
	using base::base;

	[[nodiscard]]
	constexpr Traverser get_traverser() const noexcept {
		return base::template get<0>();
	}

	[[nodiscard]]
	constexpr MPI_Comm get_comm() const noexcept {
		return base::template get<1>();
	}

	constexpr auto state() const noexcept { return get_traverser().state(); }

	constexpr auto get_struct() const noexcept { return get_traverser().get_struct(); }

	constexpr auto get_order() const noexcept { return get_traverser().get_order(); }

	constexpr auto top_struct() const noexcept { return get_traverser().top_struct(); }

	[[nodiscard]]
	friend auto operator^(mpi_traverser_t traverser, auto order) noexcept {
		using ordered = decltype(traverser.get_traverser() ^ order);
		return mpi_traverser_t<ordered>{traverser.get_traverser() ^ order, traverser.get_comm()};
	}

	template<auto... Dims, class F>
	requires (... && IsDim<decltype(Dims)>)
	constexpr void for_each(F &&f) const {
		get_traverser().template for_each<Dims...>([&f, comm = get_comm()](auto state) { std::forward<F>(f)(state); });
	}

	template<auto... Dims, class F>
	requires (... && IsDim<decltype(Dims)>)
	constexpr void for_sections(F &&f) const {
		get_traverser().template for_sections<Dims...>([&f, comm = get_comm()]<class Inner>(Inner inner) {
			std::forward<F>(f)(mpi_traverser_t<Inner>{inner, comm});
		});
	}

	template<auto... Dims, class F>
	requires (... && IsDim<decltype(Dims)>)
	constexpr void for_dims(F &&f) const {
		get_traverser().template for_dims<Dims...>([&f, comm = get_comm()]<class Inner>(Inner inner) {
			std::forward<F>(f)(mpi_traverser_t<Inner>{inner, comm});
		});
	}
};

template<class Traverser>
requires IsTraverser<Traverser>
mpi_traverser_t(Traverser, MPI_Comm) -> mpi_traverser_t<Traverser>;

template<class T>
concept IsMPITraverser = IsSpecialization<T, mpi_traverser_t>;

template<class Traverser>
struct to_traverser<mpi_traverser_t<Traverser>> : std::true_type {
	using type = Traverser;

	[[nodiscard]]
	static constexpr type convert(const mpi_traverser_t<Traverser> &traverser) noexcept {
		return traverser.get_traverser();
	}
};

template<class Traverser>
struct to_state<mpi_traverser_t<Traverser>> : std::true_type {
	using type = decltype(std::declval<Traverser>().state());

	[[nodiscard]]
	static constexpr type convert(const mpi_traverser_t<Traverser> &traverser) noexcept {
		return traverser.get_traverser().state();
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

template<class Traverser>
struct to_MPI_Comm<mpi_traverser_t<Traverser>> : std::true_type {
	using type = MPI_Comm;

	[[nodiscard]]
	static constexpr type convert(const mpi_traverser_t<Traverser> &traverser) noexcept {
		return traverser.get_comm();
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

// mpi bag
template<class Bag>
struct to_MPI_Datatype<mpi_bag<Bag>> : std::true_type {
	using type = MPI_Datatype;

	[[nodiscard]]
	static constexpr type convert(const mpi_bag<Bag> &bag) noexcept {
		return bag.get_mpi_type();
	}
};

template<IsMPITraverser Traverser>
constexpr auto operator|(Traverser traverser, auto f) -> decltype(traverser.for_each(f)) {
	return traverser.for_each(f);
}

template<IsMPITraverser Traverser, auto... Dims, class F>
constexpr auto operator|(Traverser traverser, const helpers::for_each_t<F, Dims...> &f)
	-> decltype(traverser.template for_each<Dims...>(f)) {
	return traverser.template for_each<Dims...>(f);
}

template<IsMPITraverser Traverser, auto... Dims, class F>
constexpr auto operator|(Traverser traverser, const helpers::for_sections_t<F, Dims...> &f)
	-> decltype(traverser.template for_sections<Dims...>(f)) {
	return traverser.template for_sections<Dims...>(f);
}

template<IsMPITraverser Traverser, auto... Dims, class F>
constexpr auto operator|(Traverser traverser, const helpers::for_dims_t<F, Dims...> &f)
	-> decltype(traverser.template for_dims<Dims...>(f)) {
	return traverser.template for_dims<Dims...>(f);
}

template<class... Bags>
requires (... && IsBag<Bags>)
constexpr auto mpi_run(auto trav, const Bags &...bags) {
	constexpr auto Dim = helpers::traviter_top_dim<decltype(trav.top_struct())>;
	const auto comm = trav.get_comm();

	return [trav, comm, ... custom_types = mpi_transform_builder{}.process(bags.structure()),
	        ... bags = bags.get_ref()](auto &&F) {
		trav ^ mpi_bind<Dim>(comm) | noarr::for_dims<>([=, &F, ... types = MPI_Datatype(custom_types)](auto inner) {
			F(inner, mpi_bag(bags, types)...);
		});
	};
}

template<auto Dim, class... Bags>
requires (IsDim<decltype(Dim)> && ... && IsBag<Bags>)
constexpr auto mpi_for(auto trav, const Bags &...bags) {
	const auto comm = trav.get_comm();
	const auto bind = mpi_bind<Dim>(comm);

	return [trav, comm, ... custom_types = mpi_transform_builder{}.process(bags.structure()), ... bags = bags.get_ref(),
	        bind,
	        // privatized bags
	        ... privatized_structs = vectors_like((bags.structure() ^ bind))](auto &&init, auto &&for_each,
	                                                                          auto &&finalize) {
		trav ^ bind |
			noarr::for_dims<>([=, &init, &for_each, &finalize, ... types = MPI_Datatype(custom_types)](auto inner) {
				init(inner, mpi_bag(bags, types)...);

				for_each(inner, mpi_bag(bags, types)...); // TODO: not like this...

				finalize(inner, mpi_bag(bags, types)...);
			});
	};
}

using TODO_TYPE = int;

// TODO: MPI_Comm custom wrapper with a destructor that calls MPI_Comm_free

inline void mpi_bcast(auto structure, ToMPIComm auto has_comm, TODO_TYPE rank) {
	MPICHK(MPI_Bcast(structure.data(), 1, structure.get_mpi_type(), rank, convert_to_MPI_Comm(has_comm)));
}

inline void mpi_gather(auto structure, ToMPIComm auto has_comm, TODO_TYPE rank) {
	MPICHK(MPI_Gather(structure.data(), 1, structure.get_mpi_type(), structure.data(), 1, structure.get_mpi_type(),
	                  rank, convert_to_MPI_Comm(has_comm)));
}

inline void mpi_gather(auto from, auto to, ToMPIComm auto has_comm, TODO_TYPE rank) {
	MPICHK(MPI_Gather(from.data(), 1, from.get_mpi_type(), to.data(), 1, to.get_mpi_type(), rank,
	                  convert_to_MPI_Comm(has_comm)));
}

template<auto AlongDim, auto... AllDims, class Traverser>
requires (IsDim<decltype(AlongDim)> && ... && IsDim<decltype(AllDims)>) &&
         (IsTraverser<Traverser> || IsMPITraverser<Traverser>)
inline auto mpi_comm_split_along(Traverser traverser, MPI_Comm comm) -> MPI_Comm { // TODO: custom wrapper
	static_assert(dim_sequence<AllDims...>::template contains<AlongDim>,
	              "The dimension must be present in the sequence");

	const auto space = noarr::scalar<char>() ^ noarr::vectors_like<AllDims...>(traverser.get_struct());

	MPI_Comm new_comm = MPI_COMM_NULL;
	MPICHK(MPI_Comm_split(comm, space | noarr::offset(traverser.state() - noarr::filter_indices<AlongDim>(traverser)),
	                      noarr::get_index<AlongDim>(traverser), &new_comm));
	return new_comm;
}

template<auto AlongDim, auto... AllDims, class MPITraverser>
requires (IsDim<decltype(AlongDim)> && ... && IsDim<decltype(AllDims)>) && IsMPITraverser<MPITraverser>
inline auto mpi_comm_split_along(MPITraverser traverser) -> MPI_Comm { // TODO: custom wrapper
	static_assert(dim_sequence<AllDims...>::template contains<AlongDim>,
	              "The dimension must be present in the sequence");

	const auto space = noarr::scalar<char>() ^ noarr::vectors_like<AllDims...>(traverser.get_struct());
	const auto comm = traverser.get_comm();

	MPI_Comm new_comm = MPI_COMM_NULL;
	MPICHK(MPI_Comm_split(comm, space | noarr::offset(traverser.state() - noarr::filter_indices<AlongDim>(traverser)),
	                      noarr::get_index<AlongDim>(traverser), &new_comm));
	return new_comm;
}

} // namespace noarr

auto main() -> int try {
	const MPI_session mpi_session;

	using namespace noarr;
	// to be shadowed by a local definition of order

	volatile std::size_t x = 300;
	volatile std::size_t y = 321;
	volatile std::size_t z = 801;

	std::cerr << "x: " << x << ", y: " << y << ", z: " << z << '\n';
	std::cerr << "Size: " << x * y * z << '\n';

	// this is just imaginary; no actual data exist
	auto data = noarr::bag(noarr::scalar<int>() ^ noarr::vectors<'x', 'y', 'z'>(2 * x, 2 * y, 2 * z), nullptr);

	// split the data into blocks
	auto structure = data ^ noarr::into_blocks<'x', 'X'>() ^ noarr::into_blocks<'y', 'Y'>() ^
	                 noarr::into_blocks<'z', 'Z'>() ^ noarr::set_length<'X', 'Y', 'Z'>(2, 2, 2); // TODO: set_length automatically

	// privatize a block corresponding to a single MPI rank
	auto block = noarr::bag(noarr::scalar<int>() ^ noarr::vectors_like<'x', 'y', 'z'>(structure));

	std::cerr << block.structure().size(empty_state) << '\n';

	// bind the structure to MPI_COMM_WORLD
	auto pre_trav = noarr::traverser(structure) ^ noarr::merge_blocks<'X', 'Y', 'r'>() ^
	                noarr::merge_blocks<'r', 'Z', 'r'>() ^ noarr::hoist<'r'>();
	auto trav = noarr::mpi_traverser_t{pre_trav, MPI_COMM_WORLD};
	// const MPI_custom_type mpi_rep = mpi_transform_builder{}.process(block.structure());

	// MPI_Aint lb = 0;
	// MPI_Aint extent = 0;

	// MPICHK(MPI_Type_get_extent((MPI_Datatype)mpi_rep, &lb, &extent));
	// std::cerr << "Extent: " << extent << '\n';
	// std::cerr << "Lower bound: " << lb << '\n';

	// TODO `block -> b` is not the most elegant solution
	mpi_run(trav, data, block)([x, y, z](const auto inner, const auto d, const auto b) {
		// TODO: magical scatter `d -> b`

		MPICHK(MPI_Barrier(inner.get_comm()));

		std::cerr << "begin" << '\n';

		// get the indices of the corresponding block
		const auto [X, Y, Z] = noarr::get_indices<'X', 'Y', 'Z'>(inner);

		// communicate along X
		// MPICHK(MPI_Comm_split(MPI_COMM_WORLD, Y * z + Z, X, &x_comm));
		MPI_Comm x_comm = noarr::mpi_comm_split_along<'X', /*all_dims: */ 'X', 'Y', 'Z'>(inner);

		// communicate along Y
		// MPICHK(MPI_Comm_split(MPI_COMM_WORLD, Z * x + X, Y, &y_comm));
		MPI_Comm y_comm = noarr::mpi_comm_split_along<'Y', /*all_dims: */ 'X', 'Y', 'Z'>(inner);

		// communicate along Z
		// MPICHK(MPI_Comm_split(MPI_COMM_WORLD, X * y + Y, Z, &z_comm));
		MPI_Comm z_comm = noarr::mpi_comm_split_along<'Z', /*all_dims: */ 'X', 'Y', 'Z'>(inner);

		// -> we wanna create a shortcut for `MPI_Comm_split(the original communicator, all other indices, the index
		// we are communicating along, &the new communicator)`
		// -> we wanna generalize the above to `split(the original communicator, all other indices, the indices we
		// are communicating along, &the new communicator)`

		// the following is just normal noarr code
		if (X + Y + Z == 0) {
			inner | [b](auto state) {
				auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

				b[state] = 1 + x + y + z;
			};
		}

		// broadcast along the communicators
		// MPICHK(MPI_Bcast(b.data(), 1, b.get_mpi_type(), 0, x_comm));
		mpi_bcast(b, x_comm, 0);

		static_assert(requires {
			// broadcast globally in the traverser; just compile, never execute
			{ mpi_bcast(b, inner, 0) } -> std::same_as<void>;
		});

		// -> we wanna generalize the above to `broadcast(b, the communicator we are broadcasting along)`

		if (Y == 0 && Z == 0) {
			inner | [b](auto state) {
				auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

				if ((std::size_t)b[state] != 1 + x + y + z) {
					std::cerr << "Error: " << b[state] << " != " << 1 + x + y + z << std::endl;
				}
			};
		} else {
			inner | [b](auto state) {
				if ((std::size_t)b[state] != 0) {
					std::cerr << "Error: " << b[state] << " != 0" << std::endl;
				}
			};
		}

		// MPICHK(MPI_Bcast(b.data(), 1, b.get_mpi_type(), 0, y_comm));
		mpi_bcast(b, y_comm, 0);

		if (Z == 0) {
			inner | [b](auto state) {
				auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

				if ((std::size_t)b[state] != 1 + x + y + z) {
					std::cerr << "Error: " << b[state] << " != " << 1 + x + y + z << std::endl;
				}
			};
		} else {
			inner | [b](auto state) {
				if ((std::size_t)b[state] != 0) {
					std::cerr << "Error: " << b[state] << " != 0" << std::endl;
				}
			};
		}

		// MPICHK(MPI_Bcast(b.data(), 1, b.get_mpi_type(), 0, z_comm));
		mpi_bcast(b, z_comm, 0);

		inner | [b](auto state) {
			auto [x, y, z] = noarr::get_indices<'x', 'y', 'z'>(state);

			if ((std::size_t)b[state] != 1 + x + y + z) {
				std::cerr << "Error: " << b[state] << " != " << 1 + x + y + z << std::endl;
			}
		};

		// free the communicators
		MPICHK(MPI_Comm_free(&x_comm));
		MPICHK(MPI_Comm_free(&y_comm));
		MPICHK(MPI_Comm_free(&z_comm));
		// TODO: -> we wanna destroy them using the raii pattern

		// TODO: gather the results (`b -> d`)
	});

	MPICHK(MPI_Barrier(MPI_COMM_WORLD));

	std::cerr << "end" << '\n';
} catch (const std::exception &e) {

	std::cerr << "Exception: " << e.what() << '\n';
	return 1;
} catch (...) {
	std::cerr << "Unknown exception" << '\n';
	return 1;
}
