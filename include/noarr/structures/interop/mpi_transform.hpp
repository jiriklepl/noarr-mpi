#ifndef NOARR_STRUCTURES_INTEROP_MPI_TRANSFORM_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_TRANSFORM_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <map>
#include <memory>
#include <set>
#include <vector>

#include <mpi.h>

#include <noarr/structures/structs/bcast.hpp>
#include <noarr/structures/structs/blocks.hpp>
#include <noarr/structures/structs/layouts.hpp>
#include <noarr/structures/structs/scalar.hpp>
#include <noarr/structures/structs/setters.hpp>
#include <noarr/structures/structs/slice.hpp>
#include <noarr/structures/structs/views.hpp>

#include "../extra/tokenizer.hpp"
#include "../interop/mpi_utility.hpp"

namespace noarr {

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
class translate<scalar<T>> {
public:
	static auto get() -> erasure { return erasure::get<scalar<T>>(); }
};

template<auto Dim, class T>
class translate<vector_t<Dim, T>> {
public:
	static auto get() -> erasure { return erasure::get<dim<Dim>>(); }
};

template<auto Dim, class T, class LenT>
class translate<set_length_t<Dim, T, LenT>> {
public:
	static auto get() -> erasure { return erasure::get<dim<Dim>>(); }
};

template<auto OldDim, auto DimMajor, auto DimMinor, class T>
class translate<into_blocks_t<OldDim, DimMajor, DimMinor, T>> {
public:
	static auto get() -> erasure {
		// get the major dimension, as it is the outermost one
		return erasure::get<dim<DimMajor>>();
	}
};

template<auto Dim, class T>
class translate<hoist_t<Dim, T>> {
public:
	static auto get() -> erasure { return erasure::get<dim<Dim>>(); }
};

template<auto Dim, class LenT>
class translate<set_length_proto<Dim, LenT>> {
public:
	static auto get() -> erasure { return erasure::get<dim<Dim>>(); }
};

template<auto Dim, class T, class LenT>
class translate<shift_t<Dim, T, LenT>> {
public:
	static auto get() -> erasure { return erasure::get<dim<Dim>>(); }
};

template<auto Dim, class T, class StartT, class LenT>
class translate<slice_t<Dim, T, StartT, LenT>> {
public:
	static auto get() -> erasure { return erasure::get<dim<Dim>>(); }
};

template<auto Dim, class T, class StartT, class EndT>
class translate<span_t<Dim, T, StartT, EndT>> {
public:
	static auto get() -> erasure { return erasure::get<dim<Dim>>(); }
};

template<auto Dim, class T>
class translate<bcast_t<Dim, T>> {
public:
	static auto get() -> erasure { return erasure::get<dim<Dim>>(); }
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

inline auto make_size_expression(std::size_t value) -> size_expression::ptr {
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

inline auto make_size_expression() -> size_expression::ptr { return std::make_unique<unknown_size_expression>(); }

template<dimension_data::param Param>
class param_size_expression : public size_expression {
	using param = dimension_data::param;

public:
	explicit param_size_expression(erasure e) : m_erasure(e) {}

	[[nodiscard]]
	auto get(std::map<erasure, dimension_data> &dimensions) const -> std::size_t override {
		const auto it = dimensions.find(m_erasure);
		if (it == dimensions.end()) {
			throw std::runtime_error(__FILE__ + std::string(":") + std::to_string(__LINE__) +
			                         std::string(": Dimension not found"));
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
inline auto make_size_expression(erasure e) -> size_expression::ptr {
	return std::make_unique<param_size_expression<p>>(e);
}

template<dimension_data::param p>
inline auto make_size_expression(dimension_data &data) -> size_expression::ptr {
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
inline auto make_size_expression(size_expression::ptr left, size_expression::ptr right) -> size_expression::ptr {
	return std::make_unique<binary_size_expression<Op>>(std::move(left), std::move(right));
}

struct range_size_t {};

constexpr range_size_t range_size;

inline auto make_size_expression(range_size_t /*unused*/, erasure e) -> size_expression::ptr {
	using param = dimension_data::param;
	return make_size_expression<param::extent>(e);
}

inline auto make_size_expression(range_size_t /*unused*/, dimension_data &e) -> size_expression::ptr {
	return e.extent->clone(); // TODO: check whether clone is a good idea
}

class mpi_transform_builder {
public:
	template<class T>
	void operator()(scalar<T> /*unused*/) {
		const auto e = erasure::get<scalar<T>>();
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
	requires IsDim<decltype(Dim)>
	void operator()(vector_t<Dim, T> /*unused*/) {
		const auto e = erasure::get<dim<Dim>>();
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
	requires IsDim<decltype(Dim)> && IsDim<decltype(DimMajor)> && IsDim<decltype(DimMinor)>
	void operator()(into_blocks_t<Dim, DimMajor, DimMinor, T> /*unused*/) {
		// take the dimension at Dim
		const auto old = erasure::get<dim<Dim>>();
		const auto old_it = m_dimensions.find(old);
		if (old_it == m_dimensions.end()) {
			throw std::runtime_error(__FILE__ + std::string(":") + std::to_string(__LINE__) +
			                         std::string(": Dimension not found"));
		}

		// kill the old dimension
		m_graveyard.emplace_back(std::move(*old_it));
		m_dimensions.erase(old_it);

		auto &moved_data = m_graveyard.back().second;

		moved_data.type = MPI_DATATYPE_NULL;

		const auto e_minor = erasure::get<dim<DimMinor>>();
		const auto e_major = erasure::get<dim<DimMajor>>();

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

		m_dimensions.try_emplace(
			e_minor, dimension_data{
						 .start = make_size_expression(0),
						 .end = make_size_expression<std::divides<>>(make_size_expression(range_size, moved_data),
		                                                             make_size_expression(range_size, e_major)),
						 .extent = make_size_expression<std::divides<>>(make_size_expression(range_size, moved_data),
		                                                                make_size_expression(range_size, e_major)),
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
	requires IsDim<decltype(Dim)>
	void operator()(hoist_t<Dim, T> /*unused*/) {
		/* do nothing */
	}

	template<auto Dim, class T, class LenT>
	requires IsDim<decltype(Dim)>
	void operator()(set_length_t<Dim, T, LenT> sl) {
		const auto e = erasure::get<dim<Dim>>();

		const auto it = m_dimensions.find(e);
		if (it == m_dimensions.end()) {
			return;
		}

		it->second.start = make_size_expression(0);
		it->second.extent = make_size_expression(sl.len());
		it->second.end = make_size_expression(sl.len());
	}

	template<auto Dim, class T, class LenT>
	requires IsDim<decltype(Dim)>
	void operator()(shift_t<Dim, T, LenT> shift) {
		const auto e = erasure::get<dim<Dim>>();
		const auto it = m_dimensions.find(e);

		if (it == m_dimensions.end()) {
			throw std::runtime_error(__FILE__ + std::string(":") + std::to_string(__LINE__) +
			                         std::string(": Dimension not found"));
		}

		if (dynamic_cast<unknown_size_expression *>(it->second.start.get())) {
			// TODO: in noarr (or mu), it is technically possible to first shift the dimension and then set the length
			throw std::runtime_error("Start not set");
		}

		it->second.start =
			make_size_expression<std::plus<>>(std::move(it->second.start), make_size_expression(shift.start()));
	}

	template<auto Dim, class T, class IdxT>
	requires IsDim<decltype(Dim)>
	void operator()(fix_t<Dim, T, IdxT> fix) {
		const auto e = erasure::get<dim<Dim>>();
		const auto it = m_dimensions.find(e);

		if (it == m_dimensions.end()) {
			return;
		}

		if (dynamic_cast<unknown_size_expression *>(it->second.start.get())) {
			// TODO: in noarr (or mu), it is technically possible to first shift the dimension and then set the length
			throw std::runtime_error("Start not set");
		}

		it->second.start =
			make_size_expression<std::plus<>>(std::move(it->second.start), make_size_expression(fix.idx()));
		it->second.end =
			make_size_expression<std::plus<>>(std::move(it->second.end), make_size_expression(fix.idx() + 1));
	}

	template<auto Dim, class T, class StartT, class LenT>
	requires IsDim<decltype(Dim)>
	void operator()(slice_t<Dim, T, StartT, LenT> slice) {
		const auto e = erasure::get<dim<Dim>>();
		const auto it = m_dimensions.find(e);

		if (it == m_dimensions.end()) {
			throw std::runtime_error(__FILE__ + std::string(":") + std::to_string(__LINE__) +
			                         std::string(": Dimension not found"));
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
	requires IsDim<decltype(Dim)>
	void operator()(span_t<Dim, T, StartT, EndT> span) {
		const auto e = erasure::get<dim<Dim>>();
		const auto it = m_dimensions.find(e);

		if (it == m_dimensions.end()) {
			throw std::runtime_error(__FILE__ + std::string(":") + std::to_string(__LINE__) +
			                         std::string(": Dimension not found"));
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
	void operator()(bcast_t<Dim, T> /*unused*/) {
		const auto e = erasure::get<dim<Dim>>();
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

	void operator()(IsProtoStruct auto /*unused*/) {
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
		constexpr auto primitive_factory = generic_token_factory([](auto arg) { return token_list(arg); });

		print_tokens(*this, tokenizer(arg).tokenize(primitive_factory));

		return finalize();
	}

private:
	std::map<erasure, dimension_data> m_dimensions;
	std::vector<decltype(m_dimensions)::value_type> m_graveyard;
};

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_TRANSFORM_HPP
