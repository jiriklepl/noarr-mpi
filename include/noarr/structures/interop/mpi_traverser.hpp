#ifndef NOARR_STRUCTURES_INTEROP_MPI_TRAVERSER_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_TRAVERSER_HPP

#include <mpi.h>

#include <noarr/structures/base/contain.hpp>
#include <noarr/structures/base/utility.hpp>
#include <noarr/structures/extra/traverser.hpp>

#include "../interop/mpi_bag.hpp"
#include "../interop/mpi_structs.hpp"

namespace noarr {

template<IsDim auto Dim, class Traverser>
requires IsTraverser<Traverser>
struct mpi_traverser_t : strict_contain<Traverser, MPI_Comm> {
	using base = strict_contain<Traverser, MPI_Comm>;
	using base::base;

	static constexpr auto dim = Dim;

	[[nodiscard]]
	constexpr auto get_bind() const noexcept {
		return mpi_bind<Dim>(get_comm());
	}

	[[nodiscard]]
	constexpr Traverser get_traverser() const noexcept {
		return base::template get<0>();
	}

	[[nodiscard]]
	constexpr MPI_Comm get_comm() const noexcept {
		return base::template get<1>();
	}

	[[nodiscard]]
	constexpr auto state() const noexcept {
		return get_traverser().state();
	}

	[[nodiscard]]
	constexpr auto get_struct() const noexcept {
		return get_traverser().get_struct();
	}

	[[nodiscard]]
	constexpr auto get_order() const noexcept {
		return get_traverser().get_order();
	}

	[[nodiscard]]
	constexpr auto top_struct() const noexcept {
		return get_traverser().top_struct();
	}

	[[nodiscard]]
	friend auto operator^(mpi_traverser_t traverser, auto order) noexcept {
		using ordered = decltype(traverser.get_traverser() ^ order);
		return mpi_traverser_t<Dim, ordered>{traverser.get_traverser() ^ order, traverser.get_comm()};
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
			std::forward<F>(f)(mpi_traverser_t<Dim, Inner>{inner, comm});
		});
	}

	template<auto... Dims, class F>
	requires (... && IsDim<decltype(Dims)>)
	constexpr void for_dims(F &&f) const {
		get_traverser().template for_dims<Dims...>([&f, comm = get_comm()]<class Inner>(Inner inner) {
			std::forward<F>(f)(mpi_traverser_t<Dim, Inner>{inner, comm});
		});
	}
};

template<IsDim auto Dim, IsTraverser Traverser>
constexpr auto mpi_traverser(Traverser traverser, MPI_Comm comm) noexcept {
	using trav = decltype(traverser ^ mpi_bind<Dim>(comm));
	return mpi_traverser_t<Dim, trav>{traverser ^ mpi_bind<Dim>(comm), comm};
}

// TODO: the version with top-dim

template<class T>
struct is_mpi_traverser : std::false_type {};

template<class T>
constexpr bool is_mpi_traverser_v = is_mpi_traverser<T>::value;

template<class T>
concept IsMpiTraverser = is_mpi_traverser_v<std::remove_cvref_t<T>>;

template<IsDim auto Dim, IsTraverser Traverser>
struct is_mpi_traverser<mpi_traverser_t<Dim, Traverser>> : std::true_type {};

template<IsMpiTraverser Traverser>
struct to_traverser<Traverser> : std::true_type {
	using type = std::remove_cvref_t<decltype(std::declval<Traverser>().get_traverser())>;

	[[nodiscard]]
	static constexpr type convert(const Traverser &traverser) noexcept {
		return traverser.get_traverser();
	}
};

template<IsMpiTraverser Traverser>
struct to_state<Traverser> : std::true_type {
	using type = decltype(std::declval<Traverser>().state());

	[[nodiscard]]
	static constexpr type convert(const Traverser &traverser) noexcept {
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

template<IsMpiTraverser Traverser>
struct to_MPI_Comm<Traverser> : std::true_type {
	using type = decltype(std::declval<Traverser>().get_comm());

	[[nodiscard]]
	static constexpr type convert(const Traverser &traverser) noexcept {
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

template<IsMpiTraverser Traverser>
constexpr auto operator|(Traverser traverser, auto f) -> decltype(traverser.for_each(f)) {
	return traverser.for_each(f);
}

template<IsMpiTraverser Traverser, auto... Dims, class F>
constexpr auto operator|(Traverser traverser, const helpers::for_each_t<F, Dims...> &f)
	-> decltype(traverser.template for_each<Dims...>(f)) {
	return traverser.template for_each<Dims...>(f);
}

template<IsMpiTraverser Traverser, auto... Dims, class F>
constexpr auto operator|(Traverser traverser, const helpers::for_sections_t<F, Dims...> &f)
	-> decltype(traverser.template for_sections<Dims...>(f)) {
	return traverser.template for_sections<Dims...>(f);
}

template<IsMpiTraverser Traverser, auto... Dims, class F>
constexpr auto operator|(Traverser traverser, const helpers::for_dims_t<F, Dims...> &f)
	-> decltype(traverser.template for_dims<Dims...>(f)) {
	return traverser.template for_dims<Dims...>(f);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_TRAVERSER_HPP
