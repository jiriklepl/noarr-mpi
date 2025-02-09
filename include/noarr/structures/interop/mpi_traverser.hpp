#ifndef NOARR_STRUCTURES_INTEROP_MPI_TRAVERSER_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_TRAVERSER_HPP

#include <mpi.h>

#include <noarr/structures/base/contain.hpp>
#include <noarr/structures/base/state.hpp>
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
	constexpr auto get_bind() const {
		int rank = 0;
		int size = 0;

		MPICHK(MPI_Comm_rank(get_comm(), &rank));
		MPICHK(MPI_Comm_size(get_comm(), &size));

		if constexpr (decltype(get_traverser().top_struct())::template has_length<Dim, noarr::state<>>()) {
			if (get_traverser().top_struct().template length<Dim>(empty_state) != size) {
				throw std::runtime_error("The MPI communicator size does not match the structure length");
			}

			return fix<Dim>(rank);
		} else {
			return set_length<Dim>(size) ^ fix<Dim>(rank);
		}
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
	constexpr auto state() const {
		return (get_traverser() ^ get_bind()).state();
	}

	constexpr auto state(int root) const noexcept {
		int size = 0;
		MPICHK(MPI_Comm_size(get_comm(), &size));
		return (get_traverser() ^ set_length<Dim>(size) ^ fix<Dim>(root)).state();
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
	constexpr auto top_struct() const {
		return (get_traverser() ^ get_bind()).top_struct();
	}

	[[nodiscard]]
	constexpr auto top_struct(int root) const {
		return (get_traverser() ^ fix<Dim>(root) ^ get_bind()).top_struct();
	}

	template<auto... Dims, class F>
	requires (... && IsDim<decltype(Dims)>)
	constexpr void for_each(F &&f) const {
		(get_traverser() ^ get_bind()).template for_each<Dims...>([&f, comm = get_comm()](auto state) {
			std::forward<F>(f)(state);
		});
	}

	template<auto... Dims, class F>
	requires (... && IsDim<decltype(Dims)>)
	constexpr void for_sections(F &&f) const {
		(get_traverser() ^ get_bind())
			.template for_sections<Dims...>([&f, comm = get_comm()]<class Inner>(Inner inner) {
				std::forward<F>(f)(mpi_traverser_t<Dim, Inner>{inner, comm});
			});
	}

	template<auto... Dims, class F>
	requires (... && IsDim<decltype(Dims)>)
	constexpr void for_dims(F &&f) const {
		(get_traverser() ^ get_bind()).template for_dims<Dims...>([&f, comm = get_comm()]<class Inner>(Inner inner) {
			std::forward<F>(f)(mpi_traverser_t<Dim, Inner>{inner, comm});
		});
	}
};

template<IsDim auto Dim, IsTraverser Traverser>
constexpr auto mpi_traverser(Traverser traverser, const ToMPIComm auto &has_comm) noexcept {
	const auto comm = convert_to_MPI_Comm(has_comm);

	return mpi_traverser_t<Dim, Traverser>{traverser, comm};
}

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
	using type =
		std::remove_cvref_t<decltype(std::declval<Traverser>().get_traverser() ^ std::declval<Traverser>().get_bind())>;

	[[nodiscard]]
	static constexpr type convert(const Traverser &traverser) noexcept {
		return traverser.get_traverser() ^ traverser.get_bind();
	}
};

template<IsMpiTraverser Traverser>
struct to_state<Traverser> : std::true_type {
	using type = decltype(std::declval<Traverser>().state());

	[[nodiscard]]
	static constexpr type convert(const Traverser &traverser) noexcept {
		return traverser.state();
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
