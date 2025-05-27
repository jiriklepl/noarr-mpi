#ifndef NOARR_MPI_TRAVERSER_HPP
#define NOARR_MPI_TRAVERSER_HPP

#include <stdexcept>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include <noarr/structures/base/contain.hpp>
#include <noarr/structures/base/state.hpp>
#include <noarr/structures/base/utility.hpp>
#include <noarr/structures/extra/traverser.hpp>

#include "../mpi/utility.hpp"

namespace noarr::mpi {

namespace helpers {
struct declare_safe {};
}

template<IsDim auto Dim, class Traverser>
requires IsTraverser<Traverser>
struct mpi_traverser_t : strict_contain<Traverser, MPI_Comm, int, int> {
	using base = strict_contain<Traverser, MPI_Comm, int, int>;
	using base::base;

	mpi_traverser_t(Traverser traverser, MPI_Comm comm) : base{traverser, comm, 0, 0} {
		MPICHK(MPI_Comm_rank(comm, &const_cast<int &>(base::template get<2>())));
		MPICHK(MPI_Comm_size(comm, &const_cast<int &>(base::template get<3>())));

		if constexpr (decltype(get_traverser().top_struct())::template has_length<Dim, noarr::state<>>()) {
			if (get_traverser().top_struct().template length<Dim>(empty_state) != base::template get<3>()) {
				throw std::runtime_error("The MPI communicator size does not match the structure length");
			}
		}
	}

	explicit mpi_traverser_t(Traverser traverser, MPI_Comm comm, int rank, int size)
		: base{traverser, comm, rank, size} {
		if constexpr (decltype(get_traverser().top_struct())::template has_length<Dim, noarr::state<>>()) {
			if (get_traverser().top_struct().template length<Dim>(empty_state) != size) {
				throw std::runtime_error("The MPI communicator size does not match the structure length");
			}
		}
	}

	explicit mpi_traverser_t(Traverser traverser, MPI_Comm comm, int rank, int size, helpers::declare_safe /*unused*/) noexcept
		: base{traverser, comm, rank, size} {}

	static constexpr auto dim = Dim;

	[[nodiscard]]
	constexpr int get_rank() const noexcept {
		return base::template get<2>();
	}

	[[nodiscard]]
	constexpr int get_size() const noexcept {
		return base::template get<3>();
	}

	template<IsProtoStruct Order>
	[[nodiscard]]
	constexpr auto order(Order order) const noexcept {
		const auto new_traverser = get_traverser() ^ order;
		const auto new_comm = get_comm();
		const auto new_rank = get_rank();
		const auto new_size = get_size();
		return mpi_traverser_t<Dim, decltype(new_traverser)>{new_traverser, new_comm, new_rank, new_size};
	}

	[[nodiscard]]
	constexpr auto get_bind() const noexcept {
		const int rank = get_rank();
		const int size = get_size();

		if constexpr (decltype(get_traverser().top_struct())::template has_length<Dim, noarr::state<>>()) {
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
	constexpr auto state() const noexcept {
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
	constexpr auto top_struct() const noexcept {
		return (get_traverser() ^ get_bind()).top_struct();
	}

	[[nodiscard]]
	constexpr auto top_struct(int root) const noexcept {
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
			.template for_sections<Dims...>(
				[&f, comm = get_comm(), rank = get_rank(), size = get_size()]<class Inner>(Inner inner) {
					std::forward<F>(f)(mpi_traverser_t<Dim, Inner>{inner, comm, rank, size, helpers::declare_safe{}});
				});
	}

	template<auto... Dims, class F>
	requires (... && IsDim<decltype(Dims)>)
	constexpr void for_dims(F &&f) const {
		(get_traverser() ^ get_bind())
			.template for_dims<Dims...>(
				[&f, comm = get_comm(), rank = get_rank(), size = get_size()]<class Inner>(Inner inner) {
					std::forward<F>(f)(mpi_traverser_t<Dim, Inner>{inner, comm, rank, size, helpers::declare_safe{}});
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
struct to_MPI_Comm<Traverser> : std::true_type {
	using type = decltype(std::declval<Traverser>().get_comm());

	[[nodiscard]]
	static constexpr type convert(const Traverser &traverser) noexcept {
		return traverser.get_comm();
	}
};

template<IsMpiTraverser Traverser, IsProtoStruct Struct>
constexpr auto operator^(const Traverser &traverser, Struct s) noexcept {
	return traverser.order(s);
}

template<IsMpiTraverser Traverser>
constexpr auto operator|(Traverser traverser, auto f) -> decltype(traverser.for_each(f)) {
	return traverser.for_each(f);
}

template<IsMpiTraverser Traverser, auto... Dims, class F>
constexpr auto operator|(Traverser traverser, const noarr::helpers::for_each_t<F, Dims...> &f)
	-> decltype(traverser.template for_each<Dims...>(f)) {
	return traverser.template for_each<Dims...>(f);
}

template<IsMpiTraverser Traverser, auto... Dims, class F>
constexpr auto operator|(Traverser traverser, const noarr::helpers::for_sections_t<F, Dims...> &f)
	-> decltype(traverser.template for_sections<Dims...>(f)) {
	return traverser.template for_sections<Dims...>(f);
}

template<IsMpiTraverser Traverser, auto... Dims, class F>
constexpr auto operator|(Traverser traverser, const noarr::helpers::for_dims_t<F, Dims...> &f)
	-> decltype(traverser.template for_dims<Dims...>(f)) {
	return traverser.template for_dims<Dims...>(f);
}

} // namespace noarr::mpi

namespace noarr {

template<mpi::IsMpiTraverser Traverser>
struct to_traverser<Traverser> : std::true_type {
	using type =
		std::remove_cvref_t<decltype(std::declval<Traverser>().get_traverser() ^ std::declval<Traverser>().get_bind())>;

	[[nodiscard]]
	static constexpr type convert(const Traverser &traverser) noexcept {
		return traverser.get_traverser() ^ traverser.get_bind();
	}
};

template<mpi::IsMpiTraverser Traverser>
struct to_state<Traverser> : std::true_type {
	using type = decltype(std::declval<Traverser>().state());

	[[nodiscard]]
	static constexpr type convert(const Traverser &traverser) noexcept {
		return traverser.state();
	}
};

} // namespace noarr

#endif // NOARR_MPI_TRAVERSER_HPP
