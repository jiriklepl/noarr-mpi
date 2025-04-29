#ifndef NOARR_STRUCTURES_INTEROP_MPI_BAG_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_BAG_HPP

#include <type_traits>
#include <utility>

#include <mpi.h>

#include <noarr/structures/base/utility.hpp>
#include <noarr/structures/extra/to_struct.hpp>
#include <noarr/structures/interop/bag.hpp>

#include "../interop/mpi_utility.hpp"

namespace noarr {

template<class Bag>
class mpi_bag_t : public Bag {
public:
	mpi_bag_t(const Bag &bag, MPI_custom_type mpi_type) : Bag(bag), mpi_type_(std::move(mpi_type)) {
		mpi_type_.commit();
	}

	[[nodiscard]]
	auto get_mpi_type() const -> MPI_Datatype {
		return convert_to_MPI_Datatype(mpi_type_);
	}

	[[nodiscard]]
	const Bag &get_bag() const {
		return *this;
	}

private:
	MPI_custom_type mpi_type_;
};

template<class T>
concept IsMpiBag = IsSpecialization<T, mpi_bag_t>;

template<IsMpiBag Bag>
struct to_struct<Bag> : std::true_type {
	using type = decltype(convert_to_struct(std::declval<Bag>().get_bag()));

	static constexpr type convert(const Bag &bag) { return convert_to_struct(bag.get_bag()); }
};

template<IsMpiBag Bag>
struct to_MPI_Datatype<Bag> : std::true_type {
	using type = MPI_Datatype;

	[[nodiscard]]
	static constexpr type convert(const Bag &bag) noexcept {
		return bag.get_mpi_type();
	}
};

template<class Bag>
mpi_bag_t(Bag, MPI_Datatype) -> mpi_bag_t<Bag>;

template<class Bag>
[[nodiscard]]
auto mpi_bag(const Bag &bag) -> mpi_bag_t<Bag> {
	return mpi_bag_t(bag, mpi_transform(convert_to_struct(bag)));
}

template<class Bag>
[[nodiscard]]
auto mpi_bag(const Bag &bag, MPI_custom_type mpi_type) -> mpi_bag_t<Bag> {
	return mpi_bag_t(bag, std::move(mpi_type));
}

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_BAG_HPP
