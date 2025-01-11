#ifndef NOARR_STRUCTURES_INTEROP_MPI_BAG_HPP
#define NOARR_STRUCTURES_INTEROP_MPI_BAG_HPP

#include <utility>

#include <mpi.h>

#include <noarr/structures/interop/bag.hpp>
#include <noarr/structures/extra/to_struct.hpp>
#include <noarr/structures/base/utility.hpp>

namespace noarr {

template<class Bag>
class mpi_bag : public Bag {
public:
	mpi_bag(const Bag &bag, MPI_Datatype mpi_type) : Bag(bag), mpi_type_(mpi_type) {}

	[[nodiscard]]
	auto get_mpi_type() const -> MPI_Datatype {
		return mpi_type_;
	}

	[[nodiscard]]
	const Bag &get_bag() const {
		return *this;
	}

private:
	MPI_Datatype mpi_type_;
};

template<class T>
concept IsMpiBag = IsSpecialization<T, mpi_bag>;

template<IsMpiBag Bag>
struct to_struct<Bag> : std::true_type {
	using type = decltype(convert_to_struct(std::declval<Bag>().get_bag()));

	static constexpr type convert(const Bag &bag) {
		return convert_to_struct(bag.get_bag());
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_MPI_BAG_HPP
