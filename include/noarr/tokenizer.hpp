#ifndef NOARR_TOKENIZER_HPP
#define NOARR_TOKENIZER_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "noarr/structures/base/contain.hpp"
#include "noarr/structures/base/structs_common.hpp"
#include "noarr/structures/structs/scalar.hpp"

namespace noarr {

template<class NoarrClass>
class tokenizer_t;

template<class NoarrClass>
tokenizer_t(NoarrClass &&) -> tokenizer_t<std::remove_cvref_t<NoarrClass>>;

template<class NoarrClass>
constexpr auto tokenizer(NoarrClass &&noarr_class) {
	return tokenizer_t(std::forward<NoarrClass>(noarr_class));
}

template<class F>
class generic_token_factory_t : public F {
public:
	using F::operator();

	template<class T>
	constexpr auto make_token(T &&value) const {
		return (*this)(std::forward<T>(value));
	}
};

template<class T>
constexpr auto generic_token_factory(T &&t) {
	return generic_token_factory_t<T>(std::forward<T>(t));
}

template<class... Tokens>
using token_list_t = flexible_contain<Tokens...>;

template<class... Tokens>
constexpr auto token_list(Tokens &&...tokens) {
	return token_list_t<std::remove_cvref_t<Tokens>...>(std::forward<Tokens>(tokens)...);
}

template<class... Tokens1, class... Tokens2>
constexpr auto operator+(const token_list_t<Tokens1...> &lhs, const token_list_t<Tokens2...> &rhs) {
	return contain_cat(lhs, rhs);
}

template<class Printer, std::size_t... Is>
constexpr auto print_tokens(Printer &&printer, const auto &tokens, [[maybe_unused]] std::index_sequence<Is...> is) {
	return (void(0), ..., std::forward<Printer>(printer)(tokens.template get<Is>()));
}

template<class Printer, class... Tokens>
constexpr auto print_tokens(Printer &&printer, const token_list_t<Tokens...> &tokens) {
	print_tokens(std::forward<Printer>(printer), tokens, std::index_sequence_for<Tokens...>());
}

template<IsStruct Structure>
class tokenizer_t<Structure> : flexible_contain<Structure> {
public:
	using flexible_contain<Structure>::flexible_contain;

	constexpr auto tokenize(auto &&tf) const {
		return tokenizer(this->get().sub_structure()).tokenize(tf) + tf.make_token(this->get());
		// tokenizer(typename std::remove_cvref_t<decltype(this->get())>::params()).tokenize(tf)
	}
};

template<IsProtoStruct ProtoStruct>
class tokenizer_t<ProtoStruct> : flexible_contain<ProtoStruct> {
public:
	using flexible_contain<ProtoStruct>::flexible_contain;

	constexpr auto tokenize(auto &&tf) const {
		return tf.make_token(this->get());
		// tokenizer(typename std::remove_cvref_t<decltype(this->get())>::params()).tokenize(tf)
	}
};

template<class InnerProtoStructs, class OuterProtoStruct>
class tokenizer_t<compose_proto<pack<InnerProtoStructs>, OuterProtoStruct>>
	: flexible_contain<compose_proto<pack<InnerProtoStructs>, OuterProtoStruct>> {
public:
	using flexible_contain<compose_proto<pack<InnerProtoStructs>, OuterProtoStruct>>::flexible_contain;

	constexpr auto tokenize(auto &&tf) const {
		return tokenizer(this->get().template get<0>().get()).tokenize(tf) +
		       tokenizer(this->get().template get<1>()).tokenize(tf);
		// tokenizer(typename std::remove_cvref_t<decltype(this->get())>::params()).tokenize(tf)
	}
};

template<class T>
class tokenizer_t<scalar<T>> : flexible_contain<scalar<T>> {
public:
	using flexible_contain<scalar<T>>::flexible_contain;

	template<class TokenFactory>
	constexpr auto tokenize(TokenFactory &&tf) const {
		return std::forward<TokenFactory>(tf).make_token(this->get());
	}
};

template<class... Params>
class tokenizer_t<struct_params<Params...>> : flexible_contain<struct_params<Params...>> {
public:
	using flexible_contain<struct_params<Params...>>::flexible_contain;

	template<class TokenFactory>
	constexpr auto tokenize(TokenFactory &&tf) const {
		return tokenize(std::forward<TokenFactory>(tf), std::index_sequence_for<Params...>());
	}

private:
	template<class TokenFactory, std::size_t... Is>
	constexpr auto tokenize(TokenFactory &&tf, [[maybe_unused]] std::index_sequence<Is...> is) const {
		return contain_cat(std::forward<TokenFactory>(tf).make_token(this->get().template get<Is>())...);
	}
};

} // namespace noarr

#endif // NOARR_TOKENIZER_HPP
