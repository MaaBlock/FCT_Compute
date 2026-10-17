#ifndef FCT_COMPUTE_CONCEPTS_H
#define FCT_COMPUTE_CONCEPTS_H

#include "ComputeTypeTraits.h"

namespace FCT::Compute {

    template<typename T>
    concept ScalarConcept =
        std::is_same_v<std::remove_cvref_t<T>, bool> ||
        std::is_same_v<std::remove_cvref_t<T>, int8_t> ||
        std::is_same_v<std::remove_cvref_t<T>, uint8_t> ||
        std::is_same_v<std::remove_cvref_t<T>, int16_t> ||
        std::is_same_v<std::remove_cvref_t<T>, uint16_t> ||
        std::is_same_v<std::remove_cvref_t<T>, int32_t> ||
        std::is_same_v<std::remove_cvref_t<T>, uint32_t> ||
        std::is_same_v<std::remove_cvref_t<T>, int64_t> ||
        std::is_same_v<std::remove_cvref_t<T>, uint64_t> ||
        std::is_same_v<std::remove_cvref_t<T>, luisa::half> ||
        std::is_same_v<std::remove_cvref_t<T>, float> ||
        std::is_same_v<std::remove_cvref_t<T>, double>;

    template<typename T>
    concept FCTVectorConcept =
        std::is_same_v<std::remove_cvref_t<T>, FCT::Vec2> ||
        std::is_same_v<std::remove_cvref_t<T>, FCT::Vec3> ||
        std::is_same_v<std::remove_cvref_t<T>, FCT::Vec4> ||
        std::is_same_v<std::remove_cvref_t<T>, PackedFloat3>;

    template<typename T>
    concept LuisaVectorConcept = luisa::is_vector_v<std::remove_cvref_t<T>>;

    template<typename T>
    concept LuisaMatrixConcept = luisa::is_matrix_v<std::remove_cvref_t<T>>;

    template<typename T>
    concept BufferElementConcept =
        luisa::compute::is_valid_buffer_element_v<StorageType_t<T>>;

    template<typename T>
    concept ImageElementConcept =
        luisa::compute::is_legal_image_element<std::remove_cvref_t<T>>;

    template<FCT::Format F>
    concept SupportedFormatConcept = FormatTraits<F>::isSupported;

    template<size_t Dim>
    concept KernelDimensionConcept = (Dim == 1u || Dim == 2u || Dim == 3u);

    template<typename T>
    concept AggregateStructConcept =
        std::is_aggregate_v<std::remove_cvref_t<T>> &&
        std::is_standard_layout_v<std::remove_cvref_t<T>> &&
        BufferElementConcept<T> &&
        (alignof(StorageType_t<T>) >= 4u);

    namespace Detail {

        template<typename T, size_t Count>
        constexpr auto DecomposeAggregateToTuple() noexcept {
            T instance{};
            if constexpr (Count == 1u) {
                auto [m0] = instance;
                return std::tuple<decltype(m0)>{};
            } else if constexpr (Count == 2u) {
                auto [m0, m1] = instance;
                return std::tuple<decltype(m0), decltype(m1)>{};
            } else if constexpr (Count == 3u) {
                auto [m0, m1, m2] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2)>{};
            } else if constexpr (Count == 4u) {
                auto [m0, m1, m2, m3] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3)>{};
            } else if constexpr (Count == 5u) {
                auto [m0, m1, m2, m3, m4] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4)>{};
            } else if constexpr (Count == 6u) {
                auto [m0, m1, m2, m3, m4, m5] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5)>{};
            } else if constexpr (Count == 7u) {
                auto [m0, m1, m2, m3, m4, m5, m6] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5), decltype(m6)>{};
            } else if constexpr (Count == 8u) {
                auto [m0, m1, m2, m3, m4, m5, m6, m7] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5), decltype(m6), decltype(m7)>{};
            } else if constexpr (Count == 9u) {
                auto [m0, m1, m2, m3, m4, m5, m6, m7, m8] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5), decltype(m6), decltype(m7), decltype(m8)>{};
            } else if constexpr (Count == 10u) {
                auto [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5), decltype(m6), decltype(m7), decltype(m8), decltype(m9)>{};
            } else if constexpr (Count == 11u) {
                auto [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5), decltype(m6), decltype(m7), decltype(m8), decltype(m9), decltype(m10)>{};
            } else if constexpr (Count == 12u) {
                auto [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5), decltype(m6), decltype(m7), decltype(m8), decltype(m9), decltype(m10), decltype(m11)>{};
            } else if constexpr (Count == 13u) {
                auto [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5), decltype(m6), decltype(m7), decltype(m8), decltype(m9), decltype(m10), decltype(m11), decltype(m12)>{};
            } else if constexpr (Count == 14u) {
                auto [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5), decltype(m6), decltype(m7), decltype(m8), decltype(m9), decltype(m10), decltype(m11), decltype(m12), decltype(m13)>{};
            } else if constexpr (Count == 15u) {
                auto [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5), decltype(m6), decltype(m7), decltype(m8), decltype(m9), decltype(m10), decltype(m11), decltype(m12), decltype(m13), decltype(m14)>{};
            } else if constexpr (Count == 16u) {
                auto [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15] = instance;
                return std::tuple<decltype(m0), decltype(m1), decltype(m2), decltype(m3), decltype(m4), decltype(m5), decltype(m6), decltype(m7), decltype(m8), decltype(m9), decltype(m10), decltype(m11), decltype(m12), decltype(m13), decltype(m14), decltype(m15)>{};
            } else {
                return std::tuple<>{};
            }
        }

    } // namespace Detail

    /**
     * @brief C++20 聚合类型编译期反射特征，无需宏即可提取成员数量与类型元组
     */
    template<AggregateStructConcept T>
    struct AggregateTraits {
        using CleanType = std::remove_cvref_t<T>;
        static constexpr size_t memberCount = luisa::compute::detail::count_member<CleanType>();
        using MemberTuple = decltype(Detail::DecomposeAggregateToTuple<CleanType, memberCount>());
        static constexpr size_t byteSize = sizeof(CleanType);
        static constexpr size_t alignment = alignof(CleanType);

        template<size_t Index>
            requires (Index < memberCount)
        using MemberType = std::tuple_element_t<Index, MemberTuple>;
    };

    /**
     * @brief 针对未显式声明 LUISA_STRUCT 的普通 C++20 聚合结构体，在 GPU DSL 中按编译期索引零开销访问字段
     */
    template<size_t Index, AggregateStructConcept S>
    [[nodiscard]] inline auto Field(const luisa::compute::Expr<S>& structExpr) noexcept {
        using MemberT = typename AggregateTraits<S>::template MemberType<Index>;
        return luisa::compute::Expr<MemberT>{
            luisa::compute::detail::FunctionBuilder::current()->member(
                luisa::compute::Type::of<MemberT>(),
                structExpr.expression(),
                Index)};
    }

    template<size_t Index, AggregateStructConcept S>
    [[nodiscard]] inline auto Field(const luisa::compute::Var<S>& structVar) noexcept {
        using MemberT = typename AggregateTraits<S>::template MemberType<Index>;
        return luisa::compute::detail::Ref<MemberT>{
            luisa::compute::detail::FunctionBuilder::current()->member(
                luisa::compute::Type::of<MemberT>(),
                structVar.expression(),
                Index)};
    }

} // namespace FCT::Compute

/**
 * @brief 声明 GPU/CPU 共享结构体并执行编译期布局校验与 SOA/DSL 成员注入
 */
#define FCT_COMPUTE_STRUCT(StructType, ...)                                        \
    static_assert(::FCT::Compute::BufferElementConcept<StructType>,                \
                  #StructType " must be trivially copyable and destructible");     \
    static_assert(alignof(StructType) >= 4u,                                       \
                  #StructType " alignment must be at least 4 bytes for GPU");      \
    LUISA_STRUCT(StructType, __VA_ARGS__)

#endif // FCT_COMPUTE_CONCEPTS_H
