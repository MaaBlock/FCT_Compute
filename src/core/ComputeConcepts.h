#ifndef FCT_COMPUTE_CORE_CONCEPTS_H
#define FCT_COMPUTE_CORE_CONCEPTS_H

#include "ComputeTypeTraits.h"

namespace FCT::Compute {

    template<typename T>
    concept ScalarConcept =
        std::same_as<std::remove_cvref_t<T>, bool> ||
        std::same_as<std::remove_cvref_t<T>, int16_t> ||
        std::same_as<std::remove_cvref_t<T>, uint16_t> ||
        std::same_as<std::remove_cvref_t<T>, int32_t> ||
        std::same_as<std::remove_cvref_t<T>, uint32_t> ||
        std::same_as<std::remove_cvref_t<T>, int64_t> ||
        std::same_as<std::remove_cvref_t<T>, uint64_t> ||
        std::same_as<std::remove_cvref_t<T>, float> ||
        std::same_as<std::remove_cvref_t<T>, double>;

    template<typename T>
    concept FCTVectorConcept =
        std::same_as<std::remove_cvref_t<T>, FCT::Vec2> ||
        std::same_as<std::remove_cvref_t<T>, FCT::Vec3> ||
        std::same_as<std::remove_cvref_t<T>, PackedFloat3> ||
        std::same_as<std::remove_cvref_t<T>, FCT::Vec4> ||
        std::same_as<std::remove_cvref_t<T>, FCT::Vector4<int32_t>> ||
        std::same_as<std::remove_cvref_t<T>, FCT::Vector4<uint32_t>>;

    template<typename T>
    concept BufferElementConcept =
        std::is_standard_layout_v<StorageType_t<T>> &&
        std::is_trivially_copyable_v<StorageType_t<T>> &&
        std::is_trivially_destructible_v<StorageType_t<T>> &&
        !std::is_pointer_v<StorageType_t<T>>;

    template<typename T>
    concept ImageElementConcept =
        std::same_as<T, float> ||
        std::same_as<T, int32_t> ||
        std::same_as<T, uint32_t>;

    template<FCT::Format F>
    concept SupportedFormatConcept = FormatTraits<F>::supported;

    template<size_t Dim>
    concept KernelDimensionConcept = (Dim == 1u || Dim == 2u || Dim == 3u);

    template<typename T>
    concept AggregateStructConcept =
        std::is_aggregate_v<std::remove_cvref_t<T>> &&
        std::is_standard_layout_v<std::remove_cvref_t<T>> &&
        std::is_trivially_copyable_v<std::remove_cvref_t<T>> &&
        !std::is_array_v<std::remove_cvref_t<T>>;

    namespace Detail {

        struct UniversalFieldProbe {
            template<typename T>
            constexpr operator T() const noexcept;
        };

        template<typename T, typename IndexSeq, typename = void>
        struct CanConstructWithNFields : std::false_type {};

        template<typename T, size_t... Is>
        struct CanConstructWithNFields<
            T,
            std::index_sequence<Is...>,
            std::void_t<decltype(T{(static_cast<void>(Is), UniversalFieldProbe{})...})>> : std::true_type {};

        template<typename T, size_t N = 16>
        constexpr size_t DetectAggregateFieldCount() noexcept {
            if constexpr (N == 0) {
                return 0;
            } else if constexpr (CanConstructWithNFields<T, std::make_index_sequence<N>>::value) {
                return N;
            } else {
                return DetectAggregateFieldCount<T, N - 1>();
            }
        }

        template<size_t Count>
        struct StructuredBindingExtractor;

        template<>
        struct StructuredBindingExtractor<1> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0] = std::forward<T>(obj);
                return std::forward_as_tuple(f0);
            }
        };

        template<>
        struct StructuredBindingExtractor<2> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1);
            }
        };

        template<>
        struct StructuredBindingExtractor<3> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1, f2] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1, f2);
            }
        };

        template<>
        struct StructuredBindingExtractor<4> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1, f2, f3] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1, f2, f3);
            }
        };

        template<>
        struct StructuredBindingExtractor<5> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1, f2, f3, f4] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1, f2, f3, f4);
            }
        };

        template<>
        struct StructuredBindingExtractor<6> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1, f2, f3, f4, f5] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1, f2, f3, f4, f5);
            }
        };

        template<>
        struct StructuredBindingExtractor<7> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1, f2, f3, f4, f5, f6] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1, f2, f3, f4, f5, f6);
            }
        };

        template<>
        struct StructuredBindingExtractor<8> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1, f2, f3, f4, f5, f6, f7] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1, f2, f3, f4, f5, f6, f7);
            }
        };

        template<>
        struct StructuredBindingExtractor<9> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1, f2, f3, f4, f5, f6, f7, f8] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1, f2, f3, f4, f5, f6, f7, f8);
            }
        };

        template<>
        struct StructuredBindingExtractor<10> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9);
            }
        };

        template<>
        struct StructuredBindingExtractor<11> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10);
            }
        };

        template<>
        struct StructuredBindingExtractor<12> {
            template<typename T>
            static constexpr auto toTuple(T&& obj) noexcept {
                auto&& [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11] = std::forward<T>(obj);
                return std::forward_as_tuple(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11);
            }
        };

    } // namespace Detail

    /**
     * @brief 基于 C++20 结构化绑定的编译期聚合结构体反射器（与底层计算后端无关）
     */
    template<AggregateStructConcept T>
    struct AggregateTraits {
        using RawType = std::remove_cvref_t<T>;
        static constexpr size_t memberCount = Detail::DetectAggregateFieldCount<RawType>();
        static_assert(memberCount >= 1 && memberCount <= 12, "AggregateTraits supports structs with 1 to 12 fields.");

        [[nodiscard]] static constexpr auto tieMembers(RawType& obj) noexcept {
            return Detail::StructuredBindingExtractor<memberCount>::toTuple(obj);
        }

        [[nodiscard]] static constexpr auto tieMembers(const RawType& obj) noexcept {
            return Detail::StructuredBindingExtractor<memberCount>::toTuple(obj);
        }

        using MemberRefTuple = decltype(tieMembers(std::declval<RawType&>()));

        template<size_t Index>
            requires (Index < memberCount)
        using MemberType = std::remove_cvref_t<std::tuple_element_t<Index, MemberRefTuple>>;

        template<size_t Index>
            requires (Index < memberCount)
        [[nodiscard]] static constexpr decltype(auto) get(RawType& obj) noexcept {
            return std::get<Index>(tieMembers(obj));
        }

        template<size_t Index>
            requires (Index < memberCount)
        [[nodiscard]] static constexpr decltype(auto) get(const RawType& obj) noexcept {
            return std::get<Index>(tieMembers(obj));
        }
    };

} // namespace FCT::Compute

#endif // FCT_COMPUTE_CORE_CONCEPTS_H
