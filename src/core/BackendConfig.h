#ifndef FCT_COMPUTE_CORE_BACKEND_CONFIG_H
#define FCT_COMPUTE_CORE_BACKEND_CONFIG_H

#include "BackendConcepts.h"

#if !defined(FCT_COMPUTE_BACKEND_LUISA)
#define FCT_COMPUTE_BACKEND_LUISA 1
#endif

#if FCT_COMPUTE_BACKEND_LUISA
#include "../backends/luisa/LuisaBackend.h"
namespace FCT::Compute {
    using ActiveBackend = LuisaBackend;
} // namespace FCT::Compute
#else
#error "No valid FCT_Compute backend selected at compile time."
#endif

namespace FCT::Compute {

    static_assert(ComputeBackendConcept<ActiveBackend>,
                  "Selected ActiveBackend does not satisfy ComputeBackendConcept.");

    using PixelStorage = typename ActiveBackend::PixelStorageType;
    using PixelFormat  = typename ActiveBackend::PixelFormatType;

    using Float2   = typename ActiveBackend::Float2;
    using Float3   = typename ActiveBackend::Float3;
    using Float4   = typename ActiveBackend::Float4;
    using Int2     = typename ActiveBackend::Int2;
    using Int3     = typename ActiveBackend::Int3;
    using Int4     = typename ActiveBackend::Int4;
    using UInt2    = typename ActiveBackend::UInt2;
    using UInt3    = typename ActiveBackend::UInt3;
    using UInt4    = typename ActiveBackend::UInt4;
    using Bool2    = typename ActiveBackend::Bool2;
    using Bool3    = typename ActiveBackend::Bool3;
    using Bool4    = typename ActiveBackend::Bool4;
    using Float2x2 = typename ActiveBackend::Float2x2;
    using Float3x3 = typename ActiveBackend::Float3x3;
    using Float4x4 = typename ActiveBackend::Float4x4;

    template<typename T>
    struct HostToBackendMathType {
        using Type = std::remove_cvref_t<T>;
    };

    template<>
    struct HostToBackendMathType<FCT::Vec2> {
        using Type = Float2;
    };

    template<>
    struct HostToBackendMathType<FCT::Vec3> {
        using Type = Float3;
    };

    template<>
    struct HostToBackendMathType<PackedFloat3> {
        using Type = Float3;
    };

    template<>
    struct HostToBackendMathType<FCT::Vec4> {
        using Type = Float4;
    };

    template<typename T>
    struct HostToBackendMathType<FCT::Vector4<T>> {
        using Type = typename ActiveBackend::template Vector<T, 4>;
    };

    template<>
    struct HostToBackendMathType<FCT::Mat3> {
        using Type = Float3x3;
    };

    template<>
    struct HostToBackendMathType<FCT::Mat4> {
        using Type = Float4x4;
    };

    template<typename T>
    using HostToBackendMathType_t = typename HostToBackendMathType<std::remove_cvref_t<T>>::Type;

    [[nodiscard]] constexpr Float2 ToBackendVec(const FCT::Vec2& v) noexcept {
        return ActiveBackend::MathOps::toBackendVec(v);
    }

    [[nodiscard]] constexpr Float3 ToBackendVec(const FCT::Vec3& v) noexcept {
        return ActiveBackend::MathOps::toBackendVec(v);
    }

    [[nodiscard]] constexpr Float4 ToBackendVec(const FCT::Vec4& v) noexcept {
        return ActiveBackend::MathOps::toBackendVec(v);
    }

    [[nodiscard]] inline FCT::Vec2 ToHostVec(const Float2& v) noexcept {
        return ActiveBackend::MathOps::toHostVec(v);
    }

    [[nodiscard]] inline FCT::Vec3 ToHostVec(const Float3& v) noexcept {
        return ActiveBackend::MathOps::toHostVec(v);
    }

    [[nodiscard]] inline FCT::Vec4 ToHostVec(const Float4& v) noexcept {
        return ActiveBackend::MathOps::toHostVec(v);
    }

    [[nodiscard]] constexpr Float4x4 ToBackendMat4(const FCT::Mat4& mat) noexcept {
        return ActiveBackend::MathOps::toBackendMat4(mat);
    }

    [[nodiscard]] inline FCT::Mat4 ToHostMat4(const Float4x4& mat) noexcept {
        return ActiveBackend::MathOps::toHostMat4(mat);
    }

    template<FCT::Format F>
        requires SupportedFormatConcept<F>
    [[nodiscard]] constexpr PixelStorage ToBackendPixelStorage() noexcept {
        return ActiveBackend::FormatOps::template toPixelStorage<F>();
    }

    [[nodiscard]] inline StatusOr<PixelFormat> ToBackendPixelFormat(FCT::Format format) noexcept {
        return ActiveBackend::FormatOps::toPixelFormat(format);
    }

    [[nodiscard]] inline StatusOr<PixelStorage> ToBackendPixelStorage(FCT::Format format) noexcept {
        return ActiveBackend::FormatOps::toPixelStorage(format);
    }

    [[nodiscard]] constexpr FCT::Format FromBackendPixelFormat(PixelFormat format) noexcept {
        return ActiveBackend::FormatOps::fromPixelFormat(format);
    }

} // namespace FCT::Compute

/**
 * @brief 零开销计算结构体反射宏（自动转发至当前编译期后端并执行内存布局静态断言）
 */
#define FCT_COMPUTE_STRUCT(StructType, ...)                                                               \
    static_assert(std::is_standard_layout_v<StructType>, #StructType " must be standard-layout");         \
    static_assert(std::is_trivially_copyable_v<StructType>, #StructType " must be trivially copyable");   \
    static_assert(std::is_trivially_destructible_v<StructType>, #StructType " must be trivially destructible"); \
    FCT_COMPUTE_BACKEND_REFLECT_STRUCT(StructType, __VA_ARGS__)

#endif // FCT_COMPUTE_CORE_BACKEND_CONFIG_H
