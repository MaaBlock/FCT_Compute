#ifndef FCT_COMPUTE_TYPE_TRAITS_H
#define FCT_COMPUTE_TYPE_TRAITS_H

#include "thirdparty/thirdparty.h"

namespace FCT::Compute {

    /**
     * @brief 与 FCT::Vec3 内存布局完全一致的平凡可拷贝 12 字节紧凑三维向量
     * Note: FCT::Vec3 自定义了拷贝构造函数导致 is_trivially_copyable 为 false，
     * 通过编译期 StorageType 映射为 PackedFloat3，可在不破坏 ABI 的前提下实现零拷贝 Buffer 传输与 GPU 端成员反射。
     */
    struct PackedFloat3 {
        float x;
        float y;
        float z;

        constexpr PackedFloat3() noexcept : x(0.0f), y(0.0f), z(0.0f) {}
        constexpr PackedFloat3(float inX, float inY, float inZ) noexcept : x(inX), y(inY), z(inZ) {}
        constexpr PackedFloat3(const FCT::Vec3& v) noexcept : x(v.x), y(v.y), z(v.z) {}

        [[nodiscard]] inline operator FCT::Vec3() const noexcept {
            return FCT::Vec3(x, y, z);
        }
    };

    static_assert(sizeof(PackedFloat3) == sizeof(FCT::Vec3), "PackedFloat3 must match FCT::Vec3 size (12 bytes)");
    static_assert(alignof(PackedFloat3) == alignof(FCT::Vec3), "PackedFloat3 must match FCT::Vec3 alignment (4 bytes)");
    static_assert(std::is_standard_layout_v<PackedFloat3> && std::is_standard_layout_v<FCT::Vec3>);
    static_assert(std::is_trivially_copyable_v<PackedFloat3> && std::is_trivially_destructible_v<PackedFloat3>);

} // namespace FCT::Compute

LUISA_STRUCT(FCT::Vec2, x, y) {
    [[nodiscard]] auto to_vec() const noexcept {
        return luisa::compute::make_float2(this->x, this->y);
    }
    void from_vec(luisa::compute::Expr<luisa::float2> v) noexcept {
        this->x = v.x;
        this->y = v.y;
    }
};

LUISA_STRUCT(FCT::Compute::PackedFloat3, x, y, z) {
    [[nodiscard]] auto to_vec() const noexcept {
        return luisa::compute::make_float3(this->x, this->y, this->z);
    }
    void from_vec(luisa::compute::Expr<luisa::float3> v) noexcept {
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
    }
};

LUISA_STRUCT(FCT::Vec4, x, y, z, w) {
    [[nodiscard]] auto to_vec() const noexcept {
        return luisa::compute::make_float4(this->x, this->y, this->z, this->w);
    }
    void from_vec(luisa::compute::Expr<luisa::float4> v) noexcept {
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
        this->w = v.w;
    }
};

namespace FCT::Compute {

    template<typename T>
    struct StorageType {
        using Type = std::remove_cvref_t<T>;
    };

    template<>
    struct StorageType<FCT::Vec3> {
        using Type = PackedFloat3;
    };

    template<typename T>
    using StorageType_t = typename StorageType<std::remove_cvref_t<T>>::Type;

    template<typename T>
    struct HostToLuisaMathType {
        using Type = std::remove_cvref_t<T>;
    };

    template<>
    struct HostToLuisaMathType<FCT::Vec2> {
        using Type = luisa::float2;
    };

    template<>
    struct HostToLuisaMathType<FCT::Vec3> {
        using Type = luisa::float3;
    };

    template<>
    struct HostToLuisaMathType<PackedFloat3> {
        using Type = luisa::float3;
    };

    template<>
    struct HostToLuisaMathType<FCT::Vec4> {
        using Type = luisa::float4;
    };

    template<typename T>
    struct HostToLuisaMathType<FCT::Vector4<T>> {
        using Type = luisa::Vector<T, 4>;
    };

    template<>
    struct HostToLuisaMathType<FCT::Mat3> {
        using Type = luisa::float3x3;
    };

    template<>
    struct HostToLuisaMathType<FCT::Mat4> {
        using Type = luisa::float4x4;
    };

    template<typename T>
    using HostToLuisaMathType_t = typename HostToLuisaMathType<std::remove_cvref_t<T>>::Type;

    template<typename From, typename To>
    inline constexpr bool IsLayoutIdentical_v =
        (sizeof(From) == sizeof(To)) &&
        (alignof(From) == alignof(To)) &&
        std::is_standard_layout_v<From> &&
        std::is_standard_layout_v<To>;

    static_assert(IsLayoutIdentical_v<FCT::Vec2, StorageType_t<FCT::Vec2>>);
    static_assert(IsLayoutIdentical_v<FCT::Vec3, StorageType_t<FCT::Vec3>>);
    static_assert(IsLayoutIdentical_v<FCT::Vec4, StorageType_t<FCT::Vec4>>);

    [[nodiscard]] constexpr luisa::float2 ToLuisaVec(const FCT::Vec2& v) noexcept {
        return luisa::make_float2(v.x, v.y);
    }

    [[nodiscard]] constexpr luisa::float3 ToLuisaVec(const FCT::Vec3& v) noexcept {
        return luisa::make_float3(v.x, v.y, v.z);
    }

    [[nodiscard]] constexpr luisa::float4 ToLuisaVec(const FCT::Vec4& v) noexcept {
        return luisa::make_float4(v.x, v.y, v.z, v.w);
    }

    [[nodiscard]] inline FCT::Vec2 ToHostVec(const luisa::float2& v) noexcept {
        return FCT::Vec2(v.x, v.y);
    }

    [[nodiscard]] inline FCT::Vec3 ToHostVec(const luisa::float3& v) noexcept {
        return FCT::Vec3(v.x, v.y, v.z);
    }

    [[nodiscard]] inline FCT::Vec4 ToHostVec(const luisa::float4& v) noexcept {
        return FCT::Vec4(v.x, v.y, v.z, v.w);
    }

    /**
     * @brief 将 FCT 行主序 Mat4 转换为 Luisa 列主序 float4x4
     */
    [[nodiscard]] constexpr luisa::float4x4 ToLuisaMat4(const FCT::Mat4& mat) noexcept {
        return luisa::make_float4x4(
            luisa::make_float4(mat.m[0], mat.m[4], mat.m[8],  mat.m[12]),
            luisa::make_float4(mat.m[1], mat.m[5], mat.m[9],  mat.m[13]),
            luisa::make_float4(mat.m[2], mat.m[6], mat.m[10], mat.m[14]),
            luisa::make_float4(mat.m[3], mat.m[7], mat.m[11], mat.m[15]));
    }

    /**
     * @brief 将 Luisa 列主序 float4x4 转换为 FCT 行主序 Mat4
     */
    [[nodiscard]] inline FCT::Mat4 ToHostMat4(const luisa::float4x4& mat) noexcept {
        return FCT::Mat4(
            mat[0].x, mat[1].x, mat[2].x, mat[3].x,
            mat[0].y, mat[1].y, mat[2].y, mat[3].y,
            mat[0].z, mat[1].z, mat[2].z, mat[3].z,
            mat[0].w, mat[1].w, mat[2].w, mat[3].w);
    }

    /**
     * @brief 编译期 FCT::Format 到 Luisa PixelStorage / PixelFormat / Image 元素类型的元编程映射
     */
    template<FCT::Format F>
    struct FormatTraits {
        static constexpr bool isSupported = false;
        using ElementType = void;
        static constexpr uint32_t channels = 0u;
    };

#define FCT_COMPUTE_DEFINE_FORMAT_TRAIT(FctFmt, LuisaStorage, LuisaFmt, ElemType, Channels) \
    template<>                                                                              \
    struct FormatTraits<FCT::Format::FctFmt> {                                              \
        static constexpr bool isSupported = true;                                           \
        static constexpr luisa::compute::PixelStorage storage =                             \
            luisa::compute::PixelStorage::LuisaStorage;                                     \
        static constexpr luisa::compute::PixelFormat pixelFormat =                          \
            luisa::compute::PixelFormat::LuisaFmt;                                          \
        using ElementType = ElemType;                                                       \
        static constexpr uint32_t channels = Channels;                                      \
        static constexpr size_t byteSize = FCT::FormatSize(FCT::Format::FctFmt);            \
    }

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8_UNORM,           BYTE1,      R8UNorm,    float,    1u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8_UNORM,         BYTE2,      RG8UNorm,   float,    2u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8B8A8_UNORM,     BYTE4,      RGBA8UNorm, float,    4u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8B8A8_SRGB,      BYTE4_SRGB, RGBA8SRGB,  float,    4u);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16_SFLOAT,         HALF1,      R16F,       float,    1u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16_SFLOAT,      HALF2,      RG16F,      float,    2u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16B16A16_SFLOAT,HALF4,      RGBA16F,    float,    4u);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32_SFLOAT,         FLOAT1,     R32F,       float,    1u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32_SFLOAT,      FLOAT2,     RG32F,      float,    2u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32B32A32_SFLOAT,FLOAT4,     RGBA32F,    float,    4u);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8_SINT,            BYTE1,      R8SInt,     int32_t,  1u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8_SINT,          BYTE2,      RG8SInt,    int32_t,  2u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8B8A8_SINT,      BYTE4,      RGBA8SInt,  int32_t,  4u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16_SINT,           SHORT1,     R16SInt,    int32_t,  1u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16_SINT,        SHORT2,     RG16SInt,   int32_t,  2u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16B16A16_SINT,  SHORT4,     RGBA16SInt, int32_t,  4u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32_SINT,           INT1,       R32SInt,    int32_t,  1u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32_SINT,        INT2,       RG32SInt,   int32_t,  2u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32B32A32_SINT,  INT4,       RGBA32SInt, int32_t,  4u);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8_UINT,            BYTE1,      R8UInt,     uint32_t, 1u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8_UINT,          BYTE2,      RG8UInt,    uint32_t, 2u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8B8A8_UINT,      BYTE4,      RGBA8UInt,  uint32_t, 4u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16_UINT,           SHORT1,     R16UInt,    uint32_t, 1u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16_UINT,        SHORT2,     RG16UInt,   uint32_t, 2u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16B16A16_UINT,  SHORT4,     RGBA16UInt, uint32_t, 4u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32_UINT,           INT1,       R32UInt,    uint32_t, 1u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32_UINT,        INT2,       RG32UInt,   uint32_t, 2u);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32B32A32_UINT,  INT4,       RGBA32UInt, uint32_t, 4u);

#undef FCT_COMPUTE_DEFINE_FORMAT_TRAIT

    [[nodiscard]] inline StatusOr<luisa::compute::PixelFormat> ToLuisaPixelFormat(FCT::Format format) noexcept {
        using luisa::compute::PixelFormat;
        switch (format) {
            case FCT::Format::R8_UNORM:            return PixelFormat::R8UNorm;
            case FCT::Format::R8G8_UNORM:          return PixelFormat::RG8UNorm;
            case FCT::Format::R8G8B8A8_UNORM:      return PixelFormat::RGBA8UNorm;
            case FCT::Format::R8G8B8A8_SRGB:       return PixelFormat::RGBA8SRGB;
            case FCT::Format::R16_SFLOAT:          return PixelFormat::R16F;
            case FCT::Format::R16G16_SFLOAT:       return PixelFormat::RG16F;
            case FCT::Format::R16G16B16A16_SFLOAT: return PixelFormat::RGBA16F;
            case FCT::Format::R32_SFLOAT:          return PixelFormat::R32F;
            case FCT::Format::R32G32_SFLOAT:       return PixelFormat::RG32F;
            case FCT::Format::R32G32B32A32_SFLOAT: return PixelFormat::RGBA32F;
            case FCT::Format::R8_SINT:             return PixelFormat::R8SInt;
            case FCT::Format::R8G8_SINT:           return PixelFormat::RG8SInt;
            case FCT::Format::R8G8B8A8_SINT:       return PixelFormat::RGBA8SInt;
            case FCT::Format::R16_SINT:            return PixelFormat::R16SInt;
            case FCT::Format::R16G16_SINT:         return PixelFormat::RG16SInt;
            case FCT::Format::R16G16B16A16_SINT:   return PixelFormat::RGBA16SInt;
            case FCT::Format::R32_SINT:            return PixelFormat::R32SInt;
            case FCT::Format::R32G32_SINT:         return PixelFormat::RG32SInt;
            case FCT::Format::R32G32B32A32_SINT:   return PixelFormat::RGBA32SInt;
            case FCT::Format::R8_UINT:             return PixelFormat::R8UInt;
            case FCT::Format::R8G8_UINT:           return PixelFormat::RG8UInt;
            case FCT::Format::R8G8B8A8_UINT:       return PixelFormat::RGBA8UInt;
            case FCT::Format::R16_UINT:            return PixelFormat::R16UInt;
            case FCT::Format::R16G16_UINT:         return PixelFormat::RG16UInt;
            case FCT::Format::R16G16B16A16_UINT:   return PixelFormat::RGBA16UInt;
            case FCT::Format::R32_UINT:            return PixelFormat::R32UInt;
            case FCT::Format::R32G32_UINT:         return PixelFormat::RG32UInt;
            case FCT::Format::R32G32B32A32_UINT:   return PixelFormat::RGBA32UInt;
            default:
                return InvalidArgumentError(StrCat("Unsupported FCT::Format for LuisaCompute: ", FCT::FormatToString(format)));
        }
    }

    [[nodiscard]] inline StatusOr<luisa::compute::PixelStorage> ToLuisaPixelStorage(FCT::Format format) noexcept {
        auto pixelFormat = ToLuisaPixelFormat(format);
        FCT_CHECK_STATUS(pixelFormat);
        return luisa::compute::pixel_format_to_storage(*pixelFormat);
    }

    [[nodiscard]] constexpr FCT::Format FromLuisaPixelFormat(luisa::compute::PixelFormat format) noexcept {
        using luisa::compute::PixelFormat;
        switch (format) {
            case PixelFormat::R8UNorm:    return FCT::Format::R8_UNORM;
            case PixelFormat::RG8UNorm:   return FCT::Format::R8G8_UNORM;
            case PixelFormat::RGBA8UNorm: return FCT::Format::R8G8B8A8_UNORM;
            case PixelFormat::RGBA8SRGB:  return FCT::Format::R8G8B8A8_SRGB;
            case PixelFormat::R16F:       return FCT::Format::R16_SFLOAT;
            case PixelFormat::RG16F:      return FCT::Format::R16G16_SFLOAT;
            case PixelFormat::RGBA16F:    return FCT::Format::R16G16B16A16_SFLOAT;
            case PixelFormat::R32F:       return FCT::Format::R32_SFLOAT;
            case PixelFormat::RG32F:      return FCT::Format::R32G32_SFLOAT;
            case PixelFormat::RGBA32F:    return FCT::Format::R32G32B32A32_SFLOAT;
            case PixelFormat::R8SInt:     return FCT::Format::R8_SINT;
            case PixelFormat::RG8SInt:    return FCT::Format::R8G8_SINT;
            case PixelFormat::RGBA8SInt:  return FCT::Format::R8G8B8A8_SINT;
            case PixelFormat::R16SInt:    return FCT::Format::R16_SINT;
            case PixelFormat::RG16SInt:   return FCT::Format::R16G16_SINT;
            case PixelFormat::RGBA16SInt: return FCT::Format::R16G16B16A16_SINT;
            case PixelFormat::R32SInt:    return FCT::Format::R32_SINT;
            case PixelFormat::RG32SInt:   return FCT::Format::R32G32_SINT;
            case PixelFormat::RGBA32SInt: return FCT::Format::R32G32B32A32_SINT;
            case PixelFormat::R8UInt:     return FCT::Format::R8_UINT;
            case PixelFormat::RG8UInt:    return FCT::Format::R8G8_UINT;
            case PixelFormat::RGBA8UInt:  return FCT::Format::R8G8B8A8_UINT;
            case PixelFormat::R16UInt:    return FCT::Format::R16_UINT;
            case PixelFormat::RG16UInt:   return FCT::Format::R16G16_UINT;
            case PixelFormat::RGBA16UInt: return FCT::Format::R16G16B16A16_UINT;
            case PixelFormat::R32UInt:    return FCT::Format::R32_UINT;
            case PixelFormat::RG32UInt:   return FCT::Format::R32G32_UINT;
            case PixelFormat::RGBA32UInt: return FCT::Format::R32G32B32A32_UINT;
            default:                      return FCT::Format::UNDEFINED;
        }
    }

} // namespace FCT::Compute

#endif // FCT_COMPUTE_TYPE_TRAITS_H
