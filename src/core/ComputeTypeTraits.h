#ifndef FCT_COMPUTE_CORE_TYPE_TRAITS_H
#define FCT_COMPUTE_CORE_TYPE_TRAITS_H

#include "../thirdparty/thirdparty.h"

namespace FCT::Compute {

    /**
     * @brief 底层硬件加速设备类型枚举（与计算框架后端解耦）
     */
    enum class HardwareBackend : uint8_t {
        Cuda,
        DirectX,
        Vulkan,
        Metal,
        Fallback
    };

    [[nodiscard]] constexpr std::string_view ToHardwareBackendName(HardwareBackend backend) noexcept {
        switch (backend) {
            case HardwareBackend::Cuda:     return "cuda";
            case HardwareBackend::DirectX:  return "dx";
            case HardwareBackend::Vulkan:   return "vk";
            case HardwareBackend::Metal:    return "metal";
            case HardwareBackend::Fallback: return "fallback";
        }
        return "cuda";
    }

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

    template<typename From, typename To>
    inline constexpr bool IsLayoutIdentical_v =
        (sizeof(From) == sizeof(To)) &&
        (alignof(From) == alignof(To)) &&
        std::is_standard_layout_v<From> &&
        std::is_standard_layout_v<To>;

    static_assert(IsLayoutIdentical_v<FCT::Vec2, StorageType_t<FCT::Vec2>>);
    static_assert(IsLayoutIdentical_v<FCT::Vec3, StorageType_t<FCT::Vec3>>);
    static_assert(IsLayoutIdentical_v<FCT::Vec4, StorageType_t<FCT::Vec4>>);

    /**
     * @brief 与后端无关的编译期像素通道存储精度分类
     */
    enum class PixelChannelKind : uint8_t {
        Invalid = 0,
        Int8,
        UInt8,
        Norm8,
        Int16,
        UInt16,
        Norm16,
        Int32,
        UInt32,
        Float16,
        Float32
    };

    template<FCT::Format F>
    struct FormatTraits {
        static constexpr bool supported = false;
    };

#define FCT_COMPUTE_DEFINE_FORMAT_TRAIT(FctFmt, ElemT, Kind, Channels, ByteSz) \
    template<>                                                                  \
    struct FormatTraits<FCT::Format::FctFmt> {                                  \
        static constexpr bool supported = true;                                 \
        using ElementType = ElemT;                                              \
        static constexpr PixelChannelKind channelKind = PixelChannelKind::Kind; \
        static constexpr size_t channels = Channels;                            \
        static constexpr size_t byteSize = ByteSz;                              \
    }

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8_UNORM,           float,    Norm8,   1, 1);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8_UINT,            uint32_t, UInt8,   1, 1);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8_SINT,            int32_t,  Int8,    1, 1);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8_UNORM,         float,    Norm8,   2, 2);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8_UINT,          uint32_t, UInt8,   2, 2);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8_SINT,          int32_t,  Int8,    2, 2);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8B8A8_UNORM,     float,    Norm8,   4, 4);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8B8A8_UINT,      uint32_t, UInt8,   4, 4);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8B8A8_SINT,      int32_t,  Int8,    4, 4);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R8G8B8A8_SRGB,      float,    Norm8,   4, 4);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(B8G8R8A8_UNORM,     float,    Norm8,   4, 4);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(B8G8R8A8_SRGB,      float,    Norm8,   4, 4);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16_UINT,           uint32_t, UInt16,  1, 2);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16_SINT,           int32_t,  Int16,   1, 2);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16_SFLOAT,         float,    Float16, 1, 2);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16_UINT,        uint32_t, UInt16,  2, 4);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16_SINT,        int32_t,  Int16,   2, 4);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16_SFLOAT,      float,    Float16, 2, 4);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16B16A16_UINT,   uint32_t, UInt16,  4, 8);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16B16A16_SINT,   int32_t,  Int16,   4, 8);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R16G16B16A16_SFLOAT, float,    Float16, 4, 8);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32_UINT,           uint32_t, UInt32,  1, 4);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32_SINT,           int32_t,  Int32,   1, 4);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32_SFLOAT,         float,    Float32, 1, 4);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32_UINT,        uint32_t, UInt32,  2, 8);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32_SINT,        int32_t,  Int32,   2, 8);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32_SFLOAT,      float,    Float32, 2, 8);

    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32B32A32_UINT,   uint32_t, UInt32,  4, 16);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32B32A32_SINT,   int32_t,  Int32,   4, 16);
    FCT_COMPUTE_DEFINE_FORMAT_TRAIT(R32G32B32A32_SFLOAT, float,    Float32, 4, 16);

#undef FCT_COMPUTE_DEFINE_FORMAT_TRAIT

} // namespace FCT::Compute

#endif // FCT_COMPUTE_CORE_TYPE_TRAITS_H
