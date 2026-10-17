#ifndef FCT_COMPUTE_IMAGE_H
#define FCT_COMPUTE_IMAGE_H

#include "ComputeConcepts.h"

namespace FCT::Compute {

    template<ImageElementConcept T>
    class Image;

    /**
     * @brief 零开销 2D 纹理视图封装，尺寸与对齐同 luisa::compute::ImageView 完全一致
     */
    template<ImageElementConcept T>
    class ImageView {
    public:
        using ElementType = T;
        using NativeViewType = luisa::compute::ImageView<T>;

        /* implicit */ ImageView(NativeViewType nativeView) noexcept
            : m_view(nativeView) {}

        [[nodiscard]] operator NativeViewType() const noexcept {
            return m_view;
        }

        [[nodiscard]] const NativeViewType& raw() const noexcept {
            return m_view;
        }

        [[nodiscard]] NativeViewType& raw() noexcept {
            return m_view;
        }

        [[nodiscard]] uint64_t handle() const noexcept {
            return m_view.handle();
        }

        [[nodiscard]] void* nativeHandle() const noexcept {
            return m_view.native_handle();
        }

        [[nodiscard]] luisa::uint2 size() const noexcept {
            return m_view.size();
        }

        [[nodiscard]] uint32_t width() const noexcept {
            return m_view.size().x;
        }

        [[nodiscard]] uint32_t height() const noexcept {
            return m_view.size().y;
        }

        [[nodiscard]] size_t sizeBytes() const noexcept {
            return m_view.size_bytes();
        }

        [[nodiscard]] luisa::compute::PixelStorage storage() const noexcept {
            return m_view.storage();
        }

        [[nodiscard]] luisa::compute::PixelFormat pixelFormat() const noexcept {
            return m_view.format();
        }

        [[nodiscard]] FCT::Format fctFormat() const noexcept {
            return FromLuisaPixelFormat(m_view.format());
        }

        [[nodiscard]] uint32_t level() const noexcept {
            return m_view.level();
        }

        template<typename U>
        [[nodiscard]] auto copyFrom(U&& src) const noexcept {
            return m_view.copy_from(std::forward<U>(src));
        }

        template<typename U>
        [[nodiscard]] auto copyTo(U&& dst) const noexcept {
            return m_view.copy_to(std::forward<U>(dst));
        }

        [[nodiscard]] auto operator->() const noexcept {
            return m_view.operator->();
        }

    private:
        NativeViewType m_view;
    };

    /**
     * @brief 零开销 RAII 2D 纹理封装，尺寸与对齐同 luisa::compute::Image 完全一致
     */
    template<ImageElementConcept T>
    class Image {
    public:
        using ElementType = T;
        using NativeImageType = luisa::compute::Image<T>;
        using ViewType = ImageView<T>;

        Image() noexcept = default;

        explicit Image(NativeImageType&& nativeImage) noexcept
            : m_image(std::move(nativeImage)) {}

        Image(Image&&) noexcept = default;
        Image& operator=(Image&&) noexcept = default;
        Image(const Image&) = delete;
        Image& operator=(const Image&) = delete;
        ~Image() noexcept = default;

        [[nodiscard]] explicit operator bool() const noexcept {
            return static_cast<bool>(m_image);
        }

        [[nodiscard]] operator const NativeImageType&() const noexcept {
            return m_image;
        }

        [[nodiscard]] operator typename ViewType::NativeViewType() const noexcept {
            return m_image.view();
        }

        [[nodiscard]] operator ViewType() const noexcept {
            return ViewType(m_image.view());
        }

        [[nodiscard]] const NativeImageType& raw() const noexcept {
            return m_image;
        }

        [[nodiscard]] NativeImageType& raw() noexcept {
            return m_image;
        }

        [[nodiscard]] uint64_t handle() const noexcept {
            return m_image.handle();
        }

        [[nodiscard]] void* nativeHandle() const noexcept {
            return m_image.native_handle();
        }

        [[nodiscard]] luisa::uint2 size() const noexcept {
            return m_image.size();
        }

        [[nodiscard]] uint32_t width() const noexcept {
            return m_image.size().x;
        }

        [[nodiscard]] uint32_t height() const noexcept {
            return m_image.size().y;
        }

        [[nodiscard]] uint32_t mipLevels() const noexcept {
            return m_image.mip_levels();
        }

        [[nodiscard]] luisa::compute::PixelStorage storage() const noexcept {
            return m_image.storage();
        }

        [[nodiscard]] luisa::compute::PixelFormat pixelFormat() const noexcept {
            return m_image.format();
        }

        [[nodiscard]] FCT::Format fctFormat() const noexcept {
            return FromLuisaPixelFormat(m_image.format());
        }

        [[nodiscard]] ViewType view(uint32_t level = 0u) const noexcept {
            return ViewType(m_image.view(level));
        }

        template<typename U>
        [[nodiscard]] auto copyFrom(U&& src) const noexcept {
            return m_image.copy_from(std::forward<U>(src));
        }

        template<typename U>
        [[nodiscard]] auto copyTo(U&& dst) const noexcept {
            return m_image.copy_to(std::forward<U>(dst));
        }

        [[nodiscard]] auto operator->() const noexcept {
            return m_image.operator->();
        }

    private:
        NativeImageType m_image;
    };

    /**
     * @brief 基于 FCT::Format 编译期推导元素类型的 2D 纹理类型别名
     */
    template<FCT::Format F>
        requires SupportedFormatConcept<F>
    using FormatImage = Image<typename FormatTraits<F>::ElementType>;

    /**
     * @brief 零开销 3D 体纹理视图封装
     */
    template<ImageElementConcept T>
    class VolumeView {
    public:
        using ElementType = T;
        using NativeViewType = luisa::compute::VolumeView<T>;

        /* implicit */ VolumeView(NativeViewType nativeView) noexcept
            : m_view(nativeView) {}

        [[nodiscard]] operator NativeViewType() const noexcept {
            return m_view;
        }

        [[nodiscard]] const NativeViewType& raw() const noexcept {
            return m_view;
        }

        [[nodiscard]] uint64_t handle() const noexcept {
            return m_view.handle();
        }

        [[nodiscard]] luisa::uint3 size() const noexcept {
            return m_view.size();
        }

        [[nodiscard]] size_t sizeBytes() const noexcept {
            return m_view.size_bytes();
        }

        template<typename U>
        [[nodiscard]] auto copyFrom(U&& src) const noexcept {
            return m_view.copy_from(std::forward<U>(src));
        }

        template<typename U>
        [[nodiscard]] auto copyTo(U&& dst) const noexcept {
            return m_view.copy_to(std::forward<U>(dst));
        }

        [[nodiscard]] auto operator->() const noexcept {
            return m_view.operator->();
        }

    private:
        NativeViewType m_view;
    };

    /**
     * @brief 零开销 RAII 3D 体纹理封装
     */
    template<ImageElementConcept T>
    class Volume {
    public:
        using ElementType = T;
        using NativeVolumeType = luisa::compute::Volume<T>;
        using ViewType = VolumeView<T>;

        Volume() noexcept = default;

        explicit Volume(NativeVolumeType&& nativeVolume) noexcept
            : m_volume(std::move(nativeVolume)) {}

        Volume(Volume&&) noexcept = default;
        Volume& operator=(Volume&&) noexcept = default;
        Volume(const Volume&) = delete;
        Volume& operator=(const Volume&) = delete;
        ~Volume() noexcept = default;

        [[nodiscard]] explicit operator bool() const noexcept {
            return static_cast<bool>(m_volume);
        }

        [[nodiscard]] operator const NativeVolumeType&() const noexcept {
            return m_volume;
        }

        [[nodiscard]] operator typename ViewType::NativeViewType() const noexcept {
            return m_volume.view();
        }

        [[nodiscard]] operator ViewType() const noexcept {
            return ViewType(m_volume.view());
        }

        [[nodiscard]] const NativeVolumeType& raw() const noexcept {
            return m_volume;
        }

        [[nodiscard]] NativeVolumeType& raw() noexcept {
            return m_volume;
        }

        [[nodiscard]] luisa::uint3 size() const noexcept {
            return m_volume.size();
        }

        [[nodiscard]] ViewType view(uint32_t level = 0u) const noexcept {
            return ViewType(m_volume.view(level));
        }

        template<typename U>
        [[nodiscard]] auto copyFrom(U&& src) const noexcept {
            return m_volume.copy_from(std::forward<U>(src));
        }

        template<typename U>
        [[nodiscard]] auto copyTo(U&& dst) const noexcept {
            return m_volume.copy_to(std::forward<U>(dst));
        }

        [[nodiscard]] auto operator->() const noexcept {
            return m_volume.operator->();
        }

    private:
        NativeVolumeType m_volume;
    };

    static_assert(sizeof(Image<float>) == sizeof(luisa::compute::Image<float>),
                  "Zero-overhead guarantee: Image<T> must not add any bytes over luisa::compute::Image");
    static_assert(sizeof(ImageView<float>) == sizeof(luisa::compute::ImageView<float>),
                  "Zero-overhead guarantee: ImageView<T> must not add any bytes over luisa::compute::ImageView");
    static_assert(sizeof(Volume<float>) == sizeof(luisa::compute::Volume<float>),
                  "Zero-overhead guarantee: Volume<T> must not add any bytes over luisa::compute::Volume");
    static_assert(sizeof(VolumeView<float>) == sizeof(luisa::compute::VolumeView<float>),
                  "Zero-overhead guarantee: VolumeView<T> must not add any bytes over luisa::compute::VolumeView");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_IMAGE_H
