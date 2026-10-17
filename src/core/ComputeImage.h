#ifndef FCT_COMPUTE_CORE_IMAGE_H
#define FCT_COMPUTE_CORE_IMAGE_H

#include "BackendConfig.h"

namespace FCT::Compute {

    /**
     * @brief 零开销 2D 纹理视图模板（与编译期后端策略解耦）
     */
    template<ImageElementConcept T, ComputeBackendConcept Backend = ActiveBackend>
    class BasicImageView {
    public:
        using ElementType    = T;
        using BackendType    = Backend;
        using NativeViewType = typename Backend::template ImageViewImpl<T>;

        constexpr BasicImageView() noexcept = default;

        constexpr explicit BasicImageView(NativeViewType view) noexcept
            : m_view(view) {}

        [[nodiscard]] inline bool valid() const noexcept {
            return Backend::ImageOps::isViewValid(m_view);
        }

        [[nodiscard]] inline explicit operator bool() const noexcept {
            return valid();
        }

        [[nodiscard]] inline uint32_t width() const noexcept {
            return Backend::ImageOps::viewWidth(m_view);
        }

        [[nodiscard]] inline uint32_t height() const noexcept {
            return Backend::ImageOps::viewHeight(m_view);
        }

        [[nodiscard]] inline uint32_t level() const noexcept {
            return Backend::ImageOps::viewLevel(m_view);
        }

        [[nodiscard]] inline size_t byteSize() const noexcept {
            return Backend::ImageOps::viewByteSize(m_view);
        }

        [[nodiscard]] inline auto copyFromHost(const void* hostPixels) const noexcept {
            return Backend::ImageOps::copyFromHost(m_view, hostPixels);
        }

        [[nodiscard]] inline auto copyToHost(void* hostPixels) const noexcept {
            return Backend::ImageOps::copyToHost(m_view, hostPixels);
        }

        [[nodiscard]] inline auto copyFromImage(BasicImageView<T, Backend> srcView) const noexcept {
            return Backend::ImageOps::copyFromView(m_view, srcView.backendImpl());
        }

        [[nodiscard]] constexpr const NativeViewType& backendImpl() const noexcept {
            return m_view;
        }

        [[nodiscard]] constexpr NativeViewType& backendImpl() noexcept {
            return m_view;
        }

        [[nodiscard]] constexpr operator NativeViewType() const noexcept {
            return m_view;
        }

    private:
        NativeViewType m_view{};
    };

    /**
     * @brief 零开销 RAII 2D 纹理对象模板（与编译期后端策略解耦）
     */
    template<ImageElementConcept T, ComputeBackendConcept Backend = ActiveBackend>
    class BasicImage {
    public:
        using ElementType     = T;
        using BackendType     = Backend;
        using NativeImageType = typename Backend::template ImageImpl<T>;
        using NativeViewType  = typename Backend::template ImageViewImpl<T>;

        BasicImage() noexcept = default;

        explicit BasicImage(NativeImageType&& image) noexcept
            : m_image(std::move(image)) {}

        BasicImage(BasicImage&&) noexcept = default;
        BasicImage& operator=(BasicImage&&) noexcept = default;
        BasicImage(const BasicImage&) = delete;
        BasicImage& operator=(const BasicImage&) = delete;

        [[nodiscard]] inline bool valid() const noexcept {
            return Backend::ImageOps::isValid(m_image);
        }

        [[nodiscard]] inline explicit operator bool() const noexcept {
            return valid();
        }

        [[nodiscard]] inline uint32_t width() const noexcept {
            return Backend::ImageOps::width(m_image);
        }

        [[nodiscard]] inline uint32_t height() const noexcept {
            return Backend::ImageOps::height(m_image);
        }

        [[nodiscard]] inline uint32_t mipLevels() const noexcept {
            return Backend::ImageOps::mipLevels(m_image);
        }

        [[nodiscard]] inline typename Backend::PixelStorageType storage() const noexcept {
            return Backend::ImageOps::storage(m_image);
        }

        [[nodiscard]] inline typename Backend::PixelFormatType pixelFormat() const noexcept {
            return Backend::ImageOps::format(m_image);
        }

        [[nodiscard]] inline FCT::Format fctFormat() const noexcept {
            return Backend::FormatOps::fromPixelFormat(pixelFormat());
        }

        [[nodiscard]] inline uint64_t handle() const noexcept {
            return Backend::ImageOps::handle(m_image);
        }

        [[nodiscard]] inline void* nativeHandle() const noexcept {
            return Backend::ImageOps::nativeHandle(m_image);
        }

        [[nodiscard]] inline BasicImageView<T, Backend> view(uint32_t level = 0u) const noexcept {
            return BasicImageView<T, Backend>(Backend::ImageOps::toView(m_image, level));
        }

        [[nodiscard]] inline operator BasicImageView<T, Backend>() const noexcept {
            return view(0u);
        }

        [[nodiscard]] inline operator NativeViewType() const noexcept {
            return Backend::ImageOps::toView(m_image, 0u);
        }

        [[nodiscard]] inline auto copyFromHost(const void* hostPixels, uint32_t level = 0u) const noexcept {
            return view(level).copyFromHost(hostPixels);
        }

        [[nodiscard]] inline auto copyToHost(void* hostPixels, uint32_t level = 0u) const noexcept {
            return view(level).copyToHost(hostPixels);
        }

        [[nodiscard]] inline auto copyFromImage(BasicImageView<T, Backend> srcView, uint32_t dstLevel = 0u) const noexcept {
            return view(dstLevel).copyFromImage(srcView);
        }

        [[nodiscard]] inline auto copyFromImage(const BasicImage<T, Backend>& srcImage, uint32_t level = 0u) const noexcept {
            return view(level).copyFromImage(srcImage.view(level));
        }

        [[nodiscard]] inline const NativeImageType& backendImpl() const noexcept {
            return m_image;
        }

        [[nodiscard]] inline NativeImageType& backendImpl() noexcept {
            return m_image;
        }

    private:
        NativeImageType m_image{};
    };

    /**
     * @brief 以编译期 FCT::Format 参数化的强类型 2D 纹理模板
     */
    template<FCT::Format F, ComputeBackendConcept Backend = ActiveBackend>
        requires SupportedFormatConcept<F>
    class BasicFormatImage : public BasicImage<typename FormatTraits<F>::ElementType, Backend> {
    public:
        using Traits      = FormatTraits<F>;
        using ElementType = typename Traits::ElementType;
        using Base        = BasicImage<ElementType, Backend>;

        static constexpr FCT::Format kFctFormat = F;
        static constexpr size_t kChannels       = Traits::channels;
        static constexpr size_t kPixelByteSize  = Traits::byteSize;

        using Base::Base;

        [[nodiscard]] static constexpr typename Backend::PixelStorageType backendStorage() noexcept {
            return Backend::FormatOps::template toPixelStorage<F>();
        }
    };

    /**
     * @brief 零开销 3D 体纹理视图模板（与编译期后端策略解耦）
     */
    template<ImageElementConcept T, ComputeBackendConcept Backend = ActiveBackend>
    class BasicVolumeView {
    public:
        using ElementType    = T;
        using BackendType    = Backend;
        using NativeViewType = typename Backend::template VolumeViewImpl<T>;

        constexpr BasicVolumeView() noexcept = default;

        constexpr explicit BasicVolumeView(NativeViewType view) noexcept
            : m_view(view) {}

        [[nodiscard]] inline bool valid() const noexcept {
            return Backend::VolumeOps::isViewValid(m_view);
        }

        [[nodiscard]] inline explicit operator bool() const noexcept {
            return valid();
        }

        [[nodiscard]] inline uint32_t width() const noexcept {
            return Backend::VolumeOps::viewWidth(m_view);
        }

        [[nodiscard]] inline uint32_t height() const noexcept {
            return Backend::VolumeOps::viewHeight(m_view);
        }

        [[nodiscard]] inline uint32_t depth() const noexcept {
            return Backend::VolumeOps::viewDepth(m_view);
        }

        [[nodiscard]] inline auto copyFromHost(const void* hostVoxels) const noexcept {
            return Backend::VolumeOps::copyFromHost(m_view, hostVoxels);
        }

        [[nodiscard]] inline auto copyToHost(void* hostVoxels) const noexcept {
            return Backend::VolumeOps::copyToHost(m_view, hostVoxels);
        }

        [[nodiscard]] inline auto copyFromVolume(BasicVolumeView<T, Backend> srcView) const noexcept {
            return Backend::VolumeOps::copyFromView(m_view, srcView.backendImpl());
        }

        [[nodiscard]] constexpr const NativeViewType& backendImpl() const noexcept {
            return m_view;
        }

        [[nodiscard]] constexpr NativeViewType& backendImpl() noexcept {
            return m_view;
        }

        [[nodiscard]] constexpr operator NativeViewType() const noexcept {
            return m_view;
        }

    private:
        NativeViewType m_view{};
    };

    /**
     * @brief 零开销 RAII 3D 体纹理对象模板（与编译期后端策略解耦）
     */
    template<ImageElementConcept T, ComputeBackendConcept Backend = ActiveBackend>
    class BasicVolume {
    public:
        using ElementType      = T;
        using BackendType      = Backend;
        using NativeVolumeType = typename Backend::template VolumeImpl<T>;
        using NativeViewType   = typename Backend::template VolumeViewImpl<T>;

        BasicVolume() noexcept = default;

        explicit BasicVolume(NativeVolumeType&& volume) noexcept
            : m_volume(std::move(volume)) {}

        BasicVolume(BasicVolume&&) noexcept = default;
        BasicVolume& operator=(BasicVolume&&) noexcept = default;
        BasicVolume(const BasicVolume&) = delete;
        BasicVolume& operator=(const BasicVolume&) = delete;

        [[nodiscard]] inline bool valid() const noexcept {
            return Backend::VolumeOps::isValid(m_volume);
        }

        [[nodiscard]] inline explicit operator bool() const noexcept {
            return valid();
        }

        [[nodiscard]] inline uint32_t width() const noexcept {
            return Backend::VolumeOps::width(m_volume);
        }

        [[nodiscard]] inline uint32_t height() const noexcept {
            return Backend::VolumeOps::height(m_volume);
        }

        [[nodiscard]] inline uint32_t depth() const noexcept {
            return Backend::VolumeOps::depth(m_volume);
        }

        [[nodiscard]] inline uint32_t mipLevels() const noexcept {
            return Backend::VolumeOps::mipLevels(m_volume);
        }

        [[nodiscard]] inline typename Backend::PixelStorageType storage() const noexcept {
            return Backend::VolumeOps::storage(m_volume);
        }

        [[nodiscard]] inline typename Backend::PixelFormatType pixelFormat() const noexcept {
            return Backend::VolumeOps::format(m_volume);
        }

        [[nodiscard]] inline uint64_t handle() const noexcept {
            return Backend::VolumeOps::handle(m_volume);
        }

        [[nodiscard]] inline void* nativeHandle() const noexcept {
            return Backend::VolumeOps::nativeHandle(m_volume);
        }

        [[nodiscard]] inline BasicVolumeView<T, Backend> view(uint32_t level = 0u) const noexcept {
            return BasicVolumeView<T, Backend>(Backend::VolumeOps::toView(m_volume, level));
        }

        [[nodiscard]] inline operator BasicVolumeView<T, Backend>() const noexcept {
            return view(0u);
        }

        [[nodiscard]] inline operator NativeViewType() const noexcept {
            return Backend::VolumeOps::toView(m_volume, 0u);
        }

        [[nodiscard]] inline auto copyFromHost(const void* hostVoxels, uint32_t level = 0u) const noexcept {
            return view(level).copyFromHost(hostVoxels);
        }

        [[nodiscard]] inline auto copyToHost(void* hostVoxels, uint32_t level = 0u) const noexcept {
            return view(level).copyToHost(hostVoxels);
        }

        [[nodiscard]] inline const NativeVolumeType& backendImpl() const noexcept {
            return m_volume;
        }

        [[nodiscard]] inline NativeVolumeType& backendImpl() noexcept {
            return m_volume;
        }

    private:
        NativeVolumeType m_volume{};
    };

    template<ImageElementConcept T>
    using ImageView = BasicImageView<T, ActiveBackend>;

    template<ImageElementConcept T>
    using Image = BasicImage<T, ActiveBackend>;

    template<FCT::Format F>
        requires SupportedFormatConcept<F>
    using FormatImage = BasicFormatImage<F, ActiveBackend>;

    template<ImageElementConcept T>
    using VolumeView = BasicVolumeView<T, ActiveBackend>;

    template<ImageElementConcept T>
    using Volume = BasicVolume<T, ActiveBackend>;

    static_assert(sizeof(Image<float>) == sizeof(typename ActiveBackend::template ImageImpl<float>),
                  "Zero-overhead violation: Image<T> must match ActiveBackend::ImageImpl<T> size.");
    static_assert(sizeof(ImageView<float>) == sizeof(typename ActiveBackend::template ImageViewImpl<float>),
                  "Zero-overhead violation: ImageView<T> must match ActiveBackend::ImageViewImpl<T> size.");
    static_assert(sizeof(Volume<float>) == sizeof(typename ActiveBackend::template VolumeImpl<float>),
                  "Zero-overhead violation: Volume<T> must match ActiveBackend::VolumeImpl<T> size.");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_CORE_IMAGE_H
