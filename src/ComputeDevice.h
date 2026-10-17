#ifndef FCT_COMPUTE_DEVICE_H
#define FCT_COMPUTE_DEVICE_H

#include "ComputeStream.h"

namespace FCT::Compute {

    /**
     * @brief 零开销计算设备封装，内存布局与 luisa::compute::Device 严格一致
     */
    class ComputeDevice {
    public:
        ComputeDevice() noexcept = default;

        explicit ComputeDevice(luisa::compute::Device&& device) noexcept
            : m_device(std::move(device)) {}

        ComputeDevice(ComputeDevice&&) noexcept = default;
        ComputeDevice& operator=(ComputeDevice&&) noexcept = default;
        ComputeDevice(const ComputeDevice&) = default;
        ComputeDevice& operator=(const ComputeDevice&) = default;
        ~ComputeDevice() noexcept = default;

        [[nodiscard]] explicit operator bool() const noexcept {
            return static_cast<bool>(m_device);
        }

        [[nodiscard]] operator luisa::compute::Device&() noexcept {
            return m_device;
        }

        [[nodiscard]] operator const luisa::compute::Device&() const noexcept {
            return m_device;
        }

        [[nodiscard]] const luisa::compute::Device& raw() const noexcept {
            return m_device;
        }

        [[nodiscard]] luisa::compute::Device& raw() noexcept {
            return m_device;
        }

        [[nodiscard]] std::string_view backendName() const noexcept {
            return m_device.backend_name();
        }

        [[nodiscard]] void* nativeHandle() const noexcept {
            return m_device.native_handle();
        }

        [[nodiscard]] uint32_t computeWarpSize() const noexcept {
            return m_device.compute_warp_size();
        }

        [[nodiscard]] StatusOr<ComputeStream> createStream(
            luisa::compute::StreamTag tag = luisa::compute::StreamTag::COMPUTE) noexcept {
            if (!m_device) {
                return FailedPreconditionError("ComputeDevice is not initialized.");
            }
            auto stream = m_device.create_stream(tag);
            if (!stream) {
                return InternalError("Failed to create LuisaCompute stream.");
            }
            return ComputeStream(std::move(stream));
        }

        template<BufferElementConcept T>
        [[nodiscard]] StatusOr<Buffer<T>> createBuffer(size_t elementCount) noexcept {
            if (!m_device) {
                return FailedPreconditionError("ComputeDevice is not initialized.");
            }
            if (elementCount == 0u) {
                return InvalidArgumentError("Buffer elementCount must be greater than zero.");
            }
            auto nativeBuf = m_device.create_buffer<StorageType_t<T>>(elementCount);
            if (!nativeBuf) {
                return InternalError("Failed to create LuisaCompute buffer.");
            }
            return Buffer<T>(std::move(nativeBuf));
        }

        template<BufferElementConcept T>
        [[nodiscard]] StatusOr<Buffer<T>> importExternalBuffer(void* externalMemory, size_t elementCount) noexcept {
            if (!m_device) {
                return FailedPreconditionError("ComputeDevice is not initialized.");
            }
            if (!externalMemory || elementCount == 0u) {
                return InvalidArgumentError("Invalid external buffer memory or elementCount.");
            }
            auto nativeBuf = m_device.import_external_buffer<StorageType_t<T>>(externalMemory, elementCount);
            if (!nativeBuf) {
                return InternalError("Failed to import external buffer into LuisaCompute.");
            }
            return Buffer<T>(std::move(nativeBuf));
        }

        template<ImageElementConcept T>
        [[nodiscard]] StatusOr<Image<T>> createImage(
            luisa::compute::PixelStorage storage,
            uint32_t width,
            uint32_t height,
            uint32_t mipLevels = 1u,
            bool simultaneousAccess = false) noexcept {
            if (!m_device) {
                return FailedPreconditionError("ComputeDevice is not initialized.");
            }
            if (width == 0u || height == 0u) {
                return InvalidArgumentError("Image dimensions must be greater than zero.");
            }
            auto nativeImg = m_device.create_image<T>(storage, width, height, mipLevels, simultaneousAccess);
            if (!nativeImg) {
                return InternalError("Failed to create LuisaCompute image.");
            }
            return Image<T>(std::move(nativeImg));
        }

        /**
         * @brief 利用编译期 FormatTraits 直接创建与 FCT::Format 严格匹配的 GPU 2D 纹理
         */
        template<FCT::Format F>
            requires SupportedFormatConcept<F>
        [[nodiscard]] StatusOr<FormatImage<F>> createImageWithFormat(
            uint32_t width,
            uint32_t height,
            uint32_t mipLevels = 1u,
            bool simultaneousAccess = false) noexcept {
            using Traits = FormatTraits<F>;
            return createImage<typename Traits::ElementType>(
                Traits::storage, width, height, mipLevels, simultaneousAccess);
        }

        template<FCT::Format F>
            requires SupportedFormatConcept<F>
        [[nodiscard]] StatusOr<FormatImage<F>> importExternalImageWithFormat(
            void* externalNativeHandle,
            uint32_t width,
            uint32_t height,
            uint32_t mipLevels = 1u,
            bool simultaneousAccess = false) noexcept {
            if (!m_device) {
                return FailedPreconditionError("ComputeDevice is not initialized.");
            }
            if (!externalNativeHandle || width == 0u || height == 0u) {
                return InvalidArgumentError("Invalid external image handle or dimensions.");
            }
            using Traits = FormatTraits<F>;
            auto nativeImg = m_device.import_external_image<typename Traits::ElementType>(
                Traits::storage,
                externalNativeHandle,
                luisa::make_uint2(width, height),
                mipLevels,
                simultaneousAccess);
            if (!nativeImg) {
                return InternalError("Failed to import external image into LuisaCompute.");
            }
            return FormatImage<F>(std::move(nativeImg));
        }

        template<ImageElementConcept T>
        [[nodiscard]] StatusOr<Volume<T>> createVolume(
            luisa::compute::PixelStorage storage,
            uint32_t width,
            uint32_t height,
            uint32_t depth,
            uint32_t mipLevels = 1u,
            bool simultaneousAccess = false) noexcept {
            if (!m_device) {
                return FailedPreconditionError("ComputeDevice is not initialized.");
            }
            if (width == 0u || height == 0u || depth == 0u) {
                return InvalidArgumentError("Volume dimensions must be greater than zero.");
            }
            auto nativeVol = m_device.create_volume<T>(storage, width, height, depth, mipLevels, simultaneousAccess);
            if (!nativeVol) {
                return InternalError("Failed to create LuisaCompute volume.");
            }
            return Volume<T>(std::move(nativeVol));
        }

        template<size_t Dim, typename... Args>
            requires KernelDimensionConcept<Dim>
        [[nodiscard]] StatusOr<CompiledKernel<Dim, Args...>> compile(
            const luisa::compute::Kernel<Dim, Args...>& kernel,
            const luisa::compute::ShaderOption& option = {}) noexcept {
            if (!m_device) {
                return FailedPreconditionError("ComputeDevice is not initialized.");
            }
            auto shader = m_device.compile(kernel, option);
            if (!shader) {
                return InternalError("Failed to compile LuisaCompute kernel.");
            }
            return CompiledKernel<Dim, Args...>(std::move(shader));
        }

        template<size_t Dim, typename BlockCfg = DefaultBlockSize_t<Dim>, typename Def>
            requires (KernelDimensionConcept<Dim> &&
                      !luisa::compute::is_kernel<std::remove_cvref_t<Def>>::value)
        [[nodiscard]] auto compile(
            Def&& def,
            const luisa::compute::ShaderOption& option = {}) noexcept {
            auto kernel = MakeKernel<Dim, BlockCfg>(std::forward<Def>(def));
            return compile(kernel, option);
        }

        template<uint32_t BlockX = 256u, typename Def>
        [[nodiscard]] auto compile1D(
            Def&& def,
            const luisa::compute::ShaderOption& option = {}) noexcept {
            return compile<1u, BlockSize<BlockX, 1u, 1u>>(std::forward<Def>(def), option);
        }

        template<uint32_t BlockX = 16u, uint32_t BlockY = 16u, typename Def>
        [[nodiscard]] auto compile2D(
            Def&& def,
            const luisa::compute::ShaderOption& option = {}) noexcept {
            return compile<2u, BlockSize<BlockX, BlockY, 1u>>(std::forward<Def>(def), option);
        }

        template<uint32_t BlockX = 8u, uint32_t BlockY = 8u, uint32_t BlockZ = 8u, typename Def>
        [[nodiscard]] auto compile3D(
            Def&& def,
            const luisa::compute::ShaderOption& option = {}) noexcept {
            return compile<3u, BlockSize<BlockX, BlockY, BlockZ>>(std::forward<Def>(def), option);
        }

    private:
        luisa::compute::Device m_device;
    };

    static_assert(sizeof(ComputeDevice) == sizeof(luisa::compute::Device),
                  "Zero-overhead guarantee: ComputeDevice must match luisa::compute::Device size");
    static_assert(alignof(ComputeDevice) == alignof(luisa::compute::Device),
                  "Zero-overhead guarantee: ComputeDevice must match luisa::compute::Device alignment");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_DEVICE_H
