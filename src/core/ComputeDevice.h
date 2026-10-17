#ifndef FCT_COMPUTE_CORE_DEVICE_H
#define FCT_COMPUTE_CORE_DEVICE_H

#include "ComputeBuffer.h"
#include "ComputeImage.h"
#include "ComputeStream.h"
#include "../dsl/ComputeKernel.h"

namespace FCT::Compute {

    /**
     * @brief 零开销计算设备封装（与编译期后端策略解耦）
     */
    template<ComputeBackendConcept Backend = ActiveBackend>
    class BasicDevice {
    public:
        using BackendType      = Backend;
        using NativeDeviceType = typename Backend::DeviceImpl;

        BasicDevice() noexcept = default;

        explicit BasicDevice(NativeDeviceType&& device) noexcept
            : m_device(std::move(device)) {}

        BasicDevice(const BasicDevice&) noexcept = default;
        BasicDevice& operator=(const BasicDevice&) noexcept = default;
        BasicDevice(BasicDevice&&) noexcept = default;
        BasicDevice& operator=(BasicDevice&&) noexcept = default;

        [[nodiscard]] inline bool valid() const noexcept {
            return Backend::DeviceOps::isValid(m_device);
        }

        [[nodiscard]] inline explicit operator bool() const noexcept {
            return valid();
        }

        [[nodiscard]] inline void* nativeHandle() const noexcept {
            return Backend::DeviceOps::nativeHandle(m_device);
        }

        [[nodiscard]] inline std::string_view hardwareBackendName() const noexcept {
            return Backend::DeviceOps::backendName(m_device);
        }

        [[nodiscard]] static constexpr std::string_view computeBackendName() noexcept {
            return Backend::backendName;
        }

        [[nodiscard]] inline BasicStream<Backend> createStream(bool forPresent = false) noexcept {
            return BasicStream<Backend>(Backend::DeviceOps::createStream(m_device, forPresent));
        }

        template<BufferElementConcept T>
        [[nodiscard]] inline BasicBuffer<T, Backend> createBuffer(size_t elementCount) noexcept {
            using StorageT = StorageType_t<T>;
            return BasicBuffer<T, Backend>(Backend::DeviceOps::template createBuffer<StorageT>(m_device, elementCount));
        }

        template<BufferElementConcept T>
        [[nodiscard]] inline BasicBuffer<T, Backend> importExternalBuffer(void* nativeAddress, size_t elementCount) noexcept {
            using StorageT = StorageType_t<T>;
            return BasicBuffer<T, Backend>(
                Backend::DeviceOps::template importExternalBuffer<StorageT>(m_device, nativeAddress, elementCount));
        }

        template<ImageElementConcept T>
        [[nodiscard]] inline BasicImage<T, Backend> createImage(
            typename Backend::PixelStorageType storage,
            uint32_t width,
            uint32_t height,
            uint32_t mipLevels = 1u) noexcept {
            return BasicImage<T, Backend>(
                Backend::DeviceOps::template createImage<T>(m_device, storage, width, height, mipLevels));
        }

        template<FCT::Format F>
            requires SupportedFormatConcept<F>
        [[nodiscard]] inline BasicFormatImage<F, Backend> createImageWithFormat(
            uint32_t width,
            uint32_t height,
            uint32_t mipLevels = 1u) noexcept {
            using ElemT = typename FormatTraits<F>::ElementType;
            constexpr auto storage = Backend::FormatOps::template toPixelStorage<F>();
            return BasicFormatImage<F, Backend>(
                Backend::DeviceOps::template createImage<ElemT>(m_device, storage, width, height, mipLevels));
        }

        template<FCT::Format F>
            requires SupportedFormatConcept<F>
        [[nodiscard]] inline BasicFormatImage<F, Backend> importExternalImageWithFormat(
            uint64_t externalNativeHandle,
            uint32_t width,
            uint32_t height,
            uint32_t mipLevels = 1u) noexcept {
            using ElemT = typename FormatTraits<F>::ElementType;
            constexpr auto storage = Backend::FormatOps::template toPixelStorage<F>();
            return BasicFormatImage<F, Backend>(
                Backend::DeviceOps::template importExternalImage<ElemT>(
                    m_device, externalNativeHandle, storage, width, height, mipLevels));
        }

        template<ImageElementConcept T>
        [[nodiscard]] inline BasicVolume<T, Backend> createVolume(
            typename Backend::PixelStorageType storage,
            uint32_t width,
            uint32_t height,
            uint32_t depth,
            uint32_t mipLevels = 1u) noexcept {
            return BasicVolume<T, Backend>(
                Backend::DeviceOps::template createVolume<T>(m_device, storage, width, height, depth, mipLevels));
        }

        template<size_t Dim, typename BlockCfg, typename Func>
            requires KernelDimensionConcept<Dim>
        [[nodiscard]] inline auto compile(const BasicKernelDef<Dim, BlockCfg, Func, Backend>& kernelDef) noexcept {
            using HostArgTuple       = typename BasicKernelDef<Dim, BlockCfg, Func, Backend>::HostArgTuple;
            using CompiledKernelType = Detail::CompiledKernelFromTuple_t<Dim, Backend, HostArgTuple>;
            return CompiledKernelType(Backend::DeviceOps::compileKernel(m_device, kernelDef.backendImpl()));
        }

        template<typename BlockCfg = DefaultBlockSize_t<1>, typename Func>
        [[nodiscard]] inline auto compile1D(Func&& func) noexcept {
            auto kernelDef = BasicKernelDef<1, BlockCfg, std::remove_cvref_t<Func>, Backend>(std::forward<Func>(func));
            return compile(kernelDef);
        }

        template<typename BlockCfg = DefaultBlockSize_t<2>, typename Func>
        [[nodiscard]] inline auto compile2D(Func&& func) noexcept {
            auto kernelDef = BasicKernelDef<2, BlockCfg, std::remove_cvref_t<Func>, Backend>(std::forward<Func>(func));
            return compile(kernelDef);
        }

        template<typename BlockCfg = DefaultBlockSize_t<3>, typename Func>
        [[nodiscard]] inline auto compile3D(Func&& func) noexcept {
            auto kernelDef = BasicKernelDef<3, BlockCfg, std::remove_cvref_t<Func>, Backend>(std::forward<Func>(func));
            return compile(kernelDef);
        }

        [[nodiscard]] inline NativeDeviceType& backendImpl() noexcept {
            return m_device;
        }

        [[nodiscard]] inline const NativeDeviceType& backendImpl() const noexcept {
            return m_device;
        }

    private:
        NativeDeviceType m_device{};
    };

    using ComputeDevice = BasicDevice<ActiveBackend>;

    static_assert(sizeof(ComputeDevice) == sizeof(typename ActiveBackend::DeviceImpl),
                  "Zero-overhead violation: ComputeDevice must match ActiveBackend::DeviceImpl size.");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_CORE_DEVICE_H
