#ifndef FCT_COMPUTE_CORE_CONTEXT_H
#define FCT_COMPUTE_CORE_CONTEXT_H

#include "ComputeDevice.h"

namespace FCT::Compute {

    using Backend = HardwareBackend;

    /**
     * @brief 零开销计算运行时上下文封装（与编译期后端策略解耦）
     */
    template<ComputeBackendConcept BackendPolicy = ActiveBackend>
    class BasicContext {
    public:
        using BackendType       = BackendPolicy;
        using NativeContextType = typename BackendPolicy::ContextImpl;

        explicit BasicContext(const std::filesystem::path& runtimePath) noexcept
            : m_context(BackendPolicy::ContextOps::create(runtimePath)) {}

        explicit BasicContext(const char* argv0) noexcept
            : m_context(BackendPolicy::ContextOps::create(argv0)) {}

        explicit BasicContext(NativeContextType&& context) noexcept
            : m_context(std::move(context)) {}

        BasicContext(const BasicContext&) noexcept = default;
        BasicContext& operator=(const BasicContext&) noexcept = default;
        BasicContext(BasicContext&&) noexcept = default;
        BasicContext& operator=(BasicContext&&) noexcept = default;

        [[nodiscard]] inline StatusOr<BasicDevice<BackendPolicy>> createDevice(
            HardwareBackend hwBackend = HardwareBackend::Cuda,
            uint32_t deviceIndex = 0u) noexcept {
            return createDevice(ToHardwareBackendName(hwBackend), deviceIndex);
        }

        [[nodiscard]] inline StatusOr<BasicDevice<BackendPolicy>> createDevice(
            std::string_view hwBackendName,
            uint32_t deviceIndex = 0u) noexcept {
            auto dev = BackendPolicy::ContextOps::createDevice(m_context, hwBackendName, deviceIndex);
            if (!BackendPolicy::DeviceOps::isValid(dev)) {
                return InternalError(StrCat(
                    "Failed to create ", BackendPolicy::backendName,
                    " device for hardware backend: ", hwBackendName));
            }
            return BasicDevice<BackendPolicy>(std::move(dev));
        }

        [[nodiscard]] inline std::filesystem::path runtimeDirectory() const noexcept {
            return BackendPolicy::ContextOps::runtimeDirectory(m_context);
        }

        [[nodiscard]] static constexpr std::string_view computeBackendName() noexcept {
            return BackendPolicy::backendName;
        }

        [[nodiscard]] inline NativeContextType& backendImpl() noexcept {
            return m_context;
        }

        [[nodiscard]] inline const NativeContextType& backendImpl() const noexcept {
            return m_context;
        }

    private:
        NativeContextType m_context;
    };

    using ComputeContext = BasicContext<ActiveBackend>;

    static_assert(sizeof(ComputeContext) == sizeof(typename ActiveBackend::ContextImpl),
                  "Zero-overhead violation: ComputeContext must match ActiveBackend::ContextImpl size.");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_CORE_CONTEXT_H
