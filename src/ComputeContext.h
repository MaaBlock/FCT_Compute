#ifndef FCT_COMPUTE_CONTEXT_H
#define FCT_COMPUTE_CONTEXT_H

#include "ComputeDevice.h"

namespace FCT::Compute {

    enum class Backend : uint8_t {
        Cuda,
        DirectX,
        Vulkan,
        Metal,
        Fallback
    };

    [[nodiscard]] constexpr std::string_view ToBackendName(Backend backend) noexcept {
        switch (backend) {
            case Backend::Cuda:     return "cuda";
            case Backend::DirectX:  return "dx";
            case Backend::Vulkan:   return "vk";
            case Backend::Metal:    return "metal";
            case Backend::Fallback: return "fallback";
        }
        return "cuda";
    }

    /**
     * @brief 零开销计算上下文封装，内存布局与 luisa::compute::Context 严格一致
     */
    class ComputeContext {
    public:
        explicit ComputeContext(std::string_view programPath) noexcept
            : m_context(luisa::string_view{programPath.data(), programPath.size()}) {}

        explicit ComputeContext(std::string_view programPath, std::string_view dataDir) noexcept
            : m_context(
                  luisa::string_view{programPath.data(), programPath.size()},
                  luisa::string_view{dataDir.data(), dataDir.size()}) {}

        explicit ComputeContext(luisa::compute::Context&& context) noexcept
            : m_context(std::move(context)) {}

        ComputeContext(ComputeContext&&) noexcept = default;
        ComputeContext& operator=(ComputeContext&&) noexcept = default;
        ComputeContext(const ComputeContext&) = default;
        ComputeContext& operator=(const ComputeContext&) = default;
        ~ComputeContext() noexcept = default;

        [[nodiscard]] const luisa::compute::Context& raw() const noexcept {
            return m_context;
        }

        [[nodiscard]] luisa::compute::Context& raw() noexcept {
            return m_context;
        }

        [[nodiscard]] luisa::span<const luisa::string> installedBackends() const noexcept {
            return m_context.installed_backends();
        }

        [[nodiscard]] bool isBackendAvailable(std::string_view backendName) const noexcept {
            for (const auto& name : m_context.installed_backends()) {
                if (std::string_view{name.data(), name.size()} == backendName) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] bool isBackendAvailable(Backend backend) const noexcept {
            return isBackendAvailable(ToBackendName(backend));
        }

        [[nodiscard]] StatusOr<ComputeDevice> createDevice(
            std::string_view backendName,
            const luisa::compute::DeviceConfig* config = nullptr) noexcept {
            if (!isBackendAvailable(backendName)) {
                return NotFoundError(StrCat("LuisaCompute backend is not installed: ", backendName));
            }
            auto dev = m_context.create_device(
                luisa::string_view{backendName.data(), backendName.size()},
                config);
            if (!dev) {
                return InternalError(StrCat("Failed to create LuisaCompute device for backend: ", backendName));
            }
            return ComputeDevice(std::move(dev));
        }

        [[nodiscard]] StatusOr<ComputeDevice> createDevice(
            Backend backend,
            const luisa::compute::DeviceConfig* config = nullptr) noexcept {
            return createDevice(ToBackendName(backend), config);
        }

        [[nodiscard]] StatusOr<ComputeDevice> createDefaultDevice() noexcept {
            if (m_context.installed_backends().empty()) {
                return NotFoundError("No LuisaCompute backends are installed in the runtime directory.");
            }
            auto dev = m_context.create_default_device();
            if (!dev) {
                return InternalError("Failed to create default LuisaCompute device.");
            }
            return ComputeDevice(std::move(dev));
        }

    private:
        luisa::compute::Context m_context;
    };

    static_assert(sizeof(ComputeContext) == sizeof(luisa::compute::Context),
                  "Zero-overhead guarantee: ComputeContext must match luisa::compute::Context size");
    static_assert(alignof(ComputeContext) == alignof(luisa::compute::Context),
                  "Zero-overhead guarantee: ComputeContext must match luisa::compute::Context alignment");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_CONTEXT_H
