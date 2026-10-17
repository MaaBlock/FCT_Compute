#ifndef FCT_COMPUTE_CORE_PASS_H
#define FCT_COMPUTE_CORE_PASS_H

#include "ComputeContext.h"

namespace FCT::Compute {

    /**
     * @brief 计算通道执行上下文（提供与 FCT RenderGraph 纹理互操作的零开销桥接）
     */
    template<ComputeBackendConcept BackendPolicy = ActiveBackend>
    class BasicPassContext {
    public:
        BasicPassContext(
            BasicDevice<BackendPolicy>& device,
            BasicStream<BackendPolicy>& stream,
            FCT::ComputeExecutionContext* graphCtx = nullptr) noexcept
            : m_device(device),
              m_stream(stream),
              m_graphCtx(graphCtx) {}

        [[nodiscard]] inline BasicDevice<BackendPolicy>& device() noexcept {
            return m_device;
        }

        [[nodiscard]] inline BasicStream<BackendPolicy>& stream() noexcept {
            return m_stream;
        }

        [[nodiscard]] inline FCT::ComputeExecutionContext* renderGraphContext() const noexcept {
            return m_graphCtx;
        }

        [[nodiscard]] inline StatusOr<FCT::Image*> getRenderGraphImage(const std::string& name) const noexcept {
            if (!m_graphCtx) {
                return FailedPreconditionError("RenderGraph ComputeExecutionContext is not bound.");
            }
            FCT::Image* image = m_graphCtx->getImage(name);
            if (!image) {
                return NotFoundError(StrCat("RenderGraph image not found: ", name));
            }
            return image;
        }

        [[nodiscard]] inline StatusOr<void*> getExternalNativeHandle(const std::string& name) const noexcept {
            if (!m_graphCtx) {
                return FailedPreconditionError("RenderGraph ComputeExecutionContext is not bound.");
            }
            void* handle = m_graphCtx->getExternalNativeHandle(name);
            if (!handle) {
                return NotFoundError(StrCat("External native handle not found: ", name));
            }
            return handle;
        }

        template<FCT::Format F>
            requires SupportedFormatConcept<F>
        [[nodiscard]] inline StatusOr<BasicFormatImage<F, BackendPolicy>> importRenderGraphImage(
            const std::string& name,
            uint32_t mipLevels = 1u) noexcept {
            auto fctImgOr = getRenderGraphImage(name);
            CHECK_STATUS(fctImgOr);
            auto rawHandleOr = getExternalNativeHandle(name);
            CHECK_STATUS(rawHandleOr);
            FCT::Image* fctImg = *fctImgOr;
            void* rawHandle    = *rawHandleOr;
            const uint32_t width  = static_cast<uint32_t>(fctImg->width());
            const uint32_t height = static_cast<uint32_t>(fctImg->height());
            return m_device.template importExternalImageWithFormat<F>(
                reinterpret_cast<uint64_t>(rawHandle), width, height, mipLevels);
        }

    private:
        BasicDevice<BackendPolicy>& m_device;
        BasicStream<BackendPolicy>& m_stream;
        FCT::ComputeExecutionContext* m_graphCtx;
    };

    using ComputePassContext = BasicPassContext<ActiveBackend>;

    /**
     * @brief 面向 FCT 渲染图调度的计算通道基类
     */
    class ComputePass : public FCT::IComputeDispatcher {
    public:
        ComputePass(std::string name, ComputeDevice device, ComputeStream stream) noexcept;
        ~ComputePass() override = default;

        Status dispatch() noexcept;
        Status dispatch(FCT::ComputeExecutionContext& context) override;

        [[nodiscard]] const std::string& name() const noexcept { return m_name; }
        [[nodiscard]] ComputeDevice& device() noexcept { return m_device; }
        [[nodiscard]] const ComputeDevice& device() const noexcept { return m_device; }
        [[nodiscard]] ComputeStream& stream() noexcept { return m_stream; }
        [[nodiscard]] const ComputeStream& stream() const noexcept { return m_stream; }

    protected:
        virtual Status execute(ComputePassContext& passCtx) = 0;

        std::string m_name;
        ComputeDevice m_device;
        ComputeStream m_stream;
    };

    /**
     * @brief 基于模板内联可调用对象的零间接分配计算通道
     * Note: 利用 [[no_unique_address]] 将无状态或带捕获的 Lambda 直接内联存储于通道对象内，避免 std::function 堆分配。
     */
    template<typename ExecuteFunc>
        requires std::invocable<ExecuteFunc, ComputePassContext&>
    class InlineComputePass : public ComputePass {
    public:
        InlineComputePass(
            std::string name,
            ComputeDevice device,
            ComputeStream stream,
            ExecuteFunc&& func) noexcept
            : ComputePass(std::move(name), std::move(device), std::move(stream)),
              m_func(std::forward<ExecuteFunc>(func)) {}

    protected:
        Status execute(ComputePassContext& passCtx) override {
            if constexpr (std::same_as<std::invoke_result_t<ExecuteFunc, ComputePassContext&>, Status>) {
                return m_func(passCtx);
            } else {
                m_func(passCtx);
                return OkStatus();
            }
        }

    private:
        [[no_unique_address]] ExecuteFunc m_func;
    };

    template<typename ExecuteFunc>
        requires std::invocable<ExecuteFunc, ComputePassContext&>
    [[nodiscard]] inline FCT::SharePtr<InlineComputePass<std::remove_cvref_t<ExecuteFunc>>> MakeComputePass(
        std::string name,
        ComputeDevice device,
        ComputeStream stream,
        ExecuteFunc&& func) {
        using PassType = InlineComputePass<std::remove_cvref_t<ExecuteFunc>>;
        return FCT::makeUnique<PassType>(
            std::move(name),
            std::move(device),
            std::move(stream),
            std::forward<ExecuteFunc>(func));
    }

} // namespace FCT::Compute

#endif // FCT_COMPUTE_CORE_PASS_H
