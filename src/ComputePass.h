#ifndef FCT_COMPUTE_PASS_WRAPPER_H
#define FCT_COMPUTE_PASS_WRAPPER_H

#include "ComputeContext.h"

namespace FCT::Compute {

    /**
     * @brief RenderGraph 计算通道执行上下文，关联 ComputeDevice、ComputeStream 与底层图资源
     */
    class ComputePassContext {
    public:
        ComputePassContext(
            ComputeDevice& device,
            ComputeStream& stream,
            FCT::ComputeExecutionContext* graphContext = nullptr) noexcept
            : m_device(device),
              m_stream(stream),
              m_graphContext(graphContext) {}

        [[nodiscard]] ComputeDevice& device() noexcept {
            return m_device;
        }

        [[nodiscard]] const ComputeDevice& device() const noexcept {
            return m_device;
        }

        [[nodiscard]] ComputeStream& stream() noexcept {
            return m_stream;
        }

        [[nodiscard]] FCT::ComputeExecutionContext* graphContext() const noexcept {
            return m_graphContext;
        }

        [[nodiscard]] FCT::Image* getGraphImage(const std::string& name) const {
            if (!m_graphContext) {
                return nullptr;
            }
            return m_graphContext->getImage(name);
        }

        [[nodiscard]] void* getExternalNativeHandle(const std::string& name) const {
            if (!m_graphContext) {
                return nullptr;
            }
            return m_graphContext->getExternalNativeHandle(name);
        }

    private:
        ComputeDevice& m_device;
        ComputeStream& m_stream;
        FCT::ComputeExecutionContext* m_graphContext;
    };

    /**
     * @brief 对接 FCT::IComputeDispatcher 的计算通道基类
     */
    class ComputePass : public FCT::IComputeDispatcher {
    public:
        ComputePass(std::string name, ComputeDevice device, ComputeStream stream) noexcept;
        ~ComputePass() override = default;

        [[nodiscard]] const std::string& name() const noexcept {
            return m_name;
        }

        [[nodiscard]] ComputeDevice& device() noexcept {
            return m_device;
        }

        [[nodiscard]] ComputeStream& stream() noexcept {
            return m_stream;
        }

        virtual Status execute(ComputePassContext& context) = 0;

        Status dispatch() noexcept;
        Status dispatch(FCT::ComputeExecutionContext& context) override;

    protected:
        std::string m_name;
        ComputeDevice m_device;
        ComputeStream m_stream;
    };

    /**
     * @brief 基于模板内联函子的计算通道，消除 std::function 类型擦除与堆分配开销
     */
    template<typename Func>
        requires std::invocable<Func&, ComputePassContext&>
    class InlineComputePass : public ComputePass {
    public:
        template<typename F>
        InlineComputePass(std::string name, ComputeDevice device, ComputeStream stream, F&& fn) noexcept
            : ComputePass(std::move(name), std::move(device), std::move(stream)),
              m_func(std::forward<F>(fn)) {}

        Status execute(ComputePassContext& context) override {
            if constexpr (std::is_same_v<std::invoke_result_t<Func&, ComputePassContext&>, Status>) {
                return m_func(context);
            } else {
                m_func(context);
                return OkStatus();
            }
        }

    private:
        [[no_unique_address]] Func m_func;
    };

    template<typename Func>
    [[nodiscard]] auto MakeInlineComputePass(
        std::string name,
        ComputeDevice device,
        ComputeStream stream,
        Func&& fn) {
        using CleanFunc = std::decay_t<Func>;
        return new InlineComputePass<CleanFunc>(
            std::move(name),
            std::move(device),
            std::move(stream),
            std::forward<Func>(fn));
    }

} // namespace FCT::Compute

#endif // FCT_COMPUTE_PASS_WRAPPER_H
