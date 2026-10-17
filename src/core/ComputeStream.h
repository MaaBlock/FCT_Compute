#ifndef FCT_COMPUTE_CORE_STREAM_H
#define FCT_COMPUTE_CORE_STREAM_H

#include "BackendConfig.h"

namespace FCT::Compute {

    /**
     * @brief 零开销计算命令批处理器（与编译期后端策略解耦）
     */
    template<ComputeBackendConcept Backend = ActiveBackend>
    class BasicCommandBatch {
    public:
        using BackendType           = Backend;
        using NativeCommandListType = typename Backend::CommandListImpl;

        BasicCommandBatch() noexcept
            : m_list(Backend::StreamOps::createCommandList()) {}

        explicit BasicCommandBatch(NativeCommandListType&& list) noexcept
            : m_list(std::move(list)) {}

        BasicCommandBatch(BasicCommandBatch&&) noexcept = default;
        BasicCommandBatch& operator=(BasicCommandBatch&&) noexcept = default;
        BasicCommandBatch(const BasicCommandBatch&) = delete;
        BasicCommandBatch& operator=(const BasicCommandBatch&) = delete;

        template<typename Cmd>
        inline BasicCommandBatch& add(Cmd&& cmd) noexcept {
            Backend::StreamOps::appendCommand(m_list, std::forward<Cmd>(cmd));
            return *this;
        }

        template<typename... Cmds>
            requires (sizeof...(Cmds) > 0)
        inline BasicCommandBatch& append(Cmds&&... cmds) noexcept {
            (Backend::StreamOps::appendCommand(m_list, std::forward<Cmds>(cmds)), ...);
            return *this;
        }

        template<typename Callback>
        inline BasicCommandBatch& addCallback(Callback&& callback) noexcept {
            Backend::StreamOps::appendCallback(m_list, std::forward<Callback>(callback));
            return *this;
        }

        [[nodiscard]] inline bool empty() const noexcept {
            return Backend::StreamOps::isCommandListEmpty(m_list);
        }

        [[nodiscard]] inline NativeCommandListType&& releaseBackendImpl() noexcept {
            return std::move(m_list);
        }

        [[nodiscard]] inline NativeCommandListType& backendImpl() noexcept {
            return m_list;
        }

        [[nodiscard]] inline const NativeCommandListType& backendImpl() const noexcept {
            return m_list;
        }

    private:
        NativeCommandListType m_list;
    };

    /**
     * @brief 零开销异步计算流封装（与编译期后端策略解耦）
     */
    template<ComputeBackendConcept Backend = ActiveBackend>
    class BasicStream {
    public:
        using BackendType      = Backend;
        using NativeStreamType = typename Backend::StreamImpl;

        BasicStream() noexcept = default;

        explicit BasicStream(NativeStreamType&& stream) noexcept
            : m_stream(std::move(stream)) {}

        BasicStream(BasicStream&&) noexcept = default;
        BasicStream& operator=(BasicStream&&) noexcept = default;
        BasicStream(const BasicStream&) = delete;
        BasicStream& operator=(const BasicStream&) = delete;

        [[nodiscard]] inline bool valid() const noexcept {
            return Backend::StreamOps::isValid(m_stream);
        }

        [[nodiscard]] inline explicit operator bool() const noexcept {
            return valid();
        }

        [[nodiscard]] inline void* nativeHandle() const noexcept {
            return Backend::StreamOps::nativeHandle(m_stream);
        }

        template<typename... Cmds>
            requires (sizeof...(Cmds) > 0)
        inline BasicStream& submit(Cmds&&... cmds) noexcept {
            if constexpr (sizeof...(Cmds) == 1) {
                (Backend::StreamOps::submitSingle(m_stream, std::forward<Cmds>(cmds)), ...);
            } else {
                auto list = Backend::StreamOps::createCommandList();
                (Backend::StreamOps::appendCommand(list, std::forward<Cmds>(cmds)), ...);
                Backend::StreamOps::submitCommandList(m_stream, std::move(list));
            }
            return *this;
        }

        inline BasicStream& submitBatch(BasicCommandBatch<Backend>&& batch) noexcept {
            if (!batch.empty()) {
                Backend::StreamOps::submitCommandList(m_stream, batch.releaseBackendImpl());
            }
            return *this;
        }

        template<typename... Cmds>
            requires (sizeof...(Cmds) > 0)
        inline void submitAndSync(Cmds&&... cmds) noexcept {
            if constexpr (sizeof...(Cmds) == 1) {
                (Backend::StreamOps::submitSingleAndSync(m_stream, std::forward<Cmds>(cmds)), ...);
            } else {
                auto list = Backend::StreamOps::createCommandList();
                (Backend::StreamOps::appendCommand(list, std::forward<Cmds>(cmds)), ...);
                Backend::StreamOps::submitAndSyncCommandList(m_stream, std::move(list));
            }
        }

        inline void synchronize() noexcept {
            Backend::StreamOps::synchronize(m_stream);
        }

        template<typename Cmd>
        inline BasicStream& operator<<(Cmd&& cmd) noexcept {
            Backend::StreamOps::submitSingle(m_stream, std::forward<Cmd>(cmd));
            return *this;
        }

        [[nodiscard]] inline NativeStreamType& backendImpl() noexcept {
            return m_stream;
        }

        [[nodiscard]] inline const NativeStreamType& backendImpl() const noexcept {
            return m_stream;
        }

    private:
        NativeStreamType m_stream{};
    };

    using CommandBatch  = BasicCommandBatch<ActiveBackend>;
    using ComputeStream = BasicStream<ActiveBackend>;

    static_assert(sizeof(ComputeStream) == sizeof(typename ActiveBackend::StreamImpl),
                  "Zero-overhead violation: ComputeStream must match ActiveBackend::StreamImpl size.");
    static_assert(sizeof(CommandBatch) == sizeof(typename ActiveBackend::CommandListImpl),
                  "Zero-overhead violation: CommandBatch must match ActiveBackend::CommandListImpl size.");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_CORE_STREAM_H
