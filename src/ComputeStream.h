#ifndef FCT_COMPUTE_STREAM_H
#define FCT_COMPUTE_STREAM_H

#include "ComputeKernel.h"

namespace FCT::Compute {

    /**
     * @brief 零开销命令批次构建器，严格等价于 luisa::compute::CommandList
     */
    class CommandBatch {
    public:
        CommandBatch() noexcept = default;
        explicit CommandBatch(luisa::compute::CommandList&& list) noexcept
            : m_list(std::move(list)) {}

        CommandBatch(CommandBatch&&) noexcept = default;
        CommandBatch& operator=(CommandBatch&&) noexcept = default;
        CommandBatch(const CommandBatch&) = delete;
        CommandBatch& operator=(const CommandBatch&) = delete;
        ~CommandBatch() noexcept = default;

        template<typename Cmd>
        CommandBatch& operator<<(Cmd&& cmd) noexcept {
            m_list << std::forward<Cmd>(cmd);
            return *this;
        }

        template<typename... Cmds>
            requires (sizeof...(Cmds) > 0u)
        CommandBatch& append(Cmds&&... cmds) noexcept {
            (m_list << ... << std::forward<Cmds>(cmds));
            return *this;
        }

        [[nodiscard]] bool empty() const noexcept {
            return m_list.empty();
        }

        void clear() noexcept {
            m_list.clear();
        }

        [[nodiscard]] auto commit() noexcept {
            return m_list.commit();
        }

        [[nodiscard]] const luisa::compute::CommandList& raw() const noexcept {
            return m_list;
        }

        [[nodiscard]] luisa::compute::CommandList& raw() noexcept {
            return m_list;
        }

    private:
        luisa::compute::CommandList m_list;
    };

    /**
     * @brief 零开销计算流封装，内存布局与 luisa::compute::Stream 完全一致，支持参数包折叠批量提交
     */
    class ComputeStream {
    public:
        ComputeStream() noexcept = default;

        explicit ComputeStream(luisa::compute::Stream&& stream) noexcept
            : m_stream(std::move(stream)) {}

        ComputeStream(ComputeStream&&) noexcept = default;
        ComputeStream& operator=(ComputeStream&&) noexcept = default;
        ComputeStream(const ComputeStream&) = delete;
        ComputeStream& operator=(const ComputeStream&) = delete;
        ~ComputeStream() noexcept = default;

        [[nodiscard]] explicit operator bool() const noexcept {
            return static_cast<bool>(m_stream);
        }

        [[nodiscard]] operator luisa::compute::Stream&() noexcept {
            return m_stream;
        }

        [[nodiscard]] operator const luisa::compute::Stream&() const noexcept {
            return m_stream;
        }

        [[nodiscard]] const luisa::compute::Stream& raw() const noexcept {
            return m_stream;
        }

        [[nodiscard]] luisa::compute::Stream& raw() noexcept {
            return m_stream;
        }

        [[nodiscard]] uint64_t handle() const noexcept {
            return m_stream.handle();
        }

        [[nodiscard]] void* nativeHandle() const noexcept {
            return m_stream.native_handle();
        }

        [[nodiscard]] luisa::compute::StreamTag streamTag() const noexcept {
            return m_stream.stream_tag();
        }

        template<typename Cmd>
        decltype(auto) operator<<(Cmd&& cmd) noexcept {
            return m_stream << std::forward<Cmd>(cmd);
        }

        /**
         * @brief 利用 C++20 折叠表达式在单次委托中连续提交多个命令并显式 commit
         */
        template<typename... Cmds>
            requires (sizeof...(Cmds) > 0u)
        ComputeStream& submit(Cmds&&... cmds) noexcept {
            ((m_stream << ... << std::forward<Cmds>(cmds)) << luisa::compute::commit());
            return *this;
        }

        /**
         * @brief 利用 C++20 折叠表达式在单次委托中连续提交多个命令并同步等待完成
         */
        template<typename... Cmds>
            requires (sizeof...(Cmds) > 0u)
        ComputeStream& submitAndSync(Cmds&&... cmds) noexcept {
            ((m_stream << ... << std::forward<Cmds>(cmds)) << luisa::compute::synchronize());
            return *this;
        }

        ComputeStream& submitBatch(CommandBatch& batch) noexcept {
            m_stream << batch.commit();
            return *this;
        }

        Status synchronize() noexcept {
            if (!m_stream) {
                return FailedPreconditionError("ComputeStream is not initialized.");
            }
            m_stream.synchronize();
            return OkStatus();
        }

    private:
        luisa::compute::Stream m_stream;
    };

    static_assert(sizeof(ComputeStream) == sizeof(luisa::compute::Stream),
                  "Zero-overhead guarantee: ComputeStream must match luisa::compute::Stream size");
    static_assert(alignof(ComputeStream) == alignof(luisa::compute::Stream),
                  "Zero-overhead guarantee: ComputeStream must match luisa::compute::Stream alignment");
    static_assert(sizeof(CommandBatch) == sizeof(luisa::compute::CommandList),
                  "Zero-overhead guarantee: CommandBatch must match luisa::compute::CommandList size");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_STREAM_H
