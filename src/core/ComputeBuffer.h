#ifndef FCT_COMPUTE_CORE_BUFFER_H
#define FCT_COMPUTE_CORE_BUFFER_H

#include "BackendConfig.h"

namespace FCT::Compute {

    /**
     * @brief 零开销类型安全 GPU 缓冲区视图模板（与编译期后端策略解耦）
     */
    template<BufferElementConcept T, ComputeBackendConcept Backend = ActiveBackend>
    class BasicBufferView {
    public:
        using ElementType        = T;
        using StorageElementType = StorageType_t<T>;
        using BackendType        = Backend;
        using NativeViewType     = typename Backend::template BufferViewImpl<StorageElementType>;

        constexpr BasicBufferView() noexcept = default;

        constexpr explicit BasicBufferView(NativeViewType view) noexcept
            : m_view(view) {}

        [[nodiscard]] inline bool valid() const noexcept {
            return Backend::BufferOps::isViewValid(m_view);
        }

        [[nodiscard]] inline explicit operator bool() const noexcept {
            return valid();
        }

        [[nodiscard]] inline size_t size() const noexcept {
            return Backend::BufferOps::viewSize(m_view);
        }

        [[nodiscard]] inline size_t byteSize() const noexcept {
            return Backend::BufferOps::viewByteSize(m_view);
        }

        [[nodiscard]] inline size_t offset() const noexcept {
            return Backend::BufferOps::viewOffset(m_view);
        }

        [[nodiscard]] inline size_t offsetBytes() const noexcept {
            return Backend::BufferOps::viewOffsetBytes(m_view);
        }

        [[nodiscard]] inline BasicBufferView subview(size_t offsetElements, size_t countElements) const noexcept {
            return BasicBufferView(Backend::BufferOps::subviewFromView(m_view, offsetElements, countElements));
        }

        [[nodiscard]] inline auto copyFromHost(const T* hostData) const noexcept {
            static_assert(IsLayoutIdentical_v<T, StorageElementType>,
                          "Host type T and StorageElementType must have identical layout for zero-copy upload.");
            return Backend::BufferOps::copyFromHost(m_view, hostData);
        }

        [[nodiscard]] inline auto copyFromHost(std::span<const T> hostSpan) const noexcept {
            return subview(0, hostSpan.size()).copyFromHost(hostSpan.data());
        }

        [[nodiscard]] inline auto copyToHost(T* hostData) const noexcept {
            static_assert(IsLayoutIdentical_v<T, StorageElementType>,
                          "Host type T and StorageElementType must have identical layout for zero-copy download.");
            return Backend::BufferOps::copyToHost(m_view, hostData);
        }

        [[nodiscard]] inline auto copyToHost(std::span<T> hostSpan) const noexcept {
            return subview(0, hostSpan.size()).copyToHost(hostSpan.data());
        }

        [[nodiscard]] inline auto copyFromBuffer(BasicBufferView<T, Backend> srcView) const noexcept {
            return Backend::BufferOps::copyFromView(m_view, srcView.backendImpl());
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
     * @brief 零开销 RAII GPU 缓冲区模板（与编译期后端策略解耦）
     */
    template<BufferElementConcept T, ComputeBackendConcept Backend = ActiveBackend>
    class BasicBuffer {
    public:
        using ElementType        = T;
        using StorageElementType = StorageType_t<T>;
        using BackendType        = Backend;
        using NativeBufferType   = typename Backend::template BufferImpl<StorageElementType>;
        using NativeViewType     = typename Backend::template BufferViewImpl<StorageElementType>;

        BasicBuffer() noexcept = default;

        explicit BasicBuffer(NativeBufferType&& buffer) noexcept
            : m_buffer(std::move(buffer)) {}

        BasicBuffer(BasicBuffer&&) noexcept = default;
        BasicBuffer& operator=(BasicBuffer&&) noexcept = default;
        BasicBuffer(const BasicBuffer&) = delete;
        BasicBuffer& operator=(const BasicBuffer&) = delete;

        [[nodiscard]] inline bool valid() const noexcept {
            return Backend::BufferOps::isValid(m_buffer);
        }

        [[nodiscard]] inline explicit operator bool() const noexcept {
            return valid();
        }

        [[nodiscard]] inline size_t size() const noexcept {
            return Backend::BufferOps::size(m_buffer);
        }

        [[nodiscard]] inline size_t byteSize() const noexcept {
            return Backend::BufferOps::byteSize(m_buffer);
        }

        [[nodiscard]] inline uint64_t handle() const noexcept {
            return Backend::BufferOps::handle(m_buffer);
        }

        [[nodiscard]] inline void* nativeHandle() const noexcept {
            return Backend::BufferOps::nativeHandle(m_buffer);
        }

        [[nodiscard]] inline BasicBufferView<T, Backend> view() const noexcept {
            return BasicBufferView<T, Backend>(Backend::BufferOps::toView(m_buffer));
        }

        [[nodiscard]] inline BasicBufferView<T, Backend> view(size_t offsetElements, size_t countElements) const noexcept {
            return BasicBufferView<T, Backend>(Backend::BufferOps::subview(m_buffer, offsetElements, countElements));
        }

        [[nodiscard]] inline operator BasicBufferView<T, Backend>() const noexcept {
            return view();
        }

        [[nodiscard]] inline operator NativeViewType() const noexcept {
            return Backend::BufferOps::toView(m_buffer);
        }

        [[nodiscard]] inline auto copyFromHost(const T* hostData) const noexcept {
            return view().copyFromHost(hostData);
        }

        [[nodiscard]] inline auto copyFromHost(std::span<const T> hostSpan) const noexcept {
            return view(0, hostSpan.size()).copyFromHost(hostSpan.data());
        }

        [[nodiscard]] inline auto copyToHost(T* hostData) const noexcept {
            return view().copyToHost(hostData);
        }

        [[nodiscard]] inline auto copyToHost(std::span<T> hostSpan) const noexcept {
            return view(0, hostSpan.size()).copyToHost(hostSpan.data());
        }

        [[nodiscard]] inline auto copyFromBuffer(BasicBufferView<T, Backend> srcView) const noexcept {
            return view().copyFromBuffer(srcView);
        }

        [[nodiscard]] inline auto copyFromBuffer(const BasicBuffer<T, Backend>& srcBuffer) const noexcept {
            return view().copyFromBuffer(srcBuffer.view());
        }

        [[nodiscard]] inline const NativeBufferType& backendImpl() const noexcept {
            return m_buffer;
        }

        [[nodiscard]] inline NativeBufferType& backendImpl() noexcept {
            return m_buffer;
        }

    private:
        NativeBufferType m_buffer{};
    };

    template<BufferElementConcept T>
    using BufferView = BasicBufferView<T, ActiveBackend>;

    template<BufferElementConcept T>
    using Buffer = BasicBuffer<T, ActiveBackend>;

    static_assert(sizeof(Buffer<float>) == sizeof(typename ActiveBackend::template BufferImpl<float>),
                  "Zero-overhead violation: Buffer<T> must match ActiveBackend::BufferImpl<T> size.");
    static_assert(alignof(Buffer<float>) == alignof(typename ActiveBackend::template BufferImpl<float>),
                  "Zero-overhead violation: Buffer<T> must match ActiveBackend::BufferImpl<T> alignment.");
    static_assert(sizeof(BufferView<float>) == sizeof(typename ActiveBackend::template BufferViewImpl<float>),
                  "Zero-overhead violation: BufferView<T> must match ActiveBackend::BufferViewImpl<T> size.");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_CORE_BUFFER_H
