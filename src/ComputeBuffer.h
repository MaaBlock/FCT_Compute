#ifndef FCT_COMPUTE_BUFFER_H
#define FCT_COMPUTE_BUFFER_H

#include "ComputeConcepts.h"

namespace FCT::Compute {

    template<BufferElementConcept T>
    class Buffer;

    /**
     * @brief 零开销类型的 GPU 缓冲区视图封装，与 luisa::compute::BufferView 内存布局完全等价
     */
    template<BufferElementConcept T>
    class BufferView {
    public:
        using ElementType = T;
        using StorageElementType = StorageType_t<T>;
        using NativeViewType = luisa::compute::BufferView<StorageElementType>;

        constexpr BufferView() noexcept = default;

        /* implicit */ BufferView(NativeViewType nativeView) noexcept
            : m_view(nativeView) {}

        [[nodiscard]] explicit operator bool() const noexcept {
            return static_cast<bool>(m_view);
        }

        [[nodiscard]] operator NativeViewType() const noexcept {
            return m_view;
        }

        [[nodiscard]] const NativeViewType& raw() const noexcept {
            return m_view;
        }

        [[nodiscard]] NativeViewType& raw() noexcept {
            return m_view;
        }

        [[nodiscard]] uint64_t handle() const noexcept {
            return m_view.handle();
        }

        [[nodiscard]] void* nativeHandle() const noexcept {
            return m_view.native_handle();
        }

        [[nodiscard]] size_t size() const noexcept {
            return m_view.size();
        }

        [[nodiscard]] constexpr size_t stride() const noexcept {
            return m_view.stride();
        }

        [[nodiscard]] size_t offset() const noexcept {
            return m_view.offset();
        }

        [[nodiscard]] size_t offsetBytes() const noexcept {
            return m_view.offset_bytes();
        }

        [[nodiscard]] size_t sizeBytes() const noexcept {
            return m_view.size_bytes();
        }

        [[nodiscard]] BufferView<T> subview(size_t offsetElements, size_t countElements) const noexcept {
            return BufferView<T>(m_view.subview(offsetElements, countElements));
        }

        template<BufferElementConcept U>
        [[nodiscard]] BufferView<U> as() const noexcept {
            return BufferView<U>(m_view.template as<StorageType_t<U>>());
        }

        [[nodiscard]] auto copyFrom(const void* hostData) const noexcept {
            return m_view.copy_from(hostData);
        }

        [[nodiscard]] auto copyFrom(std::span<const T> hostSpan) const noexcept {
            return m_view.copy_from(hostSpan.data());
        }

        [[nodiscard]] auto copyFrom(BufferView<T> srcView) const noexcept {
            return m_view.copy_from(srcView.m_view);
        }

        [[nodiscard]] auto copyTo(void* hostData) const noexcept {
            return m_view.copy_to(hostData);
        }

        [[nodiscard]] auto copyTo(std::span<T> hostSpan) const noexcept {
            return m_view.copy_to(hostSpan.data());
        }

        [[nodiscard]] auto copyTo(BufferView<T> dstView) const noexcept {
            return m_view.copy_to(dstView.m_view);
        }

        [[nodiscard]] auto operator->() const noexcept {
            return m_view.operator->();
        }

    private:
        NativeViewType m_view;
    };

    /**
     * @brief 零开销 RAII GPU 缓冲区封装，尺寸与对齐同 luisa::compute::Buffer 完全一致
     */
    template<BufferElementConcept T>
    class Buffer {
    public:
        using ElementType = T;
        using StorageElementType = StorageType_t<T>;
        using NativeBufferType = luisa::compute::Buffer<StorageElementType>;
        using ViewType = BufferView<T>;

        Buffer() noexcept = default;

        explicit Buffer(NativeBufferType&& nativeBuffer) noexcept
            : m_buffer(std::move(nativeBuffer)) {}

        Buffer(Buffer&&) noexcept = default;
        Buffer& operator=(Buffer&&) noexcept = default;
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        ~Buffer() noexcept = default;

        [[nodiscard]] explicit operator bool() const noexcept {
            return static_cast<bool>(m_buffer);
        }

        [[nodiscard]] operator const NativeBufferType&() const noexcept {
            return m_buffer;
        }

        [[nodiscard]] operator typename ViewType::NativeViewType() const noexcept {
            return m_buffer.view();
        }

        [[nodiscard]] operator ViewType() const noexcept {
            return ViewType(m_buffer.view());
        }

        [[nodiscard]] const NativeBufferType& raw() const noexcept {
            return m_buffer;
        }

        [[nodiscard]] NativeBufferType& raw() noexcept {
            return m_buffer;
        }

        [[nodiscard]] uint64_t handle() const noexcept {
            return m_buffer.handle();
        }

        [[nodiscard]] void* nativeHandle() const noexcept {
            return m_buffer.native_handle();
        }

        [[nodiscard]] size_t size() const noexcept {
            return m_buffer.size();
        }

        [[nodiscard]] constexpr size_t stride() const noexcept {
            return m_buffer.stride();
        }

        [[nodiscard]] size_t sizeBytes() const noexcept {
            return m_buffer.size_bytes();
        }

        [[nodiscard]] ViewType view() const noexcept {
            return ViewType(m_buffer.view());
        }

        [[nodiscard]] ViewType view(size_t offsetElements, size_t countElements) const noexcept {
            return ViewType(m_buffer.view(offsetElements, countElements));
        }

        [[nodiscard]] auto copyFrom(const void* hostData) const noexcept {
            return m_buffer.copy_from(hostData);
        }

        [[nodiscard]] auto copyFrom(std::span<const T> hostSpan) const noexcept {
            return m_buffer.copy_from(hostSpan.data());
        }

        [[nodiscard]] auto copyFrom(ViewType srcView) const noexcept {
            return m_buffer.copy_from(srcView.raw());
        }

        [[nodiscard]] auto copyTo(void* hostData) const noexcept {
            return m_buffer.copy_to(hostData);
        }

        [[nodiscard]] auto copyTo(std::span<T> hostSpan) const noexcept {
            return m_buffer.copy_to(hostSpan.data());
        }

        [[nodiscard]] auto copyTo(ViewType dstView) const noexcept {
            return m_buffer.copy_to(dstView.raw());
        }

        [[nodiscard]] auto operator->() const noexcept {
            return m_buffer.operator->();
        }

    private:
        NativeBufferType m_buffer;
    };

    static_assert(sizeof(Buffer<float>) == sizeof(luisa::compute::Buffer<float>),
                  "Zero-overhead guarantee: Buffer<T> must not add any bytes over luisa::compute::Buffer");
    static_assert(alignof(Buffer<float>) == alignof(luisa::compute::Buffer<float>),
                  "Zero-overhead guarantee: Buffer<T> alignment must match luisa::compute::Buffer");
    static_assert(sizeof(BufferView<float>) == sizeof(luisa::compute::BufferView<float>),
                  "Zero-overhead guarantee: BufferView<T> must not add any bytes over luisa::compute::BufferView");
    static_assert(sizeof(Buffer<FCT::Vec3>) == sizeof(luisa::compute::Buffer<PackedFloat3>),
                  "Zero-overhead guarantee: Buffer<FCT::Vec3> must match luisa::compute::Buffer<PackedFloat3>");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_BUFFER_H
