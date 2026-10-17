#ifndef FCT_COMPUTE_BACKENDS_LUISA_BACKEND_H
#define FCT_COMPUTE_BACKENDS_LUISA_BACKEND_H

#include "LuisaHeader.h"
#include "../../core/BackendConcepts.h"

namespace FCT::Compute {

    /**
     * @brief LuisaCompute 编译期静态后端策略实现
     * Note: 将所有 luisa::compute 原生类型与 DSL/Runtime 调用封装在静态策略表内，
     * 对上层前端提供 100% 零虚表、零间接寻址、零内存膨胀的编译期映射。
     */
    struct LuisaBackend {
        static constexpr std::string_view backendName = "LuisaCompute";

        using ContextImpl      = luisa::compute::Context;
        using DeviceImpl       = luisa::compute::Device;
        using StreamImpl       = luisa::compute::Stream;
        using CommandListImpl  = luisa::compute::CommandList;
        using PixelStorageType = luisa::compute::PixelStorage;
        using PixelFormatType  = luisa::compute::PixelFormat;

        template<typename StorageT>
        using BufferImpl = luisa::compute::Buffer<StorageT>;

        template<typename StorageT>
        using BufferViewImpl = luisa::compute::BufferView<StorageT>;

        template<typename StorageT>
        using ImageImpl = luisa::compute::Image<StorageT>;

        template<typename StorageT>
        using ImageViewImpl = luisa::compute::ImageView<StorageT>;

        template<typename StorageT>
        using VolumeImpl = luisa::compute::Volume<StorageT>;

        template<typename StorageT>
        using VolumeViewImpl = luisa::compute::VolumeView<StorageT>;

        template<size_t Dim, typename... NativeArgs>
        using ShaderImpl = luisa::compute::Shader<Dim, NativeArgs...>;

        template<size_t Dim, typename... NativeArgs>
        using KernelImpl = luisa::compute::Kernel<Dim, NativeArgs...>;

        template<typename T, size_t N>
        using Vector = luisa::Vector<T, N>;

        using Float2   = luisa::float2;
        using Float3   = luisa::float3;
        using Float4   = luisa::float4;
        using Int2     = luisa::int2;
        using Int3     = luisa::int3;
        using Int4     = luisa::int4;
        using UInt2    = luisa::uint2;
        using UInt3    = luisa::uint3;
        using UInt4    = luisa::uint4;
        using Bool2    = luisa::bool2;
        using Bool3    = luisa::bool3;
        using Bool4    = luisa::bool4;
        using Float2x2 = luisa::float2x2;
        using Float3x3 = luisa::float3x3;
        using Float4x4 = luisa::float4x4;

        template<typename T>
        using VarImpl = luisa::compute::Var<T>;

        template<typename T>
        using ExprImpl = luisa::compute::Expr<T>;

        template<typename ParamT>
        struct UnwrapDSLValue {
            using Type = std::remove_cvref_t<ParamT>;
        };

        template<typename T>
        struct UnwrapDSLValue<luisa::compute::Var<T>> {
            using Type = T;
        };

        template<typename T>
        struct UnwrapDSLValue<luisa::compute::Expr<T>> {
            using Type = T;
        };

        template<typename StorageT>
        using BufferVarImpl = luisa::compute::Expr<luisa::compute::Buffer<StorageT>>;

        template<typename StorageT>
        using ImageVarImpl = luisa::compute::Expr<luisa::compute::Image<StorageT>>;

        template<typename StorageT>
        using VolumeVarImpl = luisa::compute::Expr<luisa::compute::Volume<StorageT>>;

        using IfBuilderImpl = luisa::compute::detail::IfStmtBuilder;

        struct FormatOps {
            template<FCT::Format F>
                requires SupportedFormatConcept<F>
            [[nodiscard]] static constexpr PixelStorageType toPixelStorage() noexcept {
                using Traits = FormatTraits<F>;
                constexpr size_t ch = Traits::channels;
                switch (Traits::channelKind) {
                    case PixelChannelKind::Int8:
                        if constexpr (ch == 1) return PixelStorageType::BYTE1;
                        if constexpr (ch == 2) return PixelStorageType::BYTE2;
                        return PixelStorageType::BYTE4;
                    case PixelChannelKind::UInt8:
                    case PixelChannelKind::Norm8:
                        if constexpr (ch == 1) return PixelStorageType::BYTE1;
                        if constexpr (ch == 2) return PixelStorageType::BYTE2;
                        return PixelStorageType::BYTE4;
                    case PixelChannelKind::Int16:
                    case PixelChannelKind::UInt16:
                    case PixelChannelKind::Norm16:
                        if constexpr (ch == 1) return PixelStorageType::SHORT1;
                        if constexpr (ch == 2) return PixelStorageType::SHORT2;
                        return PixelStorageType::SHORT4;
                    case PixelChannelKind::Float16:
                        if constexpr (ch == 1) return PixelStorageType::HALF1;
                        if constexpr (ch == 2) return PixelStorageType::HALF2;
                        return PixelStorageType::HALF4;
                    case PixelChannelKind::Int32:
                    case PixelChannelKind::UInt32:
                        if constexpr (ch == 1) return PixelStorageType::INT1;
                        if constexpr (ch == 2) return PixelStorageType::INT2;
                        return PixelStorageType::INT4;
                    case PixelChannelKind::Float32:
                    default:
                        if constexpr (ch == 1) return PixelStorageType::FLOAT1;
                        if constexpr (ch == 2) return PixelStorageType::FLOAT2;
                        return PixelStorageType::FLOAT4;
                }
            }

            [[nodiscard]] static inline StatusOr<PixelFormatType> toPixelFormat(FCT::Format format) noexcept {
                switch (format) {
                    case FCT::Format::R8_SINT:              return PixelFormatType::R8SInt;
                    case FCT::Format::R8_UINT:              return PixelFormatType::R8UInt;
                    case FCT::Format::R8_UNORM:             return PixelFormatType::R8UNorm;
                    case FCT::Format::R8G8_SINT:            return PixelFormatType::RG8SInt;
                    case FCT::Format::R8G8_UINT:            return PixelFormatType::RG8UInt;
                    case FCT::Format::R8G8_UNORM:           return PixelFormatType::RG8UNorm;
                    case FCT::Format::R8G8B8A8_SINT:        return PixelFormatType::RGBA8SInt;
                    case FCT::Format::R8G8B8A8_UINT:        return PixelFormatType::RGBA8UInt;
                    case FCT::Format::R8G8B8A8_UNORM:
                    case FCT::Format::R8G8B8A8_SRGB:
                    case FCT::Format::B8G8R8A8_UNORM:
                    case FCT::Format::B8G8R8A8_SRGB:        return PixelFormatType::RGBA8UNorm;
                    case FCT::Format::R16_SINT:             return PixelFormatType::R16SInt;
                    case FCT::Format::R16_UINT:             return PixelFormatType::R16UInt;
                    case FCT::Format::R16_SFLOAT:           return PixelFormatType::R16F;
                    case FCT::Format::R16G16_SINT:          return PixelFormatType::RG16SInt;
                    case FCT::Format::R16G16_UINT:          return PixelFormatType::RG16UInt;
                    case FCT::Format::R16G16_SFLOAT:        return PixelFormatType::RG16F;
                    case FCT::Format::R16G16B16A16_SINT:    return PixelFormatType::RGBA16SInt;
                    case FCT::Format::R16G16B16A16_UINT:    return PixelFormatType::RGBA16UInt;
                    case FCT::Format::R16G16B16A16_SFLOAT:  return PixelFormatType::RGBA16F;
                    case FCT::Format::R32_SINT:             return PixelFormatType::R32SInt;
                    case FCT::Format::R32_UINT:             return PixelFormatType::R32UInt;
                    case FCT::Format::R32_SFLOAT:           return PixelFormatType::R32F;
                    case FCT::Format::R32G32_SINT:          return PixelFormatType::RG32SInt;
                    case FCT::Format::R32G32_UINT:          return PixelFormatType::RG32UInt;
                    case FCT::Format::R32G32_SFLOAT:        return PixelFormatType::RG32F;
                    case FCT::Format::R32G32B32A32_SINT:    return PixelFormatType::RGBA32SInt;
                    case FCT::Format::R32G32B32A32_UINT:    return PixelFormatType::RGBA32UInt;
                    case FCT::Format::R32G32B32A32_SFLOAT:  return PixelFormatType::RGBA32F;
                    default:
                        return InvalidArgumentError("Unsupported FCT::Format for LuisaBackend.");
                }
            }

            [[nodiscard]] static inline StatusOr<PixelStorageType> toPixelStorage(FCT::Format format) noexcept {
                auto pxFmtOr = toPixelFormat(format);
                CHECK_STATUS(pxFmtOr);
                return luisa::compute::pixel_format_to_storage(*pxFmtOr);
            }

            [[nodiscard]] static constexpr FCT::Format fromPixelFormat(PixelFormatType format) noexcept {
                switch (format) {
                    case PixelFormatType::R8SInt:      return FCT::Format::R8_SINT;
                    case PixelFormatType::R8UInt:      return FCT::Format::R8_UINT;
                    case PixelFormatType::R8UNorm:     return FCT::Format::R8_UNORM;
                    case PixelFormatType::RG8SInt:     return FCT::Format::R8G8_SINT;
                    case PixelFormatType::RG8UInt:     return FCT::Format::R8G8_UINT;
                    case PixelFormatType::RG8UNorm:    return FCT::Format::R8G8_UNORM;
                    case PixelFormatType::RGBA8SInt:   return FCT::Format::R8G8B8A8_SINT;
                    case PixelFormatType::RGBA8UInt:   return FCT::Format::R8G8B8A8_UINT;
                    case PixelFormatType::RGBA8UNorm:  return FCT::Format::R8G8B8A8_UNORM;
                    case PixelFormatType::R16SInt:     return FCT::Format::R16_SINT;
                    case PixelFormatType::R16UInt:     return FCT::Format::R16_UINT;
                    case PixelFormatType::R16F:        return FCT::Format::R16_SFLOAT;
                    case PixelFormatType::RG16SInt:    return FCT::Format::R16G16_SINT;
                    case PixelFormatType::RG16UInt:    return FCT::Format::R16G16_UINT;
                    case PixelFormatType::RG16F:       return FCT::Format::R16G16_SFLOAT;
                    case PixelFormatType::RGBA16SInt:  return FCT::Format::R16G16B16A16_SINT;
                    case PixelFormatType::RGBA16UInt:  return FCT::Format::R16G16B16A16_UINT;
                    case PixelFormatType::RGBA16F:     return FCT::Format::R16G16B16A16_SFLOAT;
                    case PixelFormatType::R32SInt:     return FCT::Format::R32_SINT;
                    case PixelFormatType::R32UInt:     return FCT::Format::R32_UINT;
                    case PixelFormatType::R32F:        return FCT::Format::R32_SFLOAT;
                    case PixelFormatType::RG32SInt:    return FCT::Format::R32G32_SINT;
                    case PixelFormatType::RG32UInt:    return FCT::Format::R32G32_UINT;
                    case PixelFormatType::RG32F:       return FCT::Format::R32G32_SFLOAT;
                    case PixelFormatType::RGBA32SInt:  return FCT::Format::R32G32B32A32_SINT;
                    case PixelFormatType::RGBA32UInt:  return FCT::Format::R32G32B32A32_UINT;
                    case PixelFormatType::RGBA32F:     return FCT::Format::R32G32B32A32_SFLOAT;
                    default:                           return FCT::Format::UNDEFINED;
                }
            }
        };

        struct MathOps {
            [[nodiscard]] static constexpr Float2 toBackendVec(const FCT::Vec2& v) noexcept {
                return luisa::make_float2(v.x, v.y);
            }
            [[nodiscard]] static constexpr Float3 toBackendVec(const FCT::Vec3& v) noexcept {
                return luisa::make_float3(v.x, v.y, v.z);
            }
            [[nodiscard]] static constexpr Float4 toBackendVec(const FCT::Vec4& v) noexcept {
                return luisa::make_float4(v.x, v.y, v.z, v.w);
            }
            [[nodiscard]] static inline FCT::Vec2 toHostVec(const Float2& v) noexcept {
                return FCT::Vec2(v.x, v.y);
            }
            [[nodiscard]] static inline FCT::Vec3 toHostVec(const Float3& v) noexcept {
                return FCT::Vec3(v.x, v.y, v.z);
            }
            [[nodiscard]] static inline FCT::Vec4 toHostVec(const Float4& v) noexcept {
                return FCT::Vec4(v.x, v.y, v.z, v.w);
            }
            [[nodiscard]] static constexpr Float4x4 toBackendMat4(const FCT::Mat4& mat) noexcept {
                return luisa::make_float4x4(
                    luisa::make_float4(mat.m[0], mat.m[4], mat.m[8],  mat.m[12]),
                    luisa::make_float4(mat.m[1], mat.m[5], mat.m[9],  mat.m[13]),
                    luisa::make_float4(mat.m[2], mat.m[6], mat.m[10], mat.m[14]),
                    luisa::make_float4(mat.m[3], mat.m[7], mat.m[11], mat.m[15]));
            }
            [[nodiscard]] static inline FCT::Mat4 toHostMat4(const Float4x4& mat) noexcept {
                return FCT::Mat4(
                    mat[0].x, mat[1].x, mat[2].x, mat[3].x,
                    mat[0].y, mat[1].y, mat[2].y, mat[3].y,
                    mat[0].z, mat[1].z, mat[2].z, mat[3].z,
                    mat[0].w, mat[1].w, mat[2].w, mat[3].w);
            }
        };

        struct ContextOps {
            [[nodiscard]] static inline ContextImpl create(const std::filesystem::path& runtimePath) noexcept {
                auto pathStr = runtimePath.string();
                return ContextImpl(std::string_view(pathStr));
            }
            [[nodiscard]] static inline ContextImpl create(const char* argv0) noexcept {
                return ContextImpl(argv0);
            }
            [[nodiscard]] static inline std::filesystem::path runtimeDirectory(const ContextImpl& ctx) noexcept {
                return ctx.runtime_directory();
            }
            [[nodiscard]] static inline DeviceImpl createDevice(ContextImpl& ctx, std::string_view backendNameStr, uint32_t deviceIndex) {
                if (deviceIndex == 0u) {
                    return ctx.create_device(backendNameStr);
                }
                luisa::compute::DeviceConfig config{};
                config.device_index = deviceIndex;
                return ctx.create_device(backendNameStr, &config);
            }
        };

        struct DeviceOps {
            [[nodiscard]] static inline bool isValid(const DeviceImpl& dev) noexcept {
                return static_cast<bool>(dev);
            }
            [[nodiscard]] static inline void* nativeHandle(const DeviceImpl& dev) noexcept {
                return dev.impl()->native_handle();
            }
            [[nodiscard]] static inline std::string_view backendName(const DeviceImpl& dev) noexcept {
                return dev.backend_name();
            }
            [[nodiscard]] static inline StreamImpl createStream(DeviceImpl& dev, bool forPresent) noexcept {
                return dev.create_stream(
                    forPresent ? luisa::compute::StreamTag::GRAPHICS : luisa::compute::StreamTag::COMPUTE);
            }
            template<typename StorageT>
            [[nodiscard]] static inline BufferImpl<StorageT> createBuffer(DeviceImpl& dev, size_t count) noexcept {
                return dev.create_buffer<StorageT>(count);
            }
            template<typename StorageT>
            [[nodiscard]] static inline BufferImpl<StorageT> importExternalBuffer(DeviceImpl& dev, void* nativeAddress, size_t count) noexcept {
                return dev.import_external_buffer<StorageT>(nativeAddress, count);
            }
            template<typename StorageT>
            [[nodiscard]] static inline ImageImpl<StorageT> createImage(
                DeviceImpl& dev,
                PixelStorageType storage,
                uint32_t width,
                uint32_t height,
                uint32_t mipLevels) noexcept {
                return dev.create_image<StorageT>(storage, width, height, mipLevels);
            }
            template<typename StorageT>
            [[nodiscard]] static inline ImageImpl<StorageT> importExternalImage(
                DeviceImpl& dev,
                uint64_t nativeHandle,
                PixelStorageType storage,
                uint32_t width,
                uint32_t height,
                uint32_t mipLevels) noexcept {
                return dev.import_external_image<StorageT>(nativeHandle, storage, width, height, mipLevels);
            }
            template<typename StorageT>
            [[nodiscard]] static inline VolumeImpl<StorageT> createVolume(
                DeviceImpl& dev,
                PixelStorageType storage,
                uint32_t width,
                uint32_t height,
                uint32_t depth,
                uint32_t mipLevels) noexcept {
                return dev.create_volume<StorageT>(storage, width, height, depth, mipLevels);
            }
            template<size_t Dim, typename... NativeArgs>
            [[nodiscard]] static inline ShaderImpl<Dim, NativeArgs...> compileKernel(
                DeviceImpl& dev,
                const KernelImpl<Dim, NativeArgs...>& kernel,
                const luisa::compute::ShaderOption& option = {}) noexcept {
                return dev.compile(kernel, option);
            }
        };

        struct StreamOps {
            [[nodiscard]] static inline bool isValid(const StreamImpl& stream) noexcept {
                return static_cast<bool>(stream);
            }
            [[nodiscard]] static inline void* nativeHandle(const StreamImpl& stream) noexcept {
                return stream.native_handle();
            }
            static inline void synchronize(StreamImpl& stream) noexcept {
                stream.synchronize();
            }
            [[nodiscard]] static inline CommandListImpl createCommandList() noexcept {
                return CommandListImpl::create();
            }
            template<typename Cmd>
            static inline void appendCommand(CommandListImpl& list, Cmd&& cmd) noexcept {
                list << std::forward<Cmd>(cmd);
            }
            static inline void appendCallback(CommandListImpl& list, std::function<void()> callback) noexcept {
                list.add_callback(std::move(callback));
            }
            [[nodiscard]] static inline bool isCommandListEmpty(const CommandListImpl& list) noexcept {
                return list.empty();
            }
            static inline void submitCommandList(StreamImpl& stream, CommandListImpl&& list) noexcept {
                stream << list.commit();
            }
            static inline void submitAndSyncCommandList(StreamImpl& stream, CommandListImpl&& list) noexcept {
                stream << list.commit() << luisa::compute::synchronize();
            }
            template<typename Cmd>
            static inline void submitSingle(StreamImpl& stream, Cmd&& cmd) noexcept {
                stream << std::forward<Cmd>(cmd);
            }
            template<typename Cmd>
            static inline void submitSingleAndSync(StreamImpl& stream, Cmd&& cmd) noexcept {
                stream << std::forward<Cmd>(cmd) << luisa::compute::synchronize();
            }
        };

        struct BufferOps {
            template<typename StorageT>
            [[nodiscard]] static inline bool isValid(const BufferImpl<StorageT>& buf) noexcept {
                return static_cast<bool>(buf);
            }
            template<typename StorageT>
            [[nodiscard]] static inline bool isViewValid(const BufferViewImpl<StorageT>& view) noexcept {
                return static_cast<bool>(view);
            }
            template<typename StorageT>
            [[nodiscard]] static inline size_t size(const BufferImpl<StorageT>& buf) noexcept {
                return buf.size();
            }
            template<typename StorageT>
            [[nodiscard]] static inline size_t viewSize(const BufferViewImpl<StorageT>& view) noexcept {
                return view.size();
            }
            template<typename StorageT>
            [[nodiscard]] static inline size_t byteSize(const BufferImpl<StorageT>& buf) noexcept {
                return buf.size_bytes();
            }
            template<typename StorageT>
            [[nodiscard]] static inline size_t viewByteSize(const BufferViewImpl<StorageT>& view) noexcept {
                return view.size_bytes();
            }
            template<typename StorageT>
            [[nodiscard]] static inline size_t viewOffset(const BufferViewImpl<StorageT>& view) noexcept {
                return view.offset();
            }
            template<typename StorageT>
            [[nodiscard]] static inline size_t viewOffsetBytes(const BufferViewImpl<StorageT>& view) noexcept {
                return view.offset_bytes();
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint64_t handle(const BufferImpl<StorageT>& buf) noexcept {
                return buf.handle();
            }
            template<typename StorageT>
            [[nodiscard]] static inline void* nativeHandle(const BufferImpl<StorageT>& buf) noexcept {
                return buf.native_handle();
            }
            template<typename StorageT>
            [[nodiscard]] static inline BufferViewImpl<StorageT> toView(const BufferImpl<StorageT>& buf) noexcept {
                return buf.view();
            }
            template<typename StorageT>
            [[nodiscard]] static inline BufferViewImpl<StorageT> subview(
                const BufferImpl<StorageT>& buf,
                size_t offset,
                size_t count) noexcept {
                return buf.view(offset, count);
            }
            template<typename StorageT>
            [[nodiscard]] static inline BufferViewImpl<StorageT> subviewFromView(
                const BufferViewImpl<StorageT>& view,
                size_t offset,
                size_t count) noexcept {
                return view.subview(offset, count);
            }
            template<typename StorageT>
            [[nodiscard]] static inline auto copyToHost(const BufferViewImpl<StorageT>& view, void* dst) noexcept {
                return view.copy_to(dst);
            }
            template<typename StorageT>
            [[nodiscard]] static inline auto copyFromHost(const BufferViewImpl<StorageT>& view, const void* src) noexcept {
                return view.copy_from(src);
            }
            template<typename StorageT>
            [[nodiscard]] static inline auto copyFromView(
                const BufferViewImpl<StorageT>& dst,
                const BufferViewImpl<StorageT>& src) noexcept {
                return dst.copy_from(src);
            }
        };

        struct ImageOps {
            template<typename StorageT>
            [[nodiscard]] static inline bool isValid(const ImageImpl<StorageT>& img) noexcept {
                return static_cast<bool>(img);
            }
            template<typename StorageT>
            [[nodiscard]] static inline bool isViewValid(const ImageViewImpl<StorageT>& view) noexcept {
                return static_cast<bool>(view);
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t width(const ImageImpl<StorageT>& img) noexcept {
                return img.size().x;
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t height(const ImageImpl<StorageT>& img) noexcept {
                return img.size().y;
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t viewWidth(const ImageViewImpl<StorageT>& view) noexcept {
                return view.size().x;
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t viewHeight(const ImageViewImpl<StorageT>& view) noexcept {
                return view.size().y;
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t viewLevel(const ImageViewImpl<StorageT>& view) noexcept {
                return view.level();
            }
            template<typename StorageT>
            [[nodiscard]] static inline size_t viewByteSize(const ImageViewImpl<StorageT>& view) noexcept {
                return view.size_bytes();
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t mipLevels(const ImageImpl<StorageT>& img) noexcept {
                return img.mip_levels();
            }
            template<typename StorageT>
            [[nodiscard]] static inline PixelStorageType storage(const ImageImpl<StorageT>& img) noexcept {
                return img.storage();
            }
            template<typename StorageT>
            [[nodiscard]] static inline PixelFormatType format(const ImageImpl<StorageT>& img) noexcept {
                return img.format();
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint64_t handle(const ImageImpl<StorageT>& img) noexcept {
                return img.handle();
            }
            template<typename StorageT>
            [[nodiscard]] static inline void* nativeHandle(const ImageImpl<StorageT>& img) noexcept {
                return img.native_handle();
            }
            template<typename StorageT>
            [[nodiscard]] static inline ImageViewImpl<StorageT> toView(const ImageImpl<StorageT>& img, uint32_t level = 0u) noexcept {
                return img.view(level);
            }
            template<typename StorageT>
            [[nodiscard]] static inline auto copyToHost(const ImageViewImpl<StorageT>& view, void* dst) noexcept {
                return view.copy_to(dst);
            }
            template<typename StorageT>
            [[nodiscard]] static inline auto copyFromHost(const ImageViewImpl<StorageT>& view, const void* src) noexcept {
                return view.copy_from(src);
            }
            template<typename StorageT>
            [[nodiscard]] static inline auto copyFromView(
                const ImageViewImpl<StorageT>& dst,
                const ImageViewImpl<StorageT>& src) noexcept {
                return dst.copy_from(src);
            }
        };

        struct VolumeOps {
            template<typename StorageT>
            [[nodiscard]] static inline bool isValid(const VolumeImpl<StorageT>& vol) noexcept {
                return static_cast<bool>(vol);
            }
            template<typename StorageT>
            [[nodiscard]] static inline bool isViewValid(const VolumeViewImpl<StorageT>& view) noexcept {
                return static_cast<bool>(view);
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t width(const VolumeImpl<StorageT>& vol) noexcept {
                return vol.size().x;
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t height(const VolumeImpl<StorageT>& vol) noexcept {
                return vol.size().y;
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t depth(const VolumeImpl<StorageT>& vol) noexcept {
                return vol.size().z;
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t viewWidth(const VolumeViewImpl<StorageT>& view) noexcept {
                return view.size().x;
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t viewHeight(const VolumeViewImpl<StorageT>& view) noexcept {
                return view.size().y;
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t viewDepth(const VolumeViewImpl<StorageT>& view) noexcept {
                return view.size().z;
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint32_t mipLevels(const VolumeImpl<StorageT>& vol) noexcept {
                return vol.mip_levels();
            }
            template<typename StorageT>
            [[nodiscard]] static inline PixelStorageType storage(const VolumeImpl<StorageT>& vol) noexcept {
                return vol.storage();
            }
            template<typename StorageT>
            [[nodiscard]] static inline PixelFormatType format(const VolumeImpl<StorageT>& vol) noexcept {
                return vol.format();
            }
            template<typename StorageT>
            [[nodiscard]] static inline uint64_t handle(const VolumeImpl<StorageT>& vol) noexcept {
                return vol.handle();
            }
            template<typename StorageT>
            [[nodiscard]] static inline void* nativeHandle(const VolumeImpl<StorageT>& vol) noexcept {
                return vol.native_handle();
            }
            template<typename StorageT>
            [[nodiscard]] static inline VolumeViewImpl<StorageT> toView(const VolumeImpl<StorageT>& vol, uint32_t level = 0u) noexcept {
                return vol.view(level);
            }
            template<typename StorageT>
            [[nodiscard]] static inline auto copyToHost(const VolumeViewImpl<StorageT>& view, void* dst) noexcept {
                return view.copy_to(dst);
            }
            template<typename StorageT>
            [[nodiscard]] static inline auto copyFromHost(const VolumeViewImpl<StorageT>& view, const void* src) noexcept {
                return view.copy_from(src);
            }
            template<typename StorageT>
            [[nodiscard]] static inline auto copyFromView(
                const VolumeViewImpl<StorageT>& dst,
                const VolumeViewImpl<StorageT>& src) noexcept {
                return dst.copy_from(src);
            }
        };

        struct ShaderOps {
            template<size_t Dim, typename... NativeArgs>
            [[nodiscard]] static inline bool isValid(const ShaderImpl<Dim, NativeArgs...>& shader) noexcept {
                return static_cast<bool>(shader);
            }
            template<size_t Dim, typename... NativeArgs>
            [[nodiscard]] static inline uint64_t handle(const ShaderImpl<Dim, NativeArgs...>& shader) noexcept {
                return shader.handle();
            }
            template<size_t Dim, typename... NativeArgs>
            [[nodiscard]] static inline void* nativeHandle(const ShaderImpl<Dim, NativeArgs...>& shader) noexcept {
                return shader.native_handle();
            }
            template<size_t Dim, typename... NativeArgs, typename... CallArgs>
            [[nodiscard]] static inline auto invoke(
                const ShaderImpl<Dim, NativeArgs...>& shader,
                CallArgs&&... args) noexcept {
                return shader(std::forward<CallArgs>(args)...);
            }
            template<typename DispatchCmd>
            [[nodiscard]] static inline auto dispatch1D(DispatchCmd&& cmd, uint32_t sizeX) noexcept {
                return std::forward<DispatchCmd>(cmd).dispatch(sizeX);
            }
            template<typename DispatchCmd>
            [[nodiscard]] static inline auto dispatch2D(DispatchCmd&& cmd, uint32_t sizeX, uint32_t sizeY) noexcept {
                return std::forward<DispatchCmd>(cmd).dispatch(sizeX, sizeY);
            }
            template<typename DispatchCmd>
            [[nodiscard]] static inline auto dispatch3D(
                DispatchCmd&& cmd,
                uint32_t sizeX,
                uint32_t sizeY,
                uint32_t sizeZ) noexcept {
                return std::forward<DispatchCmd>(cmd).dispatch(sizeX, sizeY, sizeZ);
            }
        };

        struct DSLOps {
            [[nodiscard]] static inline auto dispatchId() noexcept {
                return luisa::compute::dispatch_id();
            }
            [[nodiscard]] static inline auto threadId() noexcept {
                return luisa::compute::thread_id();
            }
            [[nodiscard]] static inline auto blockId() noexcept {
                return luisa::compute::block_id();
            }
            [[nodiscard]] static inline auto dispatchSize() noexcept {
                return luisa::compute::dispatch_size();
            }
            [[nodiscard]] static inline auto kernelBlockSize() noexcept {
                return luisa::compute::block_size();
            }
            static inline void setBlockSize(uint32_t x, uint32_t y, uint32_t z) noexcept {
                luisa::compute::set_block_size(x, y, z);
            }
            static inline void synchronizeBlock() noexcept {
                luisa::compute::sync_block();
            }

            template<typename T, typename Arg>
            [[nodiscard]] static inline auto def(Arg&& arg) noexcept {
                return luisa::compute::def<T>(std::forward<Arg>(arg));
            }
            template<typename Arg>
            [[nodiscard]] static inline auto defAuto(Arg&& arg) noexcept {
                return luisa::compute::def(std::forward<Arg>(arg));
            }
            template<typename To, typename From>
            [[nodiscard]] static inline auto cast(From&& from) noexcept {
                return luisa::compute::cast<To>(std::forward<From>(from));
            }

            template<size_t Index, typename ExprT>
            [[nodiscard]] static inline decltype(auto) field(ExprT&& expr) noexcept {
                return std::forward<ExprT>(expr).template get<Index>();
            }

            template<typename Cond, typename TrueFn>
            [[nodiscard]] static inline IfBuilderImpl if_(Cond&& cond, TrueFn&& trueFn) noexcept {
                return luisa::compute::if_(std::forward<Cond>(cond), std::forward<TrueFn>(trueFn));
            }
            template<typename FalseFn>
            static inline void else_(IfBuilderImpl&& builder, FalseFn&& falseFn) noexcept {
                std::move(builder).else_(std::forward<FalseFn>(falseFn));
            }
            template<typename ElifCondFn, typename ElifTrueFn>
            [[nodiscard]] static inline IfBuilderImpl elseIf(
                IfBuilderImpl&& builder,
                ElifCondFn&& condFn,
                ElifTrueFn&& trueFn) noexcept {
                return (std::move(builder) * std::forward<ElifCondFn>(condFn)) % std::forward<ElifTrueFn>(trueFn);
            }

            template<typename BodyFn>
            static inline void loop(BodyFn&& bodyFn) noexcept {
                luisa::compute::loop(std::forward<BodyFn>(bodyFn));
            }
            template<typename EndExpr, typename BodyFn>
            static inline void forRange(EndExpr&& end, BodyFn&& bodyFn) noexcept {
                luisa::compute::loop(std::forward<EndExpr>(end), std::forward<BodyFn>(bodyFn));
            }
            template<typename BeginExpr, typename EndExpr, typename BodyFn>
            static inline void forRange(BeginExpr&& begin, EndExpr&& end, BodyFn&& bodyFn) noexcept {
                luisa::compute::loop(
                    std::forward<BeginExpr>(begin),
                    std::forward<EndExpr>(end),
                    std::forward<BodyFn>(bodyFn));
            }
            template<typename BeginExpr, typename EndExpr, typename StepExpr, typename BodyFn>
            static inline void forRange(BeginExpr&& begin, EndExpr&& end, StepExpr&& step, BodyFn&& bodyFn) noexcept {
                luisa::compute::loop(
                    std::forward<BeginExpr>(begin),
                    std::forward<EndExpr>(end),
                    std::forward<StepExpr>(step),
                    std::forward<BodyFn>(bodyFn));
            }
            template<typename CondFn, typename BodyFn>
            static inline void whileLoop(CondFn&& condFn, BodyFn&& bodyFn) noexcept {
                luisa::compute::loop([&] {
                    luisa::compute::if_(!condFn(), [] {
                        luisa::compute::break_();
                    });
                    bodyFn();
                });
            }

            static inline void break_() noexcept {
                luisa::compute::break_();
            }
            static inline void continue_() noexcept {
                luisa::compute::continue_();
            }
            static inline void return_() noexcept {
                luisa::compute::return_();
            }
            template<typename T>
            static inline void return_(T&& val) noexcept {
                luisa::compute::return_(std::forward<T>(val));
            }
            template<typename Fmt, typename... Args>
            static inline void deviceLog(Fmt&& fmt, Args&&... args) noexcept {
                luisa::compute::device_log(std::forward<Fmt>(fmt), std::forward<Args>(args)...);
            }

            template<typename... Args>
            [[nodiscard]] static inline auto makeFloat2(Args&&... args) noexcept {
                return luisa::compute::make_float2(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeFloat3(Args&&... args) noexcept {
                return luisa::compute::make_float3(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeFloat4(Args&&... args) noexcept {
                return luisa::compute::make_float4(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeInt2(Args&&... args) noexcept {
                return luisa::compute::make_int2(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeInt3(Args&&... args) noexcept {
                return luisa::compute::make_int3(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeInt4(Args&&... args) noexcept {
                return luisa::compute::make_int4(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeUInt2(Args&&... args) noexcept {
                return luisa::compute::make_uint2(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeUInt3(Args&&... args) noexcept {
                return luisa::compute::make_uint3(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeUInt4(Args&&... args) noexcept {
                return luisa::compute::make_uint4(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeBool2(Args&&... args) noexcept {
                return luisa::compute::make_bool2(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeBool3(Args&&... args) noexcept {
                return luisa::compute::make_bool3(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeBool4(Args&&... args) noexcept {
                return luisa::compute::make_bool4(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeFloat2x2(Args&&... args) noexcept {
                return luisa::compute::make_float2x2(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeFloat3x3(Args&&... args) noexcept {
                return luisa::compute::make_float3x3(std::forward<Args>(args)...);
            }
            template<typename... Args>
            [[nodiscard]] static inline auto makeFloat4x4(Args&&... args) noexcept {
                return luisa::compute::make_float4x4(std::forward<Args>(args)...);
            }

            [[nodiscard]] static inline auto toFloat2(ExprImpl<FCT::Vec2> v) noexcept {
                return luisa::compute::make_float2(v.x, v.y);
            }
            [[nodiscard]] static inline auto toFloat3(ExprImpl<PackedFloat3> v) noexcept {
                return luisa::compute::make_float3(v.x, v.y, v.z);
            }
            [[nodiscard]] static inline auto toFloat4(ExprImpl<FCT::Vec4> v) noexcept {
                return luisa::compute::make_float4(v.x, v.y, v.z, v.w);
            }
            [[nodiscard]] static inline VarImpl<FCT::Vec2> fromFloat2(ExprImpl<Float2> v) noexcept {
                VarImpl<FCT::Vec2> res;
                res.x = v.x;
                res.y = v.y;
                return res;
            }
            [[nodiscard]] static inline VarImpl<PackedFloat3> fromFloat3(ExprImpl<Float3> v) noexcept {
                VarImpl<PackedFloat3> res;
                res.x = v.x;
                res.y = v.y;
                res.z = v.z;
                return res;
            }
            [[nodiscard]] static inline VarImpl<FCT::Vec4> fromFloat4(ExprImpl<Float4> v) noexcept {
                VarImpl<FCT::Vec4> res;
                res.x = v.x;
                res.y = v.y;
                res.z = v.z;
                res.w = v.w;
                return res;
            }

#define FCT_COMPUTE_LUISA_UNARY_MATH(Name, LuisaFn)                            \
            template<typename X>                                                \
            [[nodiscard]] static inline auto Name(X&& x) noexcept {             \
                return luisa::compute::LuisaFn(std::forward<X>(x));             \
            }

#define FCT_COMPUTE_LUISA_BINARY_MATH(Name, LuisaFn)                           \
            template<typename X, typename Y>                                    \
            [[nodiscard]] static inline auto Name(X&& x, Y&& y) noexcept {      \
                return luisa::compute::LuisaFn(                                 \
                    std::forward<X>(x), std::forward<Y>(y));                    \
            }

#define FCT_COMPUTE_LUISA_TERNARY_MATH(Name, LuisaFn)                          \
            template<typename X, typename Y, typename Z>                        \
            [[nodiscard]] static inline auto Name(                              \
                X&& x, Y&& y, Z&& z) noexcept {                                 \
                return luisa::compute::LuisaFn(                                 \
                    std::forward<X>(x),                                         \
                    std::forward<Y>(y),                                         \
                    std::forward<Z>(z));                                        \
            }

            FCT_COMPUTE_LUISA_UNARY_MATH(abs, abs)
            FCT_COMPUTE_LUISA_UNARY_MATH(sin, sin)
            FCT_COMPUTE_LUISA_UNARY_MATH(cos, cos)
            FCT_COMPUTE_LUISA_UNARY_MATH(tan, tan)
            FCT_COMPUTE_LUISA_UNARY_MATH(asin, asin)
            FCT_COMPUTE_LUISA_UNARY_MATH(acos, acos)
            FCT_COMPUTE_LUISA_UNARY_MATH(atan, atan)
            FCT_COMPUTE_LUISA_UNARY_MATH(sinh, sinh)
            FCT_COMPUTE_LUISA_UNARY_MATH(cosh, cosh)
            FCT_COMPUTE_LUISA_UNARY_MATH(tanh, tanh)
            FCT_COMPUTE_LUISA_UNARY_MATH(sqrt, sqrt)
            FCT_COMPUTE_LUISA_UNARY_MATH(rsqrt, rsqrt)
            FCT_COMPUTE_LUISA_UNARY_MATH(exp, exp)
            FCT_COMPUTE_LUISA_UNARY_MATH(exp2, exp2)
            FCT_COMPUTE_LUISA_UNARY_MATH(log, log)
            FCT_COMPUTE_LUISA_UNARY_MATH(log2, log2)
            FCT_COMPUTE_LUISA_UNARY_MATH(log10, log10)
            FCT_COMPUTE_LUISA_UNARY_MATH(floor, floor)
            FCT_COMPUTE_LUISA_UNARY_MATH(ceil, ceil)
            FCT_COMPUTE_LUISA_UNARY_MATH(round, round)
            FCT_COMPUTE_LUISA_UNARY_MATH(trunc, trunc)
            FCT_COMPUTE_LUISA_UNARY_MATH(fract, fract)
            FCT_COMPUTE_LUISA_UNARY_MATH(saturate, saturate)
            FCT_COMPUTE_LUISA_UNARY_MATH(length, length)
            FCT_COMPUTE_LUISA_UNARY_MATH(lengthSquared, length_squared)
            FCT_COMPUTE_LUISA_UNARY_MATH(normalize, normalize)
            FCT_COMPUTE_LUISA_UNARY_MATH(determinant, determinant)
            FCT_COMPUTE_LUISA_UNARY_MATH(transpose, transpose)
            FCT_COMPUTE_LUISA_UNARY_MATH(inverse, inverse)
            FCT_COMPUTE_LUISA_UNARY_MATH(all, all)
            FCT_COMPUTE_LUISA_UNARY_MATH(any, any)
            FCT_COMPUTE_LUISA_UNARY_MATH(none, none)
            FCT_COMPUTE_LUISA_UNARY_MATH(isnan, isnan)
            FCT_COMPUTE_LUISA_UNARY_MATH(isinf, isinf)

            FCT_COMPUTE_LUISA_BINARY_MATH(min, min)
            FCT_COMPUTE_LUISA_BINARY_MATH(max, max)
            FCT_COMPUTE_LUISA_BINARY_MATH(atan2, atan2)
            FCT_COMPUTE_LUISA_BINARY_MATH(pow, pow)
            FCT_COMPUTE_LUISA_BINARY_MATH(step, step)
            FCT_COMPUTE_LUISA_BINARY_MATH(copysign, copysign)
            FCT_COMPUTE_LUISA_BINARY_MATH(dot, dot)
            FCT_COMPUTE_LUISA_BINARY_MATH(cross, cross)
            FCT_COMPUTE_LUISA_BINARY_MATH(distance, distance)
            FCT_COMPUTE_LUISA_BINARY_MATH(distanceSquared, distance_squared)
            FCT_COMPUTE_LUISA_BINARY_MATH(reflect, reflect)

            FCT_COMPUTE_LUISA_TERNARY_MATH(clamp, clamp)
            FCT_COMPUTE_LUISA_TERNARY_MATH(lerp, lerp)
            FCT_COMPUTE_LUISA_TERNARY_MATH(smoothstep, smoothstep)
            FCT_COMPUTE_LUISA_TERNARY_MATH(fma, fma)
            FCT_COMPUTE_LUISA_TERNARY_MATH(faceforward, faceforward)
            FCT_COMPUTE_LUISA_TERNARY_MATH(select, select)

#undef FCT_COMPUTE_LUISA_UNARY_MATH
#undef FCT_COMPUTE_LUISA_BINARY_MATH
#undef FCT_COMPUTE_LUISA_TERNARY_MATH
        };
    };

    static_assert(ComputeBackendConcept<LuisaBackend>,
                  "LuisaBackend must satisfy ComputeBackendConcept at compile time.");

} // namespace FCT::Compute

#define FCT_COMPUTE_BACKEND_REFLECT_STRUCT(StructType, ...) \
    LUISA_STRUCT(StructType, __VA_ARGS__) {}

#endif // FCT_COMPUTE_BACKENDS_LUISA_BACKEND_H
