#ifndef FCT_COMPUTE_KERNEL_H
#define FCT_COMPUTE_KERNEL_H

#include "ComputeDSL.h"

namespace FCT::Compute {

    /**
     * @brief 编译期线程块尺寸描述符
     */
    template<uint32_t X, uint32_t Y = 1u, uint32_t Z = 1u>
    struct BlockSize {
        static_assert(X > 0u && Y > 0u && Z > 0u, "BlockSize dimensions must be positive");
        static constexpr uint32_t x = X;
        static constexpr uint32_t y = Y;
        static constexpr uint32_t z = Z;
        static constexpr uint32_t totalThreads = X * Y * Z;

        [[nodiscard]] static constexpr luisa::uint3 toUint3() noexcept {
            return luisa::make_uint3(X, Y, Z);
        }
    };

    template<size_t Dim>
        requires KernelDimensionConcept<Dim>
    struct DefaultBlockSize;

    template<>
    struct DefaultBlockSize<1u> {
        using Type = BlockSize<256u, 1u, 1u>;
    };

    template<>
    struct DefaultBlockSize<2u> {
        using Type = BlockSize<16u, 16u, 1u>;
    };

    template<>
    struct DefaultBlockSize<3u> {
        using Type = BlockSize<8u, 8u, 8u>;
    };

    template<size_t Dim>
    using DefaultBlockSize_t = typename DefaultBlockSize<Dim>::Type;

    namespace Detail {

        template<typename T>
        [[nodiscard]] constexpr decltype(auto) NormalizeKernelArg(T&& arg) noexcept {
            using CleanT = std::remove_cvref_t<T>;
            if constexpr (std::is_same_v<CleanT, FCT::Vec3>) {
                return PackedFloat3(arg);
            } else {
                return std::forward<T>(arg);
            }
        }

        template<size_t Dim, typename BlockCfg, typename Signature>
        struct KernelBuilderImpl;

        template<size_t Dim, typename BlockCfg, typename... Args>
        struct KernelBuilderImpl<Dim, BlockCfg, void(Args...)> {
            template<typename Def>
            [[nodiscard]] static auto build(Def&& def) noexcept {
                return luisa::compute::Kernel<Dim, Args...>(
                    [func = std::forward<Def>(def)](luisa::compute::detail::prototype_to_creation_t<Args>... args) noexcept {
                        luisa::compute::set_block_size(BlockCfg::x, BlockCfg::y, BlockCfg::z);
                        func(std::forward<luisa::compute::detail::prototype_to_creation_t<Args>>(args)...);
                    });
            }
        };

    } // namespace Detail

    /**
     * @brief 编译期核函数签名萃取器
     */
    template<typename Func>
    struct KernelSignatureTraits {
        using Signature = luisa::compute::detail::dsl_function_t<std::remove_cvref_t<Func>>;
    };

    /**
     * @brief 使用编译期指定的维度与 BlockSize 构造 Luisa Kernel，无额外运行时包装层
     */
    template<size_t Dim, typename BlockCfg = DefaultBlockSize_t<Dim>, typename Def>
        requires KernelDimensionConcept<Dim>
    [[nodiscard]] auto MakeKernel(Def&& def) noexcept {
        using Sig = typename KernelSignatureTraits<Def>::Signature;
        return Detail::KernelBuilderImpl<Dim, BlockCfg, Sig>::build(std::forward<Def>(def));
    }

    template<uint32_t BlockX = 256u, typename Def>
    [[nodiscard]] auto MakeKernel1D(Def&& def) noexcept {
        return MakeKernel<1u, BlockSize<BlockX, 1u, 1u>>(std::forward<Def>(def));
    }

    template<uint32_t BlockX = 16u, uint32_t BlockY = 16u, typename Def>
    [[nodiscard]] auto MakeKernel2D(Def&& def) noexcept {
        return MakeKernel<2u, BlockSize<BlockX, BlockY, 1u>>(std::forward<Def>(def));
    }

    template<uint32_t BlockX = 8u, uint32_t BlockY = 8u, uint32_t BlockZ = 8u, typename Def>
    [[nodiscard]] auto MakeKernel3D(Def&& def) noexcept {
        return MakeKernel<3u, BlockSize<BlockX, BlockY, BlockZ>>(std::forward<Def>(def));
    }

    template<typename Def>
    [[nodiscard]] auto MakeCallable(Def&& def) noexcept {
        return luisa::compute::Callable(std::forward<Def>(def));
    }

    /**
     * @brief 零开销已编译计算着色器封装，内存大小与对齐严格等于 luisa::compute::Shader
     */
    template<size_t Dim, typename... Args>
        requires KernelDimensionConcept<Dim>
    class CompiledKernel {
    public:
        static constexpr size_t dimension = Dim;
        static constexpr size_t arity = sizeof...(Args);
        using NativeShaderType = luisa::compute::Shader<Dim, Args...>;

        CompiledKernel() noexcept = default;

        explicit CompiledKernel(NativeShaderType&& nativeShader) noexcept
            : m_shader(std::move(nativeShader)) {}

        CompiledKernel(CompiledKernel&&) noexcept = default;
        CompiledKernel& operator=(CompiledKernel&&) noexcept = default;
        CompiledKernel(const CompiledKernel&) = delete;
        CompiledKernel& operator=(const CompiledKernel&) = delete;
        ~CompiledKernel() noexcept = default;

        [[nodiscard]] explicit operator bool() const noexcept {
            return static_cast<bool>(m_shader);
        }

        [[nodiscard]] const NativeShaderType& raw() const noexcept {
            return m_shader;
        }

        [[nodiscard]] NativeShaderType& raw() noexcept {
            return m_shader;
        }

        [[nodiscard]] uint64_t handle() const noexcept {
            return m_shader.handle();
        }

        [[nodiscard]] luisa::uint3 blockSize() const noexcept {
            return m_shader.block_size();
        }

        [[nodiscard]] size_t uniformSize() const noexcept {
            return m_shader.uniform_size();
        }

        template<typename... CallArgs>
            requires (sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] auto operator()(CallArgs&&... args) const noexcept {
            return m_shader(Detail::NormalizeKernelArg(std::forward<CallArgs>(args))...);
        }

        template<typename... CallArgs>
            requires (Dim == 1u && sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] auto dispatch(uint32_t sizeX, CallArgs&&... args) const noexcept {
            return (*this)(std::forward<CallArgs>(args)...).dispatch(sizeX);
        }

        template<typename... CallArgs>
            requires (Dim == 2u && sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] auto dispatch(uint32_t sizeX, uint32_t sizeY, CallArgs&&... args) const noexcept {
            return (*this)(std::forward<CallArgs>(args)...).dispatch(sizeX, sizeY);
        }

        template<typename... CallArgs>
            requires (Dim == 2u && sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] auto dispatch(luisa::uint2 size, CallArgs&&... args) const noexcept {
            return (*this)(std::forward<CallArgs>(args)...).dispatch(size);
        }

        template<typename... CallArgs>
            requires (Dim == 3u && sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] auto dispatch(uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, CallArgs&&... args) const noexcept {
            return (*this)(std::forward<CallArgs>(args)...).dispatch(sizeX, sizeY, sizeZ);
        }

        template<typename... CallArgs>
            requires (Dim == 3u && sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] auto dispatch(luisa::uint3 size, CallArgs&&... args) const noexcept {
            return (*this)(std::forward<CallArgs>(args)...).dispatch(size);
        }

    private:
        NativeShaderType m_shader;
    };

    template<typename... Args>
    using CompiledKernel1D = CompiledKernel<1u, Args...>;

    template<typename... Args>
    using CompiledKernel2D = CompiledKernel<2u, Args...>;

    template<typename... Args>
    using CompiledKernel3D = CompiledKernel<3u, Args...>;

    template<size_t Dim, typename... Args>
    [[nodiscard]] CompiledKernel<Dim, Args...> ToCompiledKernel(
        luisa::compute::Shader<Dim, Args...>&& shader) noexcept {
        return CompiledKernel<Dim, Args...>(std::move(shader));
    }

    static_assert(sizeof(CompiledKernel1D<luisa::compute::Buffer<float>, float>) ==
                  sizeof(luisa::compute::Shader1D<luisa::compute::Buffer<float>, float>),
                  "Zero-overhead guarantee: CompiledKernel must match luisa::compute::Shader size");
    static_assert(alignof(CompiledKernel1D<luisa::compute::Buffer<float>, float>) ==
                  alignof(luisa::compute::Shader1D<luisa::compute::Buffer<float>, float>),
                  "Zero-overhead guarantee: CompiledKernel must match luisa::compute::Shader alignment");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_KERNEL_H
