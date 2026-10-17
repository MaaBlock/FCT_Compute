#ifndef FCT_COMPUTE_DSL_COMPUTE_KERNEL_H
#define FCT_COMPUTE_DSL_COMPUTE_KERNEL_H

#include "ComputeDSL.h"

namespace FCT::Compute {

    /**
     * @brief 编译期线程块尺寸描述符
     */
    template<uint32_t X = 256u, uint32_t Y = 1u, uint32_t Z = 1u>
        requires (X >= 1u && Y >= 1u && Z >= 1u && (X * Y * Z <= 1024u))
    struct BlockSize {
        static constexpr uint32_t x = X;
        static constexpr uint32_t y = Y;
        static constexpr uint32_t z = Z;
        static constexpr uint32_t totalThreads = X * Y * Z;
    };

    template<size_t Dim>
        requires KernelDimensionConcept<Dim>
    struct DefaultBlockSize;

    template<>
    struct DefaultBlockSize<1> {
        using Type = BlockSize<256u, 1u, 1u>;
    };

    template<>
    struct DefaultBlockSize<2> {
        using Type = BlockSize<16u, 16u, 1u>;
    };

    template<>
    struct DefaultBlockSize<3> {
        using Type = BlockSize<8u, 8u, 8u>;
    };

    template<size_t Dim>
    using DefaultBlockSize_t = typename DefaultBlockSize<Dim>::Type;

    namespace Detail {

        /**
         * @brief 编译期 Kernel 形式参数到 Host 参数类型与 Backend 原生参数类型的双向映射器
         */
        template<typename ParamT, ComputeBackendConcept Backend>
        struct DSLParamMapping {
            using CleanType           = std::remove_cvref_t<ParamT>;
            using UnwrappedType       = typename Backend::template UnwrapDSLValue<CleanType>::Type;
            using HostArgType         = UnwrappedType;
            using NativeKernelArgType = StorageType_t<UnwrappedType>;

            template<typename NativeVar>
            [[nodiscard]] static constexpr decltype(auto) wrap(NativeVar&& v) noexcept {
                return std::forward<NativeVar>(v);
            }
        };

        template<typename T, ComputeBackendConcept Backend>
        struct DSLParamMapping<DSL::BasicBufferVar<T, Backend>, Backend> {
            using HostArgType         = BasicBuffer<T, Backend>;
            using NativeKernelArgType = typename Backend::template BufferImpl<StorageType_t<T>>;

            template<typename NativeVar>
            [[nodiscard]] static inline DSL::BasicBufferVar<T, Backend> wrap(NativeVar&& v) noexcept {
                return DSL::BasicBufferVar<T, Backend>(std::forward<NativeVar>(v));
            }
        };

        template<typename T, ComputeBackendConcept Backend>
        struct DSLParamMapping<DSL::BasicImageVar<T, Backend>, Backend> {
            using HostArgType         = BasicImage<T, Backend>;
            using NativeKernelArgType = typename Backend::template ImageImpl<T>;

            template<typename NativeVar>
            [[nodiscard]] static inline DSL::BasicImageVar<T, Backend> wrap(NativeVar&& v) noexcept {
                return DSL::BasicImageVar<T, Backend>(std::forward<NativeVar>(v));
            }
        };

        template<FCT::Format F, ComputeBackendConcept Backend>
        struct DSLParamMapping<DSL::BasicFormatImageVar<F, Backend>, Backend> {
            using ElemT               = typename FormatTraits<F>::ElementType;
            using HostArgType         = BasicFormatImage<F, Backend>;
            using NativeKernelArgType = typename Backend::template ImageImpl<ElemT>;

            template<typename NativeVar>
            [[nodiscard]] static inline DSL::BasicFormatImageVar<F, Backend> wrap(NativeVar&& v) noexcept {
                return DSL::BasicFormatImageVar<F, Backend>(std::forward<NativeVar>(v));
            }
        };

        template<typename T, ComputeBackendConcept Backend>
        struct DSLParamMapping<DSL::BasicVolumeVar<T, Backend>, Backend> {
            using HostArgType         = BasicVolume<T, Backend>;
            using NativeKernelArgType = typename Backend::template VolumeImpl<T>;

            template<typename NativeVar>
            [[nodiscard]] static inline DSL::BasicVolumeVar<T, Backend> wrap(NativeVar&& v) noexcept {
                return DSL::BasicVolumeVar<T, Backend>(std::forward<NativeVar>(v));
            }
        };

        template<typename CallableObj, ComputeBackendConcept Backend>
        struct MemberFunctionTraits;

        template<typename ClassType, typename ReturnType, typename Backend, typename... Params>
        struct MemberFunctionTraits<ReturnType(ClassType::*)(Params...) const, Backend> {
            using Return               = ReturnType;
            using ParamTuple           = std::tuple<std::remove_cvref_t<Params>...>;
            using HostArgTuple         = std::tuple<typename DSLParamMapping<std::remove_cvref_t<Params>, Backend>::HostArgType...>;
            using NativeKernelArgTuple = std::tuple<typename DSLParamMapping<std::remove_cvref_t<Params>, Backend>::NativeKernelArgType...>;
            static constexpr size_t arity = sizeof...(Params);

            template<size_t Dim, typename BlockCfg, typename Func>
            [[nodiscard]] static inline auto buildBackendKernel(Func&& userFunc) noexcept {
                using NativeKernel = typename Backend::template KernelImpl<
                    Dim,
                    typename DSLParamMapping<std::remove_cvref_t<Params>, Backend>::NativeKernelArgType...>;
                return NativeKernel(
                    [fn = std::forward<Func>(userFunc)](
                        typename Backend::template VarImpl<
                            typename DSLParamMapping<std::remove_cvref_t<Params>, Backend>::NativeKernelArgType>&&... nativeVars) mutable noexcept {
                        Backend::DSLOps::setBlockSize(BlockCfg::x, BlockCfg::y, BlockCfg::z);
                        fn(DSLParamMapping<std::remove_cvref_t<Params>, Backend>::wrap(std::move(nativeVars))...);
                    });
            }
        };

        template<typename ClassType, typename ReturnType, typename Backend, typename... Params>
        struct MemberFunctionTraits<ReturnType(ClassType::*)(Params...) const noexcept, Backend>
            : MemberFunctionTraits<ReturnType(ClassType::*)(Params...) const, Backend> {};

        template<typename ClassType, typename ReturnType, typename Backend, typename... Params>
        struct MemberFunctionTraits<ReturnType(ClassType::*)(Params...), Backend>
            : MemberFunctionTraits<ReturnType(ClassType::*)(Params...) const, Backend> {};

        template<typename ClassType, typename ReturnType, typename Backend, typename... Params>
        struct MemberFunctionTraits<ReturnType(ClassType::*)(Params...) noexcept, Backend>
            : MemberFunctionTraits<ReturnType(ClassType::*)(Params...) const, Backend> {};

        template<typename HostArg, ComputeBackendConcept Backend>
        struct HostArgToNativeShaderType {
            using Type = StorageType_t<HostArg>;
        };

        template<typename T, ComputeBackendConcept Backend>
        struct HostArgToNativeShaderType<BasicBuffer<T, Backend>, Backend> {
            using Type = typename Backend::template BufferImpl<StorageType_t<T>>;
        };

        template<typename T, ComputeBackendConcept Backend>
        struct HostArgToNativeShaderType<BasicBufferView<T, Backend>, Backend> {
            using Type = typename Backend::template BufferImpl<StorageType_t<T>>;
        };

        template<typename T, ComputeBackendConcept Backend>
        struct HostArgToNativeShaderType<BasicImage<T, Backend>, Backend> {
            using Type = typename Backend::template ImageImpl<T>;
        };

        template<typename T, ComputeBackendConcept Backend>
        struct HostArgToNativeShaderType<BasicImageView<T, Backend>, Backend> {
            using Type = typename Backend::template ImageImpl<T>;
        };

        template<FCT::Format F, ComputeBackendConcept Backend>
        struct HostArgToNativeShaderType<BasicFormatImage<F, Backend>, Backend> {
            using Type = typename Backend::template ImageImpl<typename FormatTraits<F>::ElementType>;
        };

        template<typename T, ComputeBackendConcept Backend>
        struct HostArgToNativeShaderType<BasicVolume<T, Backend>, Backend> {
            using Type = typename Backend::template VolumeImpl<T>;
        };

        template<typename T, ComputeBackendConcept Backend>
        struct HostArgToNativeShaderType<BasicVolumeView<T, Backend>, Backend> {
            using Type = typename Backend::template VolumeImpl<T>;
        };

        template<typename Arg, ComputeBackendConcept Backend>
        [[nodiscard]] constexpr decltype(auto) ConvertHostDispatchArg(Arg&& arg) noexcept {
            using CleanArg = std::remove_cvref_t<Arg>;
            if constexpr (requires { arg.backendImpl(); }) {
                return arg.backendImpl();
            } else if constexpr (std::same_as<CleanArg, FCT::Vec3>) {
                return PackedFloat3(arg);
            } else {
                return std::forward<Arg>(arg);
            }
        }

    } // namespace Detail

    /**
     * @brief 编译期 Kernel 签名反射特征
     */
    template<typename Func, ComputeBackendConcept Backend = ActiveBackend>
    struct KernelSignatureTraits
        : Detail::MemberFunctionTraits<decltype(&std::remove_cvref_t<Func>::operator()), Backend> {};

    template<typename ReturnType, ComputeBackendConcept Backend, typename... Params>
    struct KernelSignatureTraits<ReturnType(*)(Params...), Backend> {
        using Return               = ReturnType;
        using ParamTuple           = std::tuple<std::remove_cvref_t<Params>...>;
        using HostArgTuple         = std::tuple<typename Detail::DSLParamMapping<std::remove_cvref_t<Params>, Backend>::HostArgType...>;
        using NativeKernelArgTuple = std::tuple<typename Detail::DSLParamMapping<std::remove_cvref_t<Params>, Backend>::NativeKernelArgType...>;
        static constexpr size_t arity = sizeof...(Params);
    };

    /**
     * @brief 零开销计算 Kernel 静态定义封装（与编译期后端策略解耦）
     */
    template<size_t Dim, typename BlockCfg, typename Func, ComputeBackendConcept Backend = ActiveBackend>
        requires KernelDimensionConcept<Dim>
    class BasicKernelDef {
    public:
        using Traits           = KernelSignatureTraits<Func, Backend>;
        using HostArgTuple     = typename Traits::HostArgTuple;
        using NativeKernelType = decltype(Traits::template buildBackendKernel<Dim, BlockCfg>(std::declval<Func>()));

        explicit BasicKernelDef(Func&& func) noexcept
            : m_kernel(Traits::template buildBackendKernel<Dim, BlockCfg>(std::forward<Func>(func))) {}

        [[nodiscard]] constexpr const NativeKernelType& backendImpl() const noexcept {
            return m_kernel;
        }

        [[nodiscard]] constexpr NativeKernelType& backendImpl() noexcept {
            return m_kernel;
        }

    private:
        NativeKernelType m_kernel;
    };

    template<size_t Dim, typename BlockCfg = DefaultBlockSize_t<Dim>, typename Func>
        requires KernelDimensionConcept<Dim>
    [[nodiscard]] inline auto MakeKernel(Func&& func) noexcept {
        return BasicKernelDef<Dim, BlockCfg, std::remove_cvref_t<Func>, ActiveBackend>(std::forward<Func>(func));
    }

    template<typename BlockCfg = DefaultBlockSize_t<1>, typename Func>
    [[nodiscard]] inline auto MakeKernel1D(Func&& func) noexcept {
        return MakeKernel<1, BlockCfg>(std::forward<Func>(func));
    }

    template<typename BlockCfg = DefaultBlockSize_t<2>, typename Func>
    [[nodiscard]] inline auto MakeKernel2D(Func&& func) noexcept {
        return MakeKernel<2, BlockCfg>(std::forward<Func>(func));
    }

    template<typename BlockCfg = DefaultBlockSize_t<3>, typename Func>
    [[nodiscard]] inline auto MakeKernel3D(Func&& func) noexcept {
        return MakeKernel<3, BlockCfg>(std::forward<Func>(func));
    }

    /**
     * @brief 零开销已编译 GPU Kernel 句柄（与编译期后端策略解耦）
     */
    template<size_t Dim, ComputeBackendConcept Backend, typename... Args>
        requires KernelDimensionConcept<Dim>
    class BasicCompiledKernel {
    public:
        static constexpr size_t kDimension = Dim;
        using BackendType      = Backend;
        using NativeShaderType = typename Backend::template ShaderImpl<
            Dim,
            typename Detail::HostArgToNativeShaderType<Args, Backend>::Type...>;

        BasicCompiledKernel() noexcept = default;

        explicit BasicCompiledKernel(NativeShaderType&& shader) noexcept
            : m_shader(std::move(shader)) {}

        BasicCompiledKernel(BasicCompiledKernel&&) noexcept = default;
        BasicCompiledKernel& operator=(BasicCompiledKernel&&) noexcept = default;
        BasicCompiledKernel(const BasicCompiledKernel&) = delete;
        BasicCompiledKernel& operator=(const BasicCompiledKernel&) = delete;

        [[nodiscard]] inline bool valid() const noexcept {
            return Backend::ShaderOps::isValid(m_shader);
        }

        [[nodiscard]] inline explicit operator bool() const noexcept {
            return valid();
        }

        [[nodiscard]] inline uint64_t handle() const noexcept {
            return Backend::ShaderOps::handle(m_shader);
        }

        [[nodiscard]] inline void* nativeHandle() const noexcept {
            return Backend::ShaderOps::nativeHandle(m_shader);
        }

        template<typename... CallArgs>
            requires (sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] inline auto operator()(CallArgs&&... args) const noexcept {
            return Backend::ShaderOps::invoke(
                m_shader,
                Detail::ConvertHostDispatchArg<CallArgs, Backend>(std::forward<CallArgs>(args))...);
        }

        template<typename... CallArgs>
            requires (Dim == 1 && sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] inline auto dispatch1D(uint32_t sizeX, CallArgs&&... args) const noexcept {
            return Backend::ShaderOps::dispatch1D((*this)(std::forward<CallArgs>(args)...), sizeX);
        }

        template<typename... CallArgs>
            requires (Dim == 2 && sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] inline auto dispatch2D(uint32_t sizeX, uint32_t sizeY, CallArgs&&... args) const noexcept {
            return Backend::ShaderOps::dispatch2D((*this)(std::forward<CallArgs>(args)...), sizeX, sizeY);
        }

        template<typename... CallArgs>
            requires (Dim == 3 && sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] inline auto dispatch3D(
            uint32_t sizeX,
            uint32_t sizeY,
            uint32_t sizeZ,
            CallArgs&&... args) const noexcept {
            return Backend::ShaderOps::dispatch3D((*this)(std::forward<CallArgs>(args)...), sizeX, sizeY, sizeZ);
        }

        template<typename... CallArgs>
            requires (sizeof...(CallArgs) == sizeof...(Args))
        [[nodiscard]] inline auto dispatch(UInt3 gridSize, CallArgs&&... args) const noexcept {
            if constexpr (Dim == 1) {
                return Backend::ShaderOps::dispatch1D((*this)(std::forward<CallArgs>(args)...), gridSize.x);
            } else if constexpr (Dim == 2) {
                return Backend::ShaderOps::dispatch2D((*this)(std::forward<CallArgs>(args)...), gridSize.x, gridSize.y);
            } else {
                return Backend::ShaderOps::dispatch3D(
                    (*this)(std::forward<CallArgs>(args)...), gridSize.x, gridSize.y, gridSize.z);
            }
        }

        [[nodiscard]] inline const NativeShaderType& backendImpl() const noexcept {
            return m_shader;
        }

        [[nodiscard]] inline NativeShaderType& backendImpl() noexcept {
            return m_shader;
        }

    private:
        NativeShaderType m_shader{};
    };

    template<size_t Dim, typename... Args>
    using CompiledKernel = BasicCompiledKernel<Dim, ActiveBackend, Args...>;

    template<typename... Args>
    using CompiledKernel1D = CompiledKernel<1, Args...>;

    template<typename... Args>
    using CompiledKernel2D = CompiledKernel<2, Args...>;

    template<typename... Args>
    using CompiledKernel3D = CompiledKernel<3, Args...>;

    namespace Detail {

        template<size_t Dim, ComputeBackendConcept Backend, typename Tuple>
        struct CompiledKernelFromTuple;

        template<size_t Dim, ComputeBackendConcept Backend, typename... Args>
        struct CompiledKernelFromTuple<Dim, Backend, std::tuple<Args...>> {
            using Type = BasicCompiledKernel<Dim, Backend, Args...>;
        };

        template<size_t Dim, ComputeBackendConcept Backend, typename Tuple>
        using CompiledKernelFromTuple_t = typename CompiledKernelFromTuple<Dim, Backend, Tuple>::Type;

    } // namespace Detail

    static_assert(sizeof(CompiledKernel1D<Buffer<float>, float>) ==
                  sizeof(typename ActiveBackend::template ShaderImpl<1, typename ActiveBackend::template BufferImpl<float>, float>),
                  "Zero-overhead violation: CompiledKernel must match ActiveBackend::ShaderImpl size.");

} // namespace FCT::Compute

#endif // FCT_COMPUTE_DSL_COMPUTE_KERNEL_H
