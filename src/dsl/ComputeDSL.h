#ifndef FCT_COMPUTE_DSL_COMPUTE_DSL_H
#define FCT_COMPUTE_DSL_COMPUTE_DSL_H

#include "../core/ComputeBuffer.h"
#include "../core/ComputeImage.h"

namespace FCT::Compute::DSL {

    template<typename T, ComputeBackendConcept Backend = ActiveBackend>
    using BasicVar = typename Backend::template VarImpl<StorageType_t<T>>;

    template<typename T, ComputeBackendConcept Backend = ActiveBackend>
    using BasicExpr = typename Backend::template ExprImpl<StorageType_t<T>>;

    template<typename T>
    using Var = BasicVar<T, ActiveBackend>;

    template<typename T>
    using Expr = BasicExpr<T, ActiveBackend>;

    using Int      = Var<int32_t>;
    using Int2     = Var<FCT::Compute::Int2>;
    using Int3     = Var<FCT::Compute::Int3>;
    using Int4     = Var<FCT::Compute::Int4>;
    using UInt     = Var<uint32_t>;
    using UInt2    = Var<FCT::Compute::UInt2>;
    using UInt3    = Var<FCT::Compute::UInt3>;
    using UInt4    = Var<FCT::Compute::UInt4>;
    using Float    = Var<float>;
    using Float2   = Var<FCT::Compute::Float2>;
    using Float3   = Var<FCT::Compute::Float3>;
    using Float4   = Var<FCT::Compute::Float4>;
    using Bool     = Var<bool>;
    using Bool2    = Var<FCT::Compute::Bool2>;
    using Bool3    = Var<FCT::Compute::Bool3>;
    using Bool4    = Var<FCT::Compute::Bool4>;
    using Float2x2 = Var<FCT::Compute::Float2x2>;
    using Float3x3 = Var<FCT::Compute::Float3x3>;
    using Float4x4 = Var<FCT::Compute::Float4x4>;

    /**
     * @brief 零开销 GPU 端 Buffer 代理（与编译期后端策略解耦）
     */
    template<BufferElementConcept T, ComputeBackendConcept Backend = ActiveBackend>
    class BasicBufferVar {
    public:
        using ElementType        = T;
        using StorageElementType = StorageType_t<T>;
        using BackendType        = Backend;
        using NativeVarType      = typename Backend::template BufferVarImpl<StorageElementType>;

        explicit BasicBufferVar(NativeVarType&& native) noexcept
            : m_native(std::move(native)) {}

        template<typename NativeArg>
            requires (!std::same_as<std::remove_cvref_t<NativeArg>, BasicBufferVar> &&
                      std::constructible_from<NativeVarType, NativeArg>)
        BasicBufferVar(NativeArg&& arg) noexcept
            : m_native(std::forward<NativeArg>(arg)) {}

        template<typename IndexExpr>
        [[nodiscard]] inline auto read(IndexExpr&& index) const noexcept {
            return m_native.read(std::forward<IndexExpr>(index));
        }

        template<typename IndexExpr>
        [[nodiscard]] inline auto readVec(IndexExpr&& index) const noexcept {
            if constexpr (std::same_as<T, FCT::Vec2>) {
                return Backend::DSLOps::toFloat2(m_native.read(std::forward<IndexExpr>(index)));
            } else if constexpr (std::same_as<T, FCT::Vec3> || std::same_as<T, PackedFloat3>) {
                return Backend::DSLOps::toFloat3(m_native.read(std::forward<IndexExpr>(index)));
            } else if constexpr (std::same_as<T, FCT::Vec4>) {
                return Backend::DSLOps::toFloat4(m_native.read(std::forward<IndexExpr>(index)));
            } else {
                return m_native.read(std::forward<IndexExpr>(index));
            }
        }

        template<typename IndexExpr>
        [[nodiscard]] inline auto volatileRead(IndexExpr&& index) const noexcept {
            return m_native.volatile_read(std::forward<IndexExpr>(index));
        }

        template<typename IndexExpr, typename ValueExpr>
        inline void write(IndexExpr&& index, ValueExpr&& value) const noexcept {
            if constexpr ((std::same_as<T, FCT::Vec3> || std::same_as<T, PackedFloat3>) &&
                          std::convertible_to<ValueExpr, typename Backend::template ExprImpl<typename Backend::Float3>> &&
                          !std::convertible_to<ValueExpr, typename Backend::template ExprImpl<StorageElementType>>) {
                m_native.write(
                    std::forward<IndexExpr>(index),
                    Backend::DSLOps::fromFloat3(std::forward<ValueExpr>(value)));
            } else if constexpr (std::same_as<T, FCT::Vec2> &&
                                 std::convertible_to<ValueExpr, typename Backend::template ExprImpl<typename Backend::Float2>> &&
                                 !std::convertible_to<ValueExpr, typename Backend::template ExprImpl<StorageElementType>>) {
                m_native.write(
                    std::forward<IndexExpr>(index),
                    Backend::DSLOps::fromFloat2(std::forward<ValueExpr>(value)));
            } else if constexpr (std::same_as<T, FCT::Vec4> &&
                                 std::convertible_to<ValueExpr, typename Backend::template ExprImpl<typename Backend::Float4>> &&
                                 !std::convertible_to<ValueExpr, typename Backend::template ExprImpl<StorageElementType>>) {
                m_native.write(
                    std::forward<IndexExpr>(index),
                    Backend::DSLOps::fromFloat4(std::forward<ValueExpr>(value)));
            } else {
                m_native.write(std::forward<IndexExpr>(index), std::forward<ValueExpr>(value));
            }
        }

        template<typename IndexExpr, typename ValueExpr>
        inline void volatileWrite(IndexExpr&& index, ValueExpr&& value) const noexcept {
            m_native.volatile_write(std::forward<IndexExpr>(index), std::forward<ValueExpr>(value));
        }

        [[nodiscard]] inline auto size() const noexcept {
            return m_native.size();
        }

        [[nodiscard]] inline auto deviceAddress() const noexcept {
            return m_native.device_address();
        }

        template<typename IndexExpr>
        [[nodiscard]] inline auto atomic(IndexExpr&& index) const noexcept {
            return m_native.atomic(std::forward<IndexExpr>(index));
        }

        [[nodiscard]] constexpr const NativeVarType& backendImpl() const noexcept {
            return m_native;
        }

    private:
        NativeVarType m_native;
    };

    /**
     * @brief 零开销 GPU 端 2D 纹理代理（与编译期后端策略解耦）
     */
    template<ImageElementConcept T, ComputeBackendConcept Backend = ActiveBackend>
    class BasicImageVar {
    public:
        using ElementType   = T;
        using BackendType   = Backend;
        using NativeVarType = typename Backend::template ImageVarImpl<T>;

        explicit BasicImageVar(NativeVarType&& native) noexcept
            : m_native(std::move(native)) {}

        template<typename NativeArg>
            requires (!std::same_as<std::remove_cvref_t<NativeArg>, BasicImageVar> &&
                      std::constructible_from<NativeVarType, NativeArg>)
        BasicImageVar(NativeArg&& arg) noexcept
            : m_native(std::forward<NativeArg>(arg)) {}

        template<typename UVExpr>
        [[nodiscard]] inline auto read(UVExpr&& uv) const noexcept {
            return m_native.read(std::forward<UVExpr>(uv));
        }

        template<typename UVExpr, typename ValueExpr>
        inline void write(UVExpr&& uv, ValueExpr&& value) const noexcept {
            m_native.write(std::forward<UVExpr>(uv), std::forward<ValueExpr>(value));
        }

        [[nodiscard]] inline auto size() const noexcept {
            return m_native.size();
        }

        [[nodiscard]] constexpr const NativeVarType& backendImpl() const noexcept {
            return m_native;
        }

    private:
        NativeVarType m_native;
    };

    template<FCT::Format F, ComputeBackendConcept Backend = ActiveBackend>
        requires SupportedFormatConcept<F>
    class BasicFormatImageVar : public BasicImageVar<typename FormatTraits<F>::ElementType, Backend> {
    public:
        using Base = BasicImageVar<typename FormatTraits<F>::ElementType, Backend>;
        using Base::Base;
    };

    /**
     * @brief 零开销 GPU 端 3D 体纹理代理（与编译期后端策略解耦）
     */
    template<ImageElementConcept T, ComputeBackendConcept Backend = ActiveBackend>
    class BasicVolumeVar {
    public:
        using ElementType   = T;
        using BackendType   = Backend;
        using NativeVarType = typename Backend::template VolumeVarImpl<T>;

        explicit BasicVolumeVar(NativeVarType&& native) noexcept
            : m_native(std::move(native)) {}

        template<typename NativeArg>
            requires (!std::same_as<std::remove_cvref_t<NativeArg>, BasicVolumeVar> &&
                      std::constructible_from<NativeVarType, NativeArg>)
        BasicVolumeVar(NativeArg&& arg) noexcept
            : m_native(std::forward<NativeArg>(arg)) {}

        template<typename UVWExpr>
        [[nodiscard]] inline auto read(UVWExpr&& uvw) const noexcept {
            return m_native.read(std::forward<UVWExpr>(uvw));
        }

        template<typename UVWExpr, typename ValueExpr>
        inline void write(UVWExpr&& uvw, ValueExpr&& value) const noexcept {
            m_native.write(std::forward<UVWExpr>(uvw), std::forward<ValueExpr>(value));
        }

        [[nodiscard]] inline auto size() const noexcept {
            return m_native.size();
        }

        [[nodiscard]] constexpr const NativeVarType& backendImpl() const noexcept {
            return m_native;
        }

    private:
        NativeVarType m_native;
    };

    template<BufferElementConcept T>
    using BufferVar = BasicBufferVar<T, ActiveBackend>;

    template<ImageElementConcept T>
    using ImageVar = BasicImageVar<T, ActiveBackend>;

    template<FCT::Format F>
        requires SupportedFormatConcept<F>
    using FormatImageVar = BasicFormatImageVar<F, ActiveBackend>;

    template<ImageElementConcept T>
    using VolumeVar = BasicVolumeVar<T, ActiveBackend>;

    static_assert(sizeof(BufferVar<float>) == sizeof(typename ActiveBackend::template BufferVarImpl<float>),
                  "Zero-overhead violation: DSL::BufferVar<T> must match ActiveBackend::BufferVarImpl<T> size.");
    static_assert(sizeof(ImageVar<float>) == sizeof(typename ActiveBackend::template ImageVarImpl<float>),
                  "Zero-overhead violation: DSL::ImageVar<T> must match ActiveBackend::ImageVarImpl<T> size.");
    static_assert(sizeof(VolumeVar<float>) == sizeof(typename ActiveBackend::template VolumeVarImpl<float>),
                  "Zero-overhead violation: DSL::VolumeVar<T> must match ActiveBackend::VolumeVarImpl<T> size.");

    /**
     * @brief 零开销条件分支构建器（支持链式 .ElseIf(...) 与 .Else(...)）
     */
    template<ComputeBackendConcept Backend = ActiveBackend>
    class IfBranch {
    public:
        using NativeBuilder = typename Backend::IfBuilderImpl;

        explicit IfBranch(NativeBuilder&& builder) noexcept
            : m_builder(std::move(builder)) {}

        template<typename FalseFn>
        inline void Else(FalseFn&& falseFn) && noexcept {
            Backend::DSLOps::else_(std::move(m_builder), std::forward<FalseFn>(falseFn));
        }

        template<typename ElifCondExpr, typename ElifTrueFn>
        [[nodiscard]] inline IfBranch ElseIf(ElifCondExpr&& condExpr, ElifTrueFn&& trueFn) && noexcept {
            return IfBranch(Backend::DSLOps::elseIf(
                std::move(m_builder),
                [cond = std::forward<ElifCondExpr>(condExpr)] { return cond; },
                std::forward<ElifTrueFn>(trueFn)));
        }

    private:
        NativeBuilder m_builder;
    };

    template<typename CondExpr, typename TrueFn>
    inline IfBranch<ActiveBackend> If(CondExpr&& cond, TrueFn&& trueFn) noexcept {
        return IfBranch<ActiveBackend>(
            ActiveBackend::DSLOps::if_(std::forward<CondExpr>(cond), std::forward<TrueFn>(trueFn)));
    }

    template<typename BodyFn>
    inline void Loop(BodyFn&& bodyFn) noexcept {
        ActiveBackend::DSLOps::loop(std::forward<BodyFn>(bodyFn));
    }

    template<typename EndExpr, typename BodyFn>
    inline void For(EndExpr&& end, BodyFn&& bodyFn) noexcept {
        ActiveBackend::DSLOps::forRange(std::forward<EndExpr>(end), std::forward<BodyFn>(bodyFn));
    }

    template<typename BeginExpr, typename EndExpr, typename BodyFn>
    inline void For(BeginExpr&& begin, EndExpr&& end, BodyFn&& bodyFn) noexcept {
        ActiveBackend::DSLOps::forRange(
            std::forward<BeginExpr>(begin),
            std::forward<EndExpr>(end),
            std::forward<BodyFn>(bodyFn));
    }

    template<typename BeginExpr, typename EndExpr, typename StepExpr, typename BodyFn>
    inline void For(BeginExpr&& begin, EndExpr&& end, StepExpr&& step, BodyFn&& bodyFn) noexcept {
        ActiveBackend::DSLOps::forRange(
            std::forward<BeginExpr>(begin),
            std::forward<EndExpr>(end),
            std::forward<StepExpr>(step),
            std::forward<BodyFn>(bodyFn));
    }

    template<typename CondFn, typename BodyFn>
    inline void While(CondFn&& condFn, BodyFn&& bodyFn) noexcept {
        ActiveBackend::DSLOps::whileLoop(std::forward<CondFn>(condFn), std::forward<BodyFn>(bodyFn));
    }

    inline void Break() noexcept {
        ActiveBackend::DSLOps::break_();
    }

    inline void Continue() noexcept {
        ActiveBackend::DSLOps::continue_();
    }

    inline void Return() noexcept {
        ActiveBackend::DSLOps::return_();
    }

    template<typename ValExpr>
    inline void Return(ValExpr&& val) noexcept {
        ActiveBackend::DSLOps::return_(std::forward<ValExpr>(val));
    }

    template<typename Fmt, typename... Args>
    inline void DeviceLog(Fmt&& fmt, Args&&... args) noexcept {
        ActiveBackend::DSLOps::deviceLog(std::forward<Fmt>(fmt), std::forward<Args>(args)...);
    }

    [[nodiscard]] inline auto DispatchId() noexcept {
        return ActiveBackend::DSLOps::dispatchId();
    }

    [[nodiscard]] inline auto ThreadId() noexcept {
        return ActiveBackend::DSLOps::threadId();
    }

    [[nodiscard]] inline auto BlockId() noexcept {
        return ActiveBackend::DSLOps::blockId();
    }

    [[nodiscard]] inline auto DispatchSize() noexcept {
        return ActiveBackend::DSLOps::dispatchSize();
    }

    [[nodiscard]] inline auto KernelBlockSize() noexcept {
        return ActiveBackend::DSLOps::kernelBlockSize();
    }

    template<uint32_t X, uint32_t Y = 1u, uint32_t Z = 1u>
    inline void SetBlockSize() noexcept {
        ActiveBackend::DSLOps::setBlockSize(X, Y, Z);
    }

    inline void SynchronizeBlock() noexcept {
        ActiveBackend::DSLOps::synchronizeBlock();
    }

    template<typename T, typename Arg>
    [[nodiscard]] inline auto Def(Arg&& arg) noexcept {
        return ActiveBackend::DSLOps::template def<StorageType_t<T>>(std::forward<Arg>(arg));
    }

    template<typename Arg>
    [[nodiscard]] inline auto Def(Arg&& arg) noexcept {
        return ActiveBackend::DSLOps::defAuto(std::forward<Arg>(arg));
    }

    template<typename To, typename From>
    [[nodiscard]] inline auto Cast(From&& from) noexcept {
        return ActiveBackend::DSLOps::template cast<To>(std::forward<From>(from));
    }

    template<size_t Index, typename StructExpr>
    [[nodiscard]] inline decltype(auto) Field(StructExpr&& expr) noexcept {
        return ActiveBackend::DSLOps::template field<Index>(std::forward<StructExpr>(expr));
    }

    /**
     * @brief 编译期静态循环展开器（在 AST 构建期完全展开为线性无分支指令序列）
     */
    template<int64_t Begin, int64_t End, int64_t Step = 1, typename Func>
        requires (Step != 0)
    constexpr void StaticFor(Func&& func) noexcept {
        static_assert((Step > 0 && Begin <= End) || (Step < 0 && Begin >= End),
                      "Invalid StaticFor range bounds.");
        constexpr int64_t count = (Step > 0) ? ((End - Begin + Step - 1) / Step)
                                             : ((Begin - End - Step - 1) / (-Step));
        if constexpr (count > 0) {
            [&]<size_t... Is>(std::index_sequence<Is...>) noexcept {
                (func.template operator()<Begin + static_cast<int64_t>(Is) * Step>(), ...);
            }(std::make_index_sequence<static_cast<size_t>(count)>{});
        }
    }

    template<typename Func, typename... Args>
    constexpr void ForEachArg(Func&& func, Args&&... args) noexcept {
        (func(std::forward<Args>(args)), ...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeFloat2(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeFloat2(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeFloat3(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeFloat3(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeFloat4(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeFloat4(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeInt2(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeInt2(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeInt3(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeInt3(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeInt4(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeInt4(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeUInt2(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeUInt2(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeUInt3(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeUInt3(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeUInt4(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeUInt4(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeBool2(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeBool2(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeBool3(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeBool3(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeBool4(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeBool4(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeFloat2x2(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeFloat2x2(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeFloat3x3(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeFloat3x3(std::forward<Args>(args)...);
    }

    template<typename... Args>
    [[nodiscard]] inline auto MakeFloat4x4(Args&&... args) noexcept {
        return ActiveBackend::DSLOps::makeFloat4x4(std::forward<Args>(args)...);
    }

    [[nodiscard]] inline auto ToFloat2(Expr<FCT::Vec2> v) noexcept {
        return ActiveBackend::DSLOps::toFloat2(v);
    }

    [[nodiscard]] inline auto ToFloat3(Expr<PackedFloat3> v) noexcept {
        return ActiveBackend::DSLOps::toFloat3(v);
    }

    [[nodiscard]] inline auto ToFloat4(Expr<FCT::Vec4> v) noexcept {
        return ActiveBackend::DSLOps::toFloat4(v);
    }

    [[nodiscard]] inline Var<FCT::Vec2> FromFloat2(Expr<FCT::Compute::Float2> v) noexcept {
        return ActiveBackend::DSLOps::fromFloat2(v);
    }

    [[nodiscard]] inline Var<PackedFloat3> FromFloat3(Expr<FCT::Compute::Float3> v) noexcept {
        return ActiveBackend::DSLOps::fromFloat3(v);
    }

    [[nodiscard]] inline Var<FCT::Vec4> FromFloat4(Expr<FCT::Compute::Float4> v) noexcept {
        return ActiveBackend::DSLOps::fromFloat4(v);
    }

#define FCT_COMPUTE_DSL_UNARY_OP(PublicName, BackendOp)                        \
    template<typename X>                                                       \
    [[nodiscard]] inline auto PublicName(X&& x) noexcept {                     \
        return ActiveBackend::DSLOps::BackendOp(std::forward<X>(x));           \
    }

#define FCT_COMPUTE_DSL_BINARY_OP(PublicName, BackendOp)                       \
    template<typename X, typename Y>                                           \
    [[nodiscard]] inline auto PublicName(X&& x, Y&& y) noexcept {              \
        return ActiveBackend::DSLOps::BackendOp(                               \
            std::forward<X>(x), std::forward<Y>(y));                           \
    }

#define FCT_COMPUTE_DSL_TERNARY_OP(PublicName, BackendOp)                      \
    template<typename X, typename Y, typename Z>                               \
    [[nodiscard]] inline auto PublicName(X&& x, Y&& y, Z&& z) noexcept {       \
        return ActiveBackend::DSLOps::BackendOp(                               \
            std::forward<X>(x), std::forward<Y>(y), std::forward<Z>(z));       \
    }

    FCT_COMPUTE_DSL_UNARY_OP(Abs,           abs)
    FCT_COMPUTE_DSL_UNARY_OP(Sin,           sin)
    FCT_COMPUTE_DSL_UNARY_OP(Cos,           cos)
    FCT_COMPUTE_DSL_UNARY_OP(Tan,           tan)
    FCT_COMPUTE_DSL_UNARY_OP(Asin,          asin)
    FCT_COMPUTE_DSL_UNARY_OP(Acos,          acos)
    FCT_COMPUTE_DSL_UNARY_OP(Atan,          atan)
    FCT_COMPUTE_DSL_UNARY_OP(Sinh,          sinh)
    FCT_COMPUTE_DSL_UNARY_OP(Cosh,          cosh)
    FCT_COMPUTE_DSL_UNARY_OP(Tanh,          tanh)
    FCT_COMPUTE_DSL_UNARY_OP(Sqrt,          sqrt)
    FCT_COMPUTE_DSL_UNARY_OP(Rsqrt,         rsqrt)
    FCT_COMPUTE_DSL_UNARY_OP(Exp,           exp)
    FCT_COMPUTE_DSL_UNARY_OP(Exp2,          exp2)
    FCT_COMPUTE_DSL_UNARY_OP(Log,           log)
    FCT_COMPUTE_DSL_UNARY_OP(Log2,          log2)
    FCT_COMPUTE_DSL_UNARY_OP(Log10,         log10)
    FCT_COMPUTE_DSL_UNARY_OP(Floor,         floor)
    FCT_COMPUTE_DSL_UNARY_OP(Ceil,          ceil)
    FCT_COMPUTE_DSL_UNARY_OP(Round,         round)
    FCT_COMPUTE_DSL_UNARY_OP(Trunc,         trunc)
    FCT_COMPUTE_DSL_UNARY_OP(Fract,         fract)
    FCT_COMPUTE_DSL_UNARY_OP(Saturate,      saturate)
    FCT_COMPUTE_DSL_UNARY_OP(Length,        length)
    FCT_COMPUTE_DSL_UNARY_OP(LengthSquared, lengthSquared)
    FCT_COMPUTE_DSL_UNARY_OP(Normalize,     normalize)
    FCT_COMPUTE_DSL_UNARY_OP(Determinant,   determinant)
    FCT_COMPUTE_DSL_UNARY_OP(Transpose,     transpose)
    FCT_COMPUTE_DSL_UNARY_OP(Inverse,       inverse)
    FCT_COMPUTE_DSL_UNARY_OP(All,           all)
    FCT_COMPUTE_DSL_UNARY_OP(Any,           any)
    FCT_COMPUTE_DSL_UNARY_OP(None,          none)
    FCT_COMPUTE_DSL_UNARY_OP(IsNan,         isnan)
    FCT_COMPUTE_DSL_UNARY_OP(IsInf,         isinf)

    FCT_COMPUTE_DSL_BINARY_OP(Min,             min)
    FCT_COMPUTE_DSL_BINARY_OP(Max,             max)
    FCT_COMPUTE_DSL_BINARY_OP(Atan2,           atan2)
    FCT_COMPUTE_DSL_BINARY_OP(Pow,             pow)
    FCT_COMPUTE_DSL_BINARY_OP(Step,            step)
    FCT_COMPUTE_DSL_BINARY_OP(CopySign,        copysign)
    FCT_COMPUTE_DSL_BINARY_OP(Dot,             dot)
    FCT_COMPUTE_DSL_BINARY_OP(Cross,           cross)
    FCT_COMPUTE_DSL_BINARY_OP(Distance,        distance)
    FCT_COMPUTE_DSL_BINARY_OP(DistanceSquared, distanceSquared)
    FCT_COMPUTE_DSL_BINARY_OP(Reflect,         reflect)

    FCT_COMPUTE_DSL_TERNARY_OP(Clamp,       clamp)
    FCT_COMPUTE_DSL_TERNARY_OP(Lerp,        lerp)
    FCT_COMPUTE_DSL_TERNARY_OP(SmoothStep,  smoothstep)
    FCT_COMPUTE_DSL_TERNARY_OP(Fma,         fma)
    FCT_COMPUTE_DSL_TERNARY_OP(FaceForward, faceforward)
    FCT_COMPUTE_DSL_TERNARY_OP(Select,      select)

#undef FCT_COMPUTE_DSL_UNARY_OP
#undef FCT_COMPUTE_DSL_BINARY_OP
#undef FCT_COMPUTE_DSL_TERNARY_OP

} // namespace FCT::Compute::DSL

#endif // FCT_COMPUTE_DSL_COMPUTE_DSL_H
