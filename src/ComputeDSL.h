#ifndef FCT_COMPUTE_DSL_H
#define FCT_COMPUTE_DSL_H

#include "ComputeBuffer.h"
#include "ComputeImage.h"

namespace FCT::Compute {

    namespace DSL {

        using namespace luisa::compute;

        template<BufferElementConcept T>
        using TypedBufferVar = luisa::compute::BufferVar<StorageType_t<T>>;

        template<ImageElementConcept T>
        using TypedImageVar = luisa::compute::ImageVar<T>;

        template<FCT::Format F>
            requires SupportedFormatConcept<F>
        using FormatImageVar = luisa::compute::ImageVar<typename FormatTraits<F>::ElementType>;

        template<ImageElementConcept T>
        using TypedVolumeVar = luisa::compute::VolumeVar<T>;

        /**
         * @brief 编译期静态展开循环，在 AST 构建阶段完全展开，消除 GPU 分支指令开销
         */
        template<int64_t Begin, int64_t End, int64_t Step = 1, typename Func>
            requires ((Step > 0 && Begin <= End) || (Step < 0 && Begin >= End))
        constexpr void StaticFor(Func&& body) noexcept {
            if constexpr ((Step > 0 && Begin < End) || (Step < 0 && Begin > End)) {
                body(std::integral_constant<int64_t, Begin>{});
                StaticFor<Begin + Step, End, Step>(std::forward<Func>(body));
            }
        }

        /**
         * @brief 编译期参数包展开辅助器
         */
        template<typename Func, typename... Args>
        constexpr void ForEachArg(Func&& fn, Args&&... args) noexcept {
            (fn(std::forward<Args>(args)), ...);
        }

        [[nodiscard]] inline Var<luisa::float2> ToFloat2(Expr<FCT::Vec2> v) noexcept {
            return make_float2(v.x, v.y);
        }

        [[nodiscard]] inline Var<luisa::float3> ToFloat3(Expr<PackedFloat3> v) noexcept {
            return make_float3(v.x, v.y, v.z);
        }

        [[nodiscard]] inline Var<luisa::float4> ToFloat4(Expr<FCT::Vec4> v) noexcept {
            return make_float4(v.x, v.y, v.z, v.w);
        }

        [[nodiscard]] inline Var<FCT::Vec2> FromFloat2(Expr<luisa::float2> v) noexcept {
            Var<FCT::Vec2> out;
            out.x = v.x;
            out.y = v.y;
            return out;
        }

        [[nodiscard]] inline Var<PackedFloat3> FromFloat3(Expr<luisa::float3> v) noexcept {
            Var<PackedFloat3> out;
            out.x = v.x;
            out.y = v.y;
            out.z = v.z;
            return out;
        }

        [[nodiscard]] inline Var<FCT::Vec4> FromFloat4(Expr<luisa::float4> v) noexcept {
            Var<FCT::Vec4> out;
            out.x = v.x;
            out.y = v.y;
            out.z = v.z;
            out.w = v.w;
            return out;
        }

    } // namespace DSL

    using DSL::TypedBufferVar;
    using DSL::TypedImageVar;
    using DSL::FormatImageVar;
    using DSL::TypedVolumeVar;
    using DSL::StaticFor;
    using DSL::ForEachArg;

} // namespace FCT::Compute

#endif // FCT_COMPUTE_DSL_H
