#ifndef FCT_COMPUTE_CORE_BACKEND_CONCEPTS_H
#define FCT_COMPUTE_CORE_BACKEND_CONCEPTS_H

#include "ComputeConcepts.h"

namespace FCT::Compute {

    /**
     * @brief 编译期计算后端策略契约 Concept
     * Note: 所有接入 FCT_Compute 的计算后端（如 LuisaBackend 及未来同层级后端）均须满足此编译期契约，
     * 通过静态多态消除一切虚表与运行时间接层。
     */
    template<typename B>
    concept ComputeBackendConcept = requires {
        { B::backendName } -> std::convertible_to<std::string_view>;

        typename B::ContextImpl;
        typename B::DeviceImpl;
        typename B::StreamImpl;
        typename B::CommandListImpl;
        typename B::PixelStorageType;
        typename B::PixelFormatType;

        typename B::template BufferImpl<float>;
        typename B::template BufferViewImpl<float>;
        typename B::template ImageImpl<float>;
        typename B::template ImageViewImpl<float>;
        typename B::template VolumeImpl<float>;
        typename B::template VolumeViewImpl<float>;
        typename B::template ShaderImpl<1, typename B::template BufferImpl<float>, float>;

        typename B::Float2;
        typename B::Float3;
        typename B::Float4;
        typename B::Int2;
        typename B::Int3;
        typename B::Int4;
        typename B::UInt2;
        typename B::UInt3;
        typename B::UInt4;
        typename B::Bool2;
        typename B::Bool3;
        typename B::Bool4;
        typename B::Float2x2;
        typename B::Float3x3;
        typename B::Float4x4;

        typename B::template VarImpl<float>;
        typename B::template ExprImpl<float>;
        typename B::template UnwrapDSLValue<float>::Type;
        typename B::template BufferVarImpl<float>;
        typename B::template ImageVarImpl<float>;
        typename B::template VolumeVarImpl<float>;
        typename B::IfBuilderImpl;

        typename B::FormatOps;
        typename B::MathOps;
        typename B::ContextOps;
        typename B::DeviceOps;
        typename B::StreamOps;
        typename B::BufferOps;
        typename B::ImageOps;
        typename B::VolumeOps;
        typename B::ShaderOps;
        typename B::DSLOps;
    };

} // namespace FCT::Compute

#endif // FCT_COMPUTE_CORE_BACKEND_CONCEPTS_H
