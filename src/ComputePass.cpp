#include "FCT_Compute.h"

namespace FCT::Compute {

    namespace {

        struct SampleParticle {
            PackedFloat3 position;
            float mass;
            PackedFloat3 velocity;
            uint32_t flags;
        };

        static_assert(AggregateStructConcept<SampleParticle>);
        static_assert(AggregateTraits<SampleParticle>::memberCount == 4);
        static_assert(sizeof(Buffer<float>) == sizeof(luisa::compute::Buffer<float>));
        static_assert(sizeof(Buffer<FCT::Vec2>) == sizeof(luisa::compute::Buffer<FCT::Vec2>));
        static_assert(sizeof(Buffer<FCT::Vec3>) == sizeof(luisa::compute::Buffer<PackedFloat3>));
        static_assert(sizeof(Buffer<FCT::Vec4>) == sizeof(luisa::compute::Buffer<FCT::Vec4>));
        static_assert(sizeof(Buffer<SampleParticle>) == sizeof(luisa::compute::Buffer<SampleParticle>));
        static_assert(sizeof(Image<float>) == sizeof(luisa::compute::Image<float>));
        static_assert(sizeof(FormatImage<FCT::Format::R32G32B32A32_SFLOAT>) == sizeof(luisa::compute::Image<float>));
        static_assert(sizeof(Volume<float>) == sizeof(luisa::compute::Volume<float>));
        static_assert(sizeof(CompiledKernel1D<Buffer<FCT::Vec3>, float>) == sizeof(luisa::compute::Shader1D<luisa::compute::Buffer<PackedFloat3>, float>));

    } // namespace

    ComputePass::ComputePass(std::string name, ComputeDevice device, ComputeStream stream) noexcept
        : m_name(std::move(name)),
          m_device(std::move(device)),
          m_stream(std::move(stream)) {}

    Status ComputePass::dispatch() noexcept {
        if (!m_device || !m_stream) {
            return FailedPreconditionError(StrCat("ComputePass [", m_name, "] device or stream is not initialized."));
        }
        ComputePassContext passCtx(m_device, m_stream, nullptr);
        return execute(passCtx);
    }

    Status ComputePass::dispatch(FCT::ComputeExecutionContext& context) {
        if (!m_device || !m_stream) {
            return FailedPreconditionError(StrCat("ComputePass [", m_name, "] device or stream is not initialized."));
        }
        ComputePassContext passCtx(m_device, m_stream, &context);
        return execute(passCtx);
    }

} // namespace FCT::Compute
