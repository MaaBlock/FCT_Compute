#include "../FCT_Compute.h"

namespace FCT::Compute {

    namespace {

        struct SampleParticle {
            PackedFloat3 position;
            float mass;
            PackedFloat3 velocity;
            uint32_t flags;
        };

        static_assert(ComputeBackendConcept<ActiveBackend>);
        static_assert(AggregateStructConcept<SampleParticle>);
        static_assert(AggregateTraits<SampleParticle>::memberCount == 4);
        static_assert(sizeof(Buffer<float>) == sizeof(typename ActiveBackend::template BufferImpl<float>));
        static_assert(sizeof(Buffer<FCT::Vec2>) == sizeof(typename ActiveBackend::template BufferImpl<FCT::Vec2>));
        static_assert(sizeof(Buffer<FCT::Vec3>) == sizeof(typename ActiveBackend::template BufferImpl<PackedFloat3>));
        static_assert(sizeof(Buffer<FCT::Vec4>) == sizeof(typename ActiveBackend::template BufferImpl<FCT::Vec4>));
        static_assert(sizeof(Buffer<SampleParticle>) == sizeof(typename ActiveBackend::template BufferImpl<SampleParticle>));
        static_assert(sizeof(Image<float>) == sizeof(typename ActiveBackend::template ImageImpl<float>));
        static_assert(sizeof(FormatImage<FCT::Format::R32G32B32A32_SFLOAT>) == sizeof(typename ActiveBackend::template ImageImpl<float>));
        static_assert(sizeof(Volume<float>) == sizeof(typename ActiveBackend::template VolumeImpl<float>));
        static_assert(sizeof(CompiledKernel1D<Buffer<FCT::Vec3>, float>) ==
                      sizeof(typename ActiveBackend::template ShaderImpl<1, typename ActiveBackend::template BufferImpl<PackedFloat3>, float>));

        [[maybe_unused]] void VerifyBackendAgnosticDSLCompilation(ComputeDevice& dev) {
            auto kernel1D = MakeKernel1D<BlockSize<256>>(
                [](DSL::BufferVar<FCT::Vec3> positions, DSL::BufferVar<float> weights, DSL::Float dt) {
                    DSL::UInt idx = DSL::DispatchId().x;
                    DSL::Float w  = weights.read(idx);
                    DSL::If(w > 0.0f, [&] {
                        auto pos = positions.readVec(idx);
                        auto dir = DSL::Normalize(pos + DSL::MakeFloat3(1.0f, 0.0f, 0.0f));
                        positions.write(idx, pos + dir * (dt * w));
                    }).Else([&] {
                        positions.write(idx, DSL::MakeFloat3(0.0f, 0.0f, 0.0f));
                    });
                });

            auto compiled = dev.compile(kernel1D);
            static_assert(std::same_as<decltype(compiled), CompiledKernel1D<Buffer<FCT::Vec3>, Buffer<float>, float>>);
        }

    } // namespace

    ComputePass::ComputePass(std::string name, ComputeDevice device, ComputeStream stream) noexcept
        : m_name(std::move(name)),
          m_device(std::move(device)),
          m_stream(std::move(stream)) {}

    Status ComputePass::dispatch() noexcept {
        ZoneScoped();
        ZoneName(m_name.c_str(), m_name.size());
        if (!m_device || !m_stream) {
            return FailedPreconditionError(StrCat("ComputePass [", m_name, "] device or stream is not initialized."));
        }
        ComputePassContext passCtx(m_device, m_stream, nullptr);
        return execute(passCtx);
    }

    Status ComputePass::dispatch(FCT::ComputeExecutionContext& context) {
        ZoneScoped();
        ZoneName(m_name.c_str(), m_name.size());
        if (!m_device || !m_stream) {
            return FailedPreconditionError(StrCat("ComputePass [", m_name, "] device or stream is not initialized."));
        }
        ComputePassContext passCtx(m_device, m_stream, &context);
        return execute(passCtx);
    }

} // namespace FCT::Compute
