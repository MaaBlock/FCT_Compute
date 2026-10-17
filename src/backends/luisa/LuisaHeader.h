#ifndef FCT_COMPUTE_BACKENDS_LUISA_HEADER_H
#define FCT_COMPUTE_BACKENDS_LUISA_HEADER_H

#include "../../core/ComputeTypeTraits.h"

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef OVERFLOW
#undef OVERFLOW
#endif
#ifdef small
#undef small
#endif

#include <luisa/luisa-compute.h>

LUISA_STRUCT(FCT::Vec2, x, y) {
    [[nodiscard]] auto to_vec() const noexcept {
        return luisa::compute::make_float2(this->x, this->y);
    }
    void from_vec(luisa::compute::Expr<luisa::float2> v) noexcept {
        this->x = v.x;
        this->y = v.y;
    }
};

LUISA_STRUCT(FCT::Compute::PackedFloat3, x, y, z) {
    [[nodiscard]] auto to_vec() const noexcept {
        return luisa::compute::make_float3(this->x, this->y, this->z);
    }
    void from_vec(luisa::compute::Expr<luisa::float3> v) noexcept {
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
    }
};

LUISA_STRUCT(FCT::Vec4, x, y, z, w) {
    [[nodiscard]] auto to_vec() const noexcept {
        return luisa::compute::make_float4(this->x, this->y, this->z, this->w);
    }
    void from_vec(luisa::compute::Expr<luisa::float4> v) noexcept {
        this->x = v.x;
        this->y = v.y;
        this->z = v.z;
        this->w = v.w;
    }
};

#endif // FCT_COMPUTE_BACKENDS_LUISA_HEADER_H
