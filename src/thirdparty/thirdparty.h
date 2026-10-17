#ifndef FCT_COMPUTE_THIRDPARTY_H
#define FCT_COMPUTE_THIRDPARTY_H

#include <FCT/Bases.h>
#include <FCT/Context/Vec.h>
#include <FCT/Context/Matrix.h>
#include <FCT/Context/Format.h>
#include <FCT/Context/Image.h>
#include <FCT/Context/ComputePass.h>

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

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace FCT::Compute {

    using FCT::Status;
    using FCT::StatusOr;
    using FCT::OkStatus;
    using FCT::InvalidArgumentError;
    using FCT::FailedPreconditionError;
    using FCT::NotFoundError;
    using FCT::AlreadyExistsError;
    using FCT::InternalError;
    using FCT::UnavailableError;
    using FCT::UnimplementedError;
    using FCT::OutOfRangeError;
    using FCT::UnknownError;
    using FCT::StrCat;
    using FCT::ToStatus;

#ifndef CHECK_STATUS
#define CHECK_STATUS(expr) FCT_CHECK_STATUS(expr)
#endif

} // namespace FCT::Compute

#endif // FCT_COMPUTE_THIRDPARTY_H
