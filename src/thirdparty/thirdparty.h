#ifndef FCT_COMPUTE_THIRDPARTY_H
#define FCT_COMPUTE_THIRDPARTY_H

#include <FCT/headers.h>
#include <FCT/Context/ComputePass.h>

#include <concepts>
#include <type_traits>
#include <tuple>
#include <utility>
#include <span>
#include <array>
#include <string>
#include <string_view>
#include <filesystem>
#include <functional>
#include <cstdint>
#include <cstddef>

namespace FCT::Compute {
    using absl::Status;
    using absl::StatusOr;
    using absl::OkStatus;
    using absl::InvalidArgumentError;
    using absl::InternalError;
    using absl::NotFoundError;
    using absl::FailedPreconditionError;
    using absl::UnimplementedError;
    using absl::StrCat;
} // namespace FCT::Compute

#ifndef CHECK_STATUS
#define CHECK_STATUS FCT_CHECK_STATUS
#endif

#endif // FCT_COMPUTE_THIRDPARTY_H
