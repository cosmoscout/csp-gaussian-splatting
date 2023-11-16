////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_GAUSSIAN_SPLATTING_LOGGER_HPP
#define CSP_GAUSSIAN_SPLATTING_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::gaussiansplatting {

/// This creates the default singleton logger for "csp-gaussiansplatting" when called for the first
/// time and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::gaussiansplatting

#endif // CSP_GAUSSIAN_SPLATTING_LOGGER_HPP
