// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#pragma once

#include <exceptions.h>
#include <ATen/ATen.h>
#include <c10/core/Device.h>

namespace nvfuser {
using DeviceIdxType = int64_t;
using DimensionType = int;
using DeviceType = c10::Device;
using Team = std::vector<DeviceIdxType>;

inline void assertBufferCount(
    const std::vector<at::Tensor>& bufs,
    size_t count) {
  NVF_ERROR(
      bufs.size() == count,
      "there must be ",
      count,
      " buffer(s), but ",
      bufs.size(),
      " were given");
}

inline void assertBuffersHaveSameSize(
    const std::vector<at::Tensor>& bufs1,
    const std::vector<at::Tensor>& bufs2) {
  if (bufs1.empty() && bufs2.empty()) {
    return;
  }
  auto sizes = (bufs1.empty() ? bufs2 : bufs1).at(0).sizes();
  for (auto& bufs : {bufs1, bufs2}) {
    for (auto& buf : bufs) {
      NVF_ERROR(buf.sizes() == sizes, "all buffers must have the same size");
    }
  }
}

} // namespace nvfuser
