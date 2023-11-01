// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <exceptions.h>
#include <multidevice/communication.h>

namespace nvfuser {

Communication::Communication(CommParams params, std::string name, bool has_root)
    : params_(std::move(params)),
      collective_type_(std::move(name)),
      has_root_(has_root) {
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  NVF_ERROR(
      std::unique(params_.team.begin(), params_.team.end()) ==
          params_.team.end(),
      "the communication must not involve the same device more than once");
  NVF_ERROR(!params_.team.empty(), "the team size must be greater than 0");
  if (has_root_) {
    auto it = std::find(params_.team.begin(), params_.team.end(), params_.root);
    NVF_ERROR(
        it != params_.team.end(),
        "root (device ",
        params_.root,
        ") must be present in the communication's team");
    // pytorch's process group expects the root to be specified
    // as an integer between 0 and world_size-1. We choose it to be
    // the device's relative index within the team
    root_relative_index_ = std::distance(params_.team.begin(), it);
  }
}

std::string Communication::toString(int indent) const {
  std::stringstream ss;
  std::string ext_indent(" ", indent);
  std::string indent1 = ext_indent + "  ";
  std::string indent2 = ext_indent + "    ";

  ss << ext_indent << "Communication " << collective_type_ << ": {\n";

  if (has_root_) {
    ss << indent1 << "root: " << params_.root << ",\n";
  }
  ss << indent1 << "team: {";
  for (auto r : params_.team) {
    ss << r << ", ";
  }
  ss << indent1 << "}\n";
  ss << indent1 << "src_bufs: {";
  for (auto& t : params_.src_bufs) {
    ss << "\n" << t;
  }
  ss << "\n" << indent1 << "}\n";
  ss << ext_indent << "}";

  return ss.str();
}

Broadcast::Broadcast(CommParams params) : Communication(params, "broadcast") {}

Gather::Gather(CommParams params) : Communication(params, "gather") {
  assertBufferCount(params_.src_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

Allgather::Allgather(CommParams params)
    : Communication(params, "allgather", false) {
  assertBufferCount(params_.src_bufs, 1);
  assertBufferCount(params_.dst_bufs, params_.team.size());
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

Scatter::Scatter(CommParams params) : Communication(params, "scatter") {
  assertBufferCount(params_.dst_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

SendRecv::SendRecv(CommParams params) : Communication(params, "send/recv") {
  NVF_ERROR(
      params_.team.size() == 1 || params_.team.size() == 2,
      "the team size should be 1 or 2");
}

} // namespace nvfuser
