// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once
#ifdef USE_DISTRIBUTED

#include <exceptions.h>
#include <multidevice/multidevice.h>
#include <multidevice/communication.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace nvfuser {
/*
   This file implements the class Communicator which sets up the inter-process
   Backend. This class contains inter-process information, such as the rank, the
   world size, as well as the Process Group that can be called to perform
   inter-process communications.

   Each process is associated with a unique deviceId and device. The actual MPI
   rank remains private to the class and should not be used by the user. The
   communicator class holds privately the mappings ranks <-> device IDs <->
   device.

*/

using RankType = DeviceIdxType;

// Supported backends.
enum class CommunicatorBackendType { nccl, ucc, gloo, none };

constexpr CommunicatorBackendType comm_backend_default = CommunicatorBackendType::nccl;
constexpr int comm_server_local_rank_default = 0;
constexpr int comm_master_port_default =
    c10d::TCPStoreOptions::kDefaultPort; // 29500

class CommunicatorBackend {
  public:
    CommunicatorBackend(CommunicatorBackendType backend_type) : backend_type_(backend_type) {}
    // creates the world's backend
    void createWorld(int size, c10::intrusive_ptr<c10d::TCPStore> store, 
      DeviceIdxType deviceId);

    // retrieve/create backend for a team
    c10::intrusive_ptr<c10d::Backend> getBackendForTeam(const Team& team, 
      const c10::intrusive_ptr<c10d::TCPStore> store,
      DeviceIdxType deviceId);

    // return world's backend
    c10::intrusive_ptr<c10d::Backend> world() {
      return world_;
    }

  protected:
    CommunicatorBackendType backend_type_;
    // stores the world's backend
    c10::intrusive_ptr<c10d::Backend> world_;
    // cache for the created backends. The keys are strings generated from Teams
    std::unordered_map<std::string, c10::intrusive_ptr<c10d::Backend>> backends_;
};

class Communicator {
 public:
  Communicator(
      CommunicatorBackendType backend = comm_backend_default,
      RankType server_local_rank = comm_server_local_rank_default,
      int master_port = comm_master_port_default);
  Communicator(const Communicator&) = delete;
  Communicator& operator=(const Communicator&) = delete;

  // returns if distributed config is available
  auto is_available() const {
    return is_available_;
  }

  // returns the number of processes in the communicator
  auto size() const {
    return size_;
  }

  // returns the local number of processes in the communicator (within the node)
  auto local_size() const {
    return local_size_;
  }

  // adds another backend type to the communicator
  void addBackend(CommunicatorBackendType backend);

  // Triggers the execution of a communication. This is a non-blocking call.
  // The communication can be posted multiple times
  c10::intrusive_ptr<c10d::Work> post(Allgather& allgather, 
    CommunicatorBackendType backend_type = CommunicatorBackendType::none);
  c10::intrusive_ptr<c10d::Work> post(Broadcast& broadcast,
    CommunicatorBackendType backend_type = CommunicatorBackendType::none);
  c10::intrusive_ptr<c10d::Work> post(Gather& gather,
    CommunicatorBackendType backend_type = CommunicatorBackendType::none);
  c10::intrusive_ptr<c10d::Work> post(Scatter& scatter,
    CommunicatorBackendType backend_type = CommunicatorBackendType::none);
  c10::intrusive_ptr<c10d::Work> post(SendRecv& sendrecv,
    CommunicatorBackendType backend_type = CommunicatorBackendType::none);

  // performs a send/receive p2p data transfer
  c10::intrusive_ptr<c10d::Work> sendRecv(
      DeviceIdxType receiver,
      DeviceIdxType sender,
      std::vector<at::Tensor>& tensor,
      CommunicatorBackendType backend_type,
      int tag = 0);

  // performs a blocking barrier in the communicator
  void barrier(CommunicatorBackendType backend_type=CommunicatorBackendType::none);

  // returns the device associated with the current process
  auto device() const {
    return at::Device("cuda:" + std::to_string(local_rank_));
  }

  // returns the device Id associated with the current process
  DeviceIdxType deviceId() const {
    return rankToDiD(rank_);
  }

 private:
  // returns the rank corresponding to a device index
  RankType dIdToRank(DeviceIdxType d_id) const {
    return static_cast<RankType>(d_id);
  }

  // returns the device index corresponding to a rank
  DeviceIdxType rankToDiD(RankType rank) const {
    return static_cast<DeviceIdxType>(rank);
  }

  // returns the backend associated with a team and backend type
  c10::intrusive_ptr<c10d::Backend> getBackendForTeam(const Team& team, CommunicatorBackendType backend_type);

  bool is_available_;
  CommunicatorBackendType default_type_;
  RankType rank_;
  int64_t size_;
  RankType local_rank_;
  int64_t local_size_;
  std::string master_addr_;
  int master_port_;
  // stores the world's store used for the backend init
  c10::intrusive_ptr<c10d::TCPStore> store_;
  // // stores the world's backend
  // c10::intrusive_ptr<c10d::Backend> world_;
  // // cache for the created backends. The keys are strings generated from Teams
  // std::unordered_map<std::string, c10::intrusive_ptr<c10d::Backend>> backends_;

  std::map<CommunicatorBackendType, CommunicatorBackend> cbackends_;
};

} // namespace nvfuser

#endif
