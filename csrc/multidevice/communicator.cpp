// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <netdb.h>

#include <multidevice/communicator.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#endif
#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif
#if defined(USE_C10D_UCC) && defined(NVFUSER_BUILD_WITH_UCC)
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#endif

namespace nvfuser {
namespace{
inline void post_common(Communication& self, Communicator& comm) {
  NVF_ERROR(
      std::find(
          self.params().team.begin(),
          self.params().team.end(),
          comm.deviceId()) != self.params().team.end(),
      "current device index ",
      comm.deviceId(),
      " must be present in the communication's team");
}

inline void doLocalCopy(const at::Tensor& dst, const at::Tensor& src) {
  dst.copy_(src, /* non-blocking */ true);
}
} // namespace

// Parse the environment to retrieve MPI rank, world size, local rank,
// local world size, and also master address and master port.
// Returns true if the distributed configuration is valid, false otherwise
bool parseEnv(
    RankType& rank,
    int64_t& size,
    RankType& local_rank,
    int64_t& local_size,
    std::string& master_addr,
    int& master_port) {
  char* env = nullptr;

  // retrieves the rank of the current process
  env = std::getenv("OMPI_COMM_WORLD_RANK");
  if (!env) {
    env = std::getenv("WORLD_RANK");
    if (!env) {
      return false;
    }
  }
  rank = std::atoi(env);

  // retrieves the size of the communicator
  env = std::getenv("OMPI_COMM_WORLD_SIZE");
  if (!env) {
    env = std::getenv("WORLD_SIZE");
    if (!env) {
      return false;
    }
  }
  size = std::atoi(env);

  // retrieves the size of the communicator
  env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env) {
    env = std::getenv("WORLD_LOCAL_RANK");
    if (!env) {
      return false;
    }
  }
  local_rank = std::atoi(env);

  // retrieves the size of the communicator
  env = std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (!env) {
    env = std::getenv("WORLD_LOCAL_SIZE");
    if (!env) {
      return false;
    }
  }
  local_size = std::atoi(env);

  // retrieves master address
  env = std::getenv("MASTER_ADDR");
  if (env) {
    // replace the potential aliased hostname by the "official" name
    master_addr = gethostbyname(env)->h_name;
  } else if (local_size == size) {
    master_addr = "localhost";
  } else {
    TORCH_WARN(
        "the environment variable MASTER_ADDR "
        "must be specified in multi-node environment");
    return false;
  }

  // retrieves master port
  env = std::getenv("MASTER_PORT");
  if (env) {
    master_port = std::atoi(env);
  } else {
    TORCH_WARN(
        "the environment variable MASTER_PORT "
        "has not been specified. Set to default");
  }

  return true;
}

inline std::string getTeamKey(const Team& team, CommunicatorBackendType backend_type) {
  std::string backend_str = (backend_type == CommunicatorBackendType::ucc) ? "ucc"  : "nccl";
  return std::accumulate(
      std::begin(team),
      std::end(team),
      std::string{backend_str},
      [](const std::string& a, const RankType& b) {
        return a.empty() ? std::to_string(b) : a + ',' + std::to_string(b);
      });
}

// creates and return a process group backend
c10::intrusive_ptr<c10d::Backend> createBackend(
    CommunicatorBackendType backend,
    ::c10::intrusive_ptr<c10d::Store> store,
    RankType rank,
    uint64_t size) {
#ifdef USE_C10D_NCCL
  if (backend == CommunicatorBackendType::nccl) {
    auto pg_opts = c10::make_intrusive<::c10d::ProcessGroupNCCL::Options>();
    return c10::make_intrusive<::c10d::ProcessGroupNCCL>(
        store, rank, size, pg_opts);
  }
#endif

#ifdef USE_C10D_GLOO
  if (backend == CommunicatorBackendType::gloo) {
    auto pg_opts = c10d::ProcessGroupGloo::Options::create();
    return c10::make_intrusive<::c10d::ProcessGroupGloo>(
        store, rank, size, pg_opts);
  }
#endif

#if defined(USE_C10D_UCC) && defined(NVFUSER_BUILD_WITH_UCC)
  if (backend == CommunicatorBackendType::ucc) {
	  std::cout << "Creating UCC backend " << rank << " " << size << std::endl;
    return c10::make_intrusive<::c10d::ProcessGroupUCC>(store, rank, size);
  }
#endif
  NVF_CHECK(false, "no distributed backend available");
}

Communicator::Communicator(
    CommunicatorBackendType backend,
    RankType server_local_rank,
    int master_port)
    : is_available_(false),
      default_type_(backend),
      rank_(0),
      size_(0),
      local_rank_(0),
      local_size_(0),
      master_port_(master_port) {
  // retrieves rank and communicator size
  is_available_ = parseEnv(
      rank_, size_, local_rank_, local_size_, master_addr_, master_port_);

  if (!is_available_) {
    return;
  }

  c10d::TCPStoreOptions store_opts;
  {
    char hostname[HOST_NAME_MAX]; // NOLINT (modernize-avoid-c-arrays)
    NVF_ERROR(
        gethostname(hostname, HOST_NAME_MAX) == 0,
        "error when retrieving hostname");
    // we define the server as the process at the master host with local rank 0
    store_opts.isServer = (master_addr_ == "localhost" ||
                           master_addr_ == gethostbyname(hostname)->h_name) &&
        local_rank_ == server_local_rank;
  }
  store_opts.port = master_port_ ? master_port_ : comm_master_port_default;
  store_ = c10::make_intrusive<c10d::TCPStore>(master_addr_, store_opts);

  addBackend(backend);
}

void Communicator::addBackend(CommunicatorBackendType backend_type) {
  // auto backend = CommunicatorBackend(backend_type);
  // backend.createWorld(size_, store_, deviceId());
  // backends_.emplace(backend_type, backend);
  std::vector<RankType> all_ranks(size_);
  std::iota(all_ranks.begin(), all_ranks.end(), 0);
  // creates world.
  getBackendForTeam(all_ranks, backend_type);
  std::cout << "Created a world backend" << std::endl;
}

c10::intrusive_ptr<c10d::Backend> Communicator::getBackendForTeam(
    const Team& team, CommunicatorBackendType backend_type) {
  if (backend_type == CommunicatorBackendType::none)
	  backend_type = default_type_;
  std::string team_key = getTeamKey(team, backend_type);
  std::cout << "Get team for " << team_key << std::endl;

  // check if backend associated with the team is present in the cache
  if (backends_.find(team_key) == backends_.end()) { // create the backend and cache it
    std::cout << "Create team for " << team_key << std::endl;
    // check that the caller's rank belongs to the requested team
    auto rank_it = std::find(team.begin(), team.end(), deviceId());
    NVF_ERROR(
        rank_it != team.end(),
        "only devices in the team should participate to its initialization");
    // retrieve the caller's rank index/position in the team
    RankType team_rank = std::distance(team.begin(), rank_it);
    // generate a string key which is unique to the team
    // create the team and cache it
    backends_[team_key] = createBackend(
        backend_type,
        c10::make_intrusive<c10d::PrefixStore>(team_key, store_),
        team_rank,
        team.size());
  }
  return backends_.at(team_key);
}

c10::intrusive_ptr<c10d::Work> Communicator::post(Allgather& ag, CommunicatorBackendType backend_type) {
  post_common(ag, *this);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list;
  buf_list = {std::move(ag.params().dst_bufs)};
  auto backend = getBackendForTeam(ag.params().team, backend_type);
  auto work = backend->allgather(buf_list, ag.params().src_bufs, {});
  ag.params().dst_bufs = std::move(buf_list.back());
  return work;
}

c10::intrusive_ptr<c10d::Work> Communicator::post(Gather& gather, CommunicatorBackendType backend_type) {
  post_common(gather, *this);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::cout << "Gather!" << deviceId() << std::endl;
  std::vector<std::vector<at::Tensor>> buf_list;
  if (deviceId() == gather.params().root) {
    assertBufferCount(gather.params().dst_bufs, gather.params().team.size());
    buf_list = {std::move(gather.params().dst_bufs)};
  } else {
    assertBufferCount(gather.params().dst_bufs, 0);
  }
  auto backend = getBackendForTeam(gather.params().team, backend_type);
  std::cout << "Post gather" << deviceId() << std::endl;
  auto work = backend->gather(
              buf_list, gather.params().src_bufs, {.rootRank = gather.root_relative_index()});
  if (deviceId() == gather.params().root) {
    gather.params().dst_bufs = std::move(buf_list.back());
  }
  return work;
}

c10::intrusive_ptr<c10d::Work> Communicator::post(Scatter& scatter, CommunicatorBackendType backend_type) {
  post_common(scatter, *this);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list;
  if (deviceId() == scatter.params().root) {
    assertBufferCount(scatter.params().src_bufs, scatter.params().team.size());
    buf_list = {std::move(scatter.params().src_bufs)};
  } else {
    assertBufferCount(scatter.params().src_bufs, 0);
  }
  auto backend = getBackendForTeam(scatter.params().team, backend_type);
  auto work = backend->scatter(
              scatter.params().dst_bufs, buf_list, {.rootRank = scatter.root_relative_index()});
  if (deviceId() == scatter.params().root) {
    scatter.params().src_bufs = std::move(buf_list.back());
  }
  return work;
}

c10::intrusive_ptr<c10d::Work> Communicator::post(Broadcast& broadcast, CommunicatorBackendType backend_type) {
  post_common(broadcast, *this);
  std::cout << "Broadcast! root " << broadcast.params().root << std::endl;
  if (deviceId() == broadcast.params().root) {
    assertBufferCount(broadcast.params().src_bufs, 1);
    if (broadcast.params().dst_bufs.size() == 1) {
      doLocalCopy(broadcast.params().dst_bufs.at(0), broadcast.params().src_bufs.at(0));
    } else {
      assertBufferCount(broadcast.params().dst_bufs, 0);
    }
  } else {
    assertBufferCount(broadcast.params().src_bufs, 0);
    assertBufferCount(broadcast.params().dst_bufs, 1);
  }

  if (broadcast.params().team.size() == 1) {
    return nullptr;
  }

  std::cout << "Get Broadcast backend " << deviceId() << std::endl;
  auto backend = getBackendForTeam(broadcast.params().team, backend_type);
  std::cout << "Got broadcast backend" << std::endl;
  return backend->broadcast(
          deviceId() == broadcast.params().root ? broadcast.params().src_bufs : broadcast.params().dst_bufs,
          {.rootRank = broadcast.root_relative_index()});
}

c10::intrusive_ptr<c10d::Work> Communicator::post(SendRecv& sr, CommunicatorBackendType backend_type) {
  post_common(sr, *this);

  if (deviceId() == sr.params().root) {
    assertBufferCount(sr.params().src_bufs, 1);
    if (sr.params().team.size() == 1) {
      assertBufferCount(sr.params().dst_bufs, 1);
      doLocalCopy(sr.params().dst_bufs.at(0), sr.params().src_bufs.at(0));
      return nullptr;
    } else {
      assertBufferCount(sr.params().dst_bufs, 0);
    }
  } else {
    assertBufferCount(sr.params().src_bufs, 0);
    assertBufferCount(sr.params().dst_bufs, 1);
  }

  return sendRecv(
      (sr.params().team.at(0) == sr.params().root) ? sr.params().team.at(1)
                                           : sr.params().team.at(0),
      sr.params().root,
      sr.params().dst_bufs.empty() ? sr.params().src_bufs : sr.params().dst_bufs,
      backend_type);
}

c10::intrusive_ptr<c10d::Backend> Communicator::world(CommunicatorBackendType backend_type) {
  if (backend_type == CommunicatorBackendType::none)
	  backend_type = default_type_;
  std::vector<RankType> all_ranks(size_);
  std::iota(all_ranks.begin(), all_ranks.end(), 0);
  std::string key = getTeamKey(all_ranks, backend_type);
  std::cout << "World rank:" << deviceId() << " key: " << key << std::endl;
  return backends_.at(key); 
}

c10::intrusive_ptr<c10d::Work> Communicator::sendRecv(
    DeviceIdxType receiver,
    DeviceIdxType sender,
    std::vector<at::Tensor>& tensors,
    CommunicatorBackendType backend_type, 
    int tag) {
  NVF_ERROR(
      deviceId() == sender || deviceId() == receiver,
      "only sender or receiver should post the sendRecv");
  NVF_ERROR(sender != receiver, "cannot send to self");
  if (deviceId() == sender) {
    return world(backend_type)->send(tensors, static_cast<int>(dIdToRank(receiver)), tag);
  }
  return world(backend_type)->recv(tensors, static_cast<int>(dIdToRank(sender)), tag);
}

void Communicator::barrier(CommunicatorBackendType backend_type) {
  std::cout << "Barrier called" << std::endl;
  world(backend_type)->barrier()->wait();
  std::cout << "Barrier returns" << std::endl;
}

} // namespace nvfuser

#endif
