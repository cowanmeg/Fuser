// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <fstream>
#include <functional>
#include <iostream>

#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <mma_type.h>
#include <multidevice/communicator.h>
#include <multidevice/executor.h>
#include <multidevice/multidevice.h>
#include <multidevice/device_mesh.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

struct stats {
  std::string description;
  std::string collective;
  CommunicatorBackend backend;
  int M;
  int N;
  int K;
  int M_local;
  int N_local;
  int K_local;
  double average_time;
};

class DistributedMatmulTest {
 public:
  DistributedMatmulTest() {
    communicator_ = new Communicator();
    num_devices_ = communicator_->size();
    mesh_ = DeviceMesh::createForNumDevices(num_devices_);
    repeats_ = 5;
  }

  MultiDeviceExecutorParams executor_params_{
      .use_fusion_executor_cache = true,
      .skip_auto_scheduling = false,
      .cache_fusion_executor = false};
  int num_devices_;
  Communicator* communicator_;
  DeviceMesh mesh_;
  int repeats_;

  std::tuple<at::Tensor, at::Tensor, at::Tensor> getInputsAndReferenceOutputs(
      MmaLayout layout,
      int M,
      int N,
      int K,
      int B=0) {
    
    c10::ScalarType type = c10::ScalarType::Half;
    auto a = matmulAtInput2D(
        layout, TensorMatmulPos::A, type, M, N, K, B, communicator_->local_rank());
    auto b = matmulAtInput2D(
        layout, TensorMatmulPos::B, type, M, N, K, B, communicator_->local_rank());
    auto c =
        atMatmul(a.to(at::kDouble), b.to(at::kDouble), layout).to(at::kFloat);
    return std::make_tuple(a, b, c);
  }

  at::Tensor shardTensor(
      at::Tensor tensor,
      TensorView* tv) {
    if (!isSharded(tv)) {
      return tensor;
    }
    auto sharded_dim = getShardedAxis(tv);
    int i = 0;
    const auto& devices = tv->getDeviceMesh().vector();
    auto it = std::find(devices.begin(), devices.end(), communicator_->deviceId());
    if (it != devices.end()) {
      i = std::distance(devices.begin(), it);
    }
    return tensor.slice(sharded_dim, i, i + 1).contiguous();
  }

  void setDeviceMesh(std::vector<TensorView*>& all_tvs) {
    for (auto tv : all_tvs) {
      tv->setDeviceMesh(mesh_);
    }
  }

  int64_t deviceId() {
    return communicator_->deviceId();
  }

  stats nvFuser_NoComms(int M, int N, int K) {
    // MmaLayout::TN A(T), B(N), C(T)
    // A and C are sharded on dimension M
    // Tests local matmul with no communication
    std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    int Mo = num_devices_;
    int Mi = M / Mo;
    std::vector<int> a_shape = {Mo, Mi, K};
    std::vector<int> b_shape = {N, K};

    TensorView* a = makeContigTensor(3, DataType::Half); // (Mo,Mi,K)
    TensorView* b = makeContigTensor(2, DataType::Half); // (N,K)
    TensorView* a_b = broadcast(a, {false, false, true, false}); // (Mo,Mi,b,K)
    TensorView* b_b = broadcast(b, {true, true, false, false}); // (b,b,N,K)
    TensorView* ab = mul(a_b, b_b); // (Mo,Mi,N,K)
    TensorView* c = sum(ab, {-1}); // (Mo,Mi,N,r)

    fusion->addInput(a);
    fusion->addInput(b);
    fusion->addOutput(c);

    // Sharding M dimension
    std::vector<TensorView*> all_tvs = {a, b, a_b, b_b, ab, c};
    setDeviceMesh(all_tvs);

    auto all_sharded_tvs = {a, a_b, b_b, ab, c};
    for (auto tv : all_sharded_tvs) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }

    // TODO: If c's allocation domain isn't set, it will fail validation at
    // csrc/device_lower/validation.cpp:419, Vectorized dim for consumer has to be
    // from a contiguous inner most position.
    c->setAllocationDomain(c->getLoopDomain(), true);

    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TN, M, N, K);
    in0 = in0.view({Mo, Mi, K});
    out = out.view({Mo, Mi, N});
    std::vector<c10::IValue> inputs = {shardTensor(in0, a), in1};
    auto expected_output = shardTensor(out, c);
    MultiDeviceExecutor runtime(std::move(fusion), *communicator_, executor_params_);
    // TODO validate the first run
    auto outputs = runtime.runWithInput(inputs);
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      runtime.runWithInput(inputs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "nvfuser generated matmul";
    entry.collective = "None";
    entry.average_time = average_time;
    entry.M_local = Mi;
    entry.N_local = N;
    entry.K_local = K;
    return entry;
  }

  stats nvFuserATen_NoComms(int M, int N, int K) {
    // MmaLayout::TN A(T), B(N), C(T)
    // A and C are sharded on dimension M
    // Tests local matmul with no communication
    std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    int Mo = num_devices_;
    int Mi = M / Mo;
    std::vector<int> a_shape = {Mo, Mi, K};
    std::vector<int> b_shape = {K, N};

    TensorView* a = makeContigTensor(3, DataType::Half); // (Mo,Mi,K)
    TensorView* b = makeContigTensor(2, DataType::Half); // (N,K)
    TensorView* c = matmul(a, b);

    fusion->addInput(a);
    fusion->addInput(b);
    fusion->addOutput(c);

    // Sharding M dimension
    std::vector<TensorView*> all_tvs = {a, b, c};
    setDeviceMesh(all_tvs);

    auto all_sharded_tvs = {a,c};
    for (auto tv : all_sharded_tvs) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }

    // TODO: If c's allocation domain isn't set, it will fail validation at
    // csrc/device_lower/validation.cpp:419, Vectorized dim for consumer has to be
    // from a contiguous inner most position.
    c->setAllocationDomain(c->getLoopDomain(), true);

    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TT, M, N, K);
    in0 = in0.view({Mo, Mi, K});
    out = out.view({Mo, Mi, N});
    std::vector<c10::IValue> inputs = {shardTensor(in0, a), in1};
    auto expected_output = shardTensor(out, c);
    MultiDeviceExecutor runtime(std::move(fusion), *communicator_, executor_params_);
    // TODO validate the first run
    auto outputs = runtime.runWithInput(inputs);
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      runtime.runWithInput(inputs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "nvfuser ATen matmul";
    entry.collective = "None";
    entry.average_time = average_time;
    entry.M_local = Mi;
    entry.N_local = N;
    entry.K_local = K;
    return entry;
  }

  stats nvFuser_AllGather(int M, int N, int K) {
    // MmaLayout::TN matmul A(T), B(N), C(T)
    // A is sharded on dimension M
    // Tests local matmul + AllGather
    std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    
    int Mo = num_devices_;
    int Mi = M / Mo;
    std::vector<int> a_shape = {Mo, Mi, K};
    std::vector<int> b_shape = {N, K};

    TensorView* a = makeContigTensor(3, DataType::Half); // (Mo,Mi,K)
    TensorView* b = makeContigTensor(2, DataType::Half); // (N,K)
    TensorView* a_b = broadcast(a, {false, false, true, false}); // (Mo,Mi,b,K)
    TensorView* b_b = broadcast(b, {true, true, false, false}); // (b,b,N,K)
    TensorView* ab = mul(a_b, b_b); // (Mo,Mi,N,K)
    TensorView* c0 = sum(ab, {-1}); // (Mo,Mi,N,r)
    TensorView* c = set(c0);

    fusion->addInput(a);
    fusion->addInput(b);
    fusion->addOutput(c);

    std::vector<TensorView*> all_tvs = {a, b, a_b, b_b, ab, c0, c};
    setDeviceMesh(all_tvs);

    // Sharding M dimension
    auto all_sharded_tvs = {a, a_b, b_b, ab, c0};
    for (auto tv : all_sharded_tvs) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }

    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TN, M, N, K);
    in0 = in0.view({Mo, Mi, K});
    out = out.view({Mo, Mi, N});

    std::vector<c10::IValue> inputs = {
        shardTensor(in0, a), in1};
    MultiDeviceExecutor runtime(std::move(fusion), *communicator_, executor_params_);
    auto outputs = runtime.runWithInput(inputs);
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      runtime.runWithInput(inputs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "nvfuser generated matmul";
    entry.collective = "AllGather";
    entry.average_time = average_time;
    entry.M_local = Mi;
    entry.N_local = N;
    entry.K_local = K;
    return entry;
  }

  stats nvFuserATen_AllGather(int M, int N, int K) {
    // A is sharded on dimension M
    // Tests local matmul + AllGather
    std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    
    int Mo = num_devices_;
    int Mi = M / Mo;
    std::vector<int> a_shape = {Mo, Mi, K};
    std::vector<int> b_shape = {K, N};

    TensorView* a = makeContigTensor(3, DataType::Half); // (Mo,Mi,K)
    TensorView* b = makeContigTensor(2, DataType::Half); // (N,K)
    TensorView* c0 = matmul(a, b);
    TensorView* c = set(c0);

    fusion->addInput(a);
    fusion->addInput(b);
    fusion->addOutput(c);

    std::vector<TensorView*> all_tvs = {a, b, c0, c};
    setDeviceMesh(all_tvs);

    // Sharding M dimension
    auto all_sharded_tvs = {a, c0};
    for (auto tv : all_sharded_tvs) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }

    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TN, M, N, K);
    in0 = in0.view({Mo, Mi, K});
    out = out.view({Mo, Mi, N});

    std::vector<c10::IValue> inputs = {
        shardTensor(in0, a), in1};
    MultiDeviceExecutor runtime(std::move(fusion), *communicator_, executor_params_);
    auto outputs = runtime.runWithInput(inputs);
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      runtime.runWithInput(inputs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "nvfuser ATen matmul";
    entry.collective = "AllGather";
    entry.average_time = average_time;
    entry.M_local = Mi;
    entry.N_local = N;
    entry.K_local = K;
    return entry;
  }

  stats nvFuser_AllReduce(int M, int N, int K) {
    // MmaLayout::NT matmul A(N), B(T), C(T)
    // Sharding: A, B are sharded along K. C is replicated.
    // Tests local matmul + allreduce
    std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    
    int Ko = num_devices_, Ki = K / Ko;
    std::vector<int> a_shape = {Ko, Ki, M};
    std::vector<int> b_shape = {Ko, Ki, N};

    TensorView* a = makeContigTensor(3, DataType::Half); // (Ko,Ki,M)
    TensorView* b = makeContigTensor(3, DataType::Half); // (Ko,Ki,N)
    // Transpose into TN layout, keep Ko (device axis) as the outermost.
    TensorView* a_t = transpose(a, 1, 2); // (Ko,M,Ki)
    TensorView* b_t = transpose(b, 1, 2); // (Ko,N,Ki)
    TensorView* a_b = broadcast(a_t, {false, false, true, false}); // (Ko,M,b,Ki)
    TensorView* b_b = broadcast(b_t, {false, true, false, false}); // (Ko,b,N,Ki)
    TensorView* ab = mul(a_b, b_b); // (Ko,M,N,Ki)
    TensorView* c0 = sum(ab, {-1}); // (Ko,M,N,r)
    TensorView* c = sum(c0, {0}); // (r,M,N)

    fusion->addInput(a);
    fusion->addInput(b);
    fusion->addOutput(c);

    std::vector<TensorView*> all_tvs = {a, b, a_t, b_t, a_b, b_b, ab, c0, c};
    setDeviceMesh(all_tvs);
    // Parallelize K on all inputs and intermediates.
    auto all_sharded_tvs = {a, b, a_t, b_t, a_b, b_b, ab, c0};
    for (auto tv : all_sharded_tvs) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }

    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::NT, M, N, K);
    in0 = in0.view({Ko, Ki, M});
    in1 = in1.view({Ko, Ki, N});
    std::vector<c10::IValue> inputs = {
        shardTensor(in0, a),
        shardTensor(in1, b)};
    MultiDeviceExecutor runtime(std::move(fusion), *communicator_, executor_params_);
    auto outputs = runtime.runWithInput(inputs);
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      runtime.runWithInput(inputs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "nvfuser generated matmul";
    entry.collective = "AllReduce";
    entry.average_time = average_time;
    entry.M_local = M;
    entry.N_local = N;
    entry.K_local = Ki;
    return entry;
  }

  stats nvFuserATen_AllReduce(int M, int N, int K) {
    std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    
    int Ko = num_devices_, Ki = K / Ko;
    std::vector<int> a_shape = {Ko, M, Ki};
    std::vector<int> b_shape = {Ko, Ki, N};

    TensorView* a = makeContigTensor(3, DataType::Half); // (Ko,M,Ki)
    TensorView* b = makeContigTensor(3, DataType::Half); // (Ko,Ki,N)
    TensorView* c0 = matmul(a, b);
    TensorView* c = sum(c0, {0}); // (r,M,N)

    fusion->addInput(a);
    fusion->addInput(b);
    fusion->addOutput(c);

    std::vector<TensorView*> all_tvs = {a, b, c0, c};
    setDeviceMesh(all_tvs);
    // Parallelize K on all inputs and intermediates.
    auto all_sharded_tvs = {a, b, c0};
    for (auto tv : all_sharded_tvs) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }

    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TT, M, N, K);
    in0 = in0.view({M, Ko, Ki}).transpose(1, 0);
    in1 = in1.view({Ko, Ki, N});
    std::vector<c10::IValue> inputs = {
        shardTensor(in0, a),
        shardTensor(in1, b)};
    MultiDeviceExecutor runtime(std::move(fusion), *communicator_, executor_params_);
    auto outputs = runtime.runWithInput(inputs);
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      runtime.runWithInput(inputs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "nvfuser ATen matmul";
    entry.collective = "AllReduce";
    entry.average_time = average_time;
    entry.M_local = M;
    entry.N_local = N;
    entry.K_local = Ki;
    return entry;
  }


  stats nvFuser_ReduceScatter(int M, int N, int K) {
    // MmaLayout::NT matmul A(N), B(T), C(T)
    // A, B are sharded on K. C is sharded on M
    // Tests local matmul + reduce scatter
    std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    int Ko = num_devices_, Ki = K / Ko;
    int Mo = num_devices_, Mi = M / Mo;
    std::vector<int> a_shape = {Ko, Ki, M};
    std::vector<int> b_shape = {Ko, Ki, N};

    TensorView* a = makeContigTensor(3, DataType::Half); // (Ko,Ki,M)
    TensorView* b = makeContigTensor(3, DataType::Half); // (Ko,Ki,N)
    TensorView* a_t = transpose(a, 1, 2); // (Ko, M, Ki)
    TensorView* b_t = transpose(b, 1, 2); // (Ko, N, Ki)
    TensorView* a_b = broadcast(a_t, {false, false, true, false}); // (Ko,M,b,Ki)
    TensorView* b_b = broadcast(b_t, {false, true, false, false}); // (Ko,b,N,Ki)
    TensorView* ab = mul(a_b, b_b); // (Ko,M,N,Ki)
    TensorView* c0 = sum(ab, {-1}); // (Ko,M,N,r)
    c0 = segment_set(c0);
    std::vector<int64_t> orig_size = {Ko, M, N};
    std::vector<int64_t> new_size = {Ko, Mo, Mi, N};
    TensorView* c1 = reshape(c0, orig_size, new_size);
    TensorView* c = sum(c1, {0});

    fusion->addInput(a);
    fusion->addInput(b);
    fusion->addOutput(c);

    std::vector<TensorView*> all_tvs = {a, b, a_t, b_t, a_b, b_b, ab, c0, c1, c};
    setDeviceMesh(all_tvs);
    // Sharding K dimension of all inputs and intermediates.
    auto all_sharded_tvs = {a, b, a_t, b_t, a_b, b_b, ab, c0, c1};
    for (auto tv : all_sharded_tvs) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }
    // Sharding M on output
    c->axis(1)->parallelize(ParallelType::DIDx);

    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::NT, M, N, K);
    in0 = in0.view({Ko, Ki, M});
    in1 = in1.view({Ko, Ki, N});
    out = out.view({Mo, Mi, N});
    std::vector<c10::IValue> inputs = {
        shardTensor(in0, a),
        shardTensor(in1, b)};
    auto expected_output =
        shardTensor(out, c).view({1, Mi, N});

    MultiDeviceExecutor runtime(std::move(fusion), *communicator_, executor_params_);
    auto outputs = runtime.runWithInput(inputs);
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      runtime.runWithInput(inputs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "nvfuser generated matmul";
    entry.collective = "ReduceScatter";
    entry.average_time = average_time;
    entry.M_local = M;
    entry.N_local = N;
    entry.K_local = Ki;
    return entry;
  }

  stats nvFuserATen_ReduceScatter(int M, int N, int K) {
    // A, B are sharded on K. C is sharded on M
    // Tests local matmul + reduce scatter
    std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    int Ko = num_devices_, Ki = K / Ko;
    int Mo = num_devices_, Mi = M / Mo;
    std::vector<int> a_shape = {Ko, Ki, M};
    std::vector<int> b_shape = {Ko, Ki, N};

    TensorView* a = makeContigTensor(3, DataType::Half); // (Ko,M,Ki)
    TensorView* b = makeContigTensor(3, DataType::Half); // (Ko,Ki,N)
    TensorView* c0 = matmul(a, b); // (Ko,M,N)
    c0 = segment_set(c0);
    std::vector<int64_t> orig_size = {Ko, M, N};
    std::vector<int64_t> new_size = {Ko, Mo, Mi, N};
    TensorView* c1 = reshape(c0, orig_size, new_size);
    TensorView* c = sum(c1, {0});

    fusion->addInput(a);
    fusion->addInput(b);
    fusion->addOutput(c);

    std::vector<TensorView*> all_tvs = {a, b, c0, c1, c};
    setDeviceMesh(all_tvs);
    // Sharding K dimension of all inputs and intermediates.
    auto all_sharded_tvs = {a, b, c0, c1};
    for (auto tv : all_sharded_tvs) {
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }
    // Sharding M on output
    c->axis(1)->parallelize(ParallelType::DIDx);

    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TT, M, N, K);
    in0 = in0.view({M, Ko, Ki}).transpose(1, 0); // Ko M Ki
    in1 = in1.view({Ko, Ki, N});
    out = out.view({Mo, Mi, N});
    std::vector<c10::IValue> inputs = {
        shardTensor(in0, a),
        shardTensor(in1, b)};
    auto expected_output =
        shardTensor(out, c).view({1, Mi, N});

    MultiDeviceExecutor runtime(std::move(fusion), *communicator_, executor_params_);
    auto outputs = runtime.runWithInput(inputs);
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      runtime.runWithInput(inputs);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "nvfuser generated matmul";
    entry.collective = "ReduceScatter";
    entry.average_time = average_time;
    entry.M_local = M;
    entry.N_local = N;
    entry.K_local = Ki;
    return entry;
  }

  stats NoComms_Baseline(int M, int N, int K) {
    int Mo = num_devices_;
    int Mi = M / Mo;
    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TT, Mi, N, K);

    auto output = at::matmul(in0, in1);
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      at::matmul(in0, in1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "pytorch eager";
    entry.collective = "None";
    entry.average_time = average_time;
    entry.M_local = Mi;
    entry.N_local = N;
    entry.K_local = K;
    return entry;
  }

  stats AllGather_Baseline(int M, int N, int K) {
    int Mo = num_devices_;
    int Mi = M / Mo;
    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TT, Mi, N, K);
    
    auto mm = at::matmul(in0, in1);
    at::Tensor ag = at::empty({M, N}, mm.options());
    auto work = communicator_->getWorld()->_allgather_base(ag, mm, {});
    work->wait();

    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      mm = at::matmul(in0, in1);
      work = communicator_->getWorld()->_allgather_base(ag, mm, {});
      work->wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "pytorch eager";
    entry.collective = "AllGather";
    entry.average_time = average_time;
    entry.M_local = Mi;
    entry.N_local = N;
    entry.K_local = K;
    return entry;
  }

  stats AllReduce_Baseline(int M, int N, int K) {
    int Ko = num_devices_;
    int Ki = K / Ko;
    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TT, M, N, Ki);
    
    auto mm = at::matmul(in0, in1);
    at::Tensor ar = at::empty({M, N}, mm.options());
    std::vector<at::Tensor> inputs{mm};

    auto work = communicator_->getWorld()->allreduce(inputs);
    work->wait();

    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      mm = at::matmul(in0, in1);
      inputs[0] = mm;
      work = communicator_->getWorld()->allreduce(inputs);
      work->wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "pytorch eager";
    entry.collective = "AllReduce";
    entry.average_time = average_time;
    entry.M_local = M;
    entry.N_local = N;
    entry.K_local = Ki;
    return entry;
  }

  stats ReduceScatter_Baseline(int M, int N, int K) {
    int Ko = num_devices_;
    int Ki = K / Ko;
    int Mi = M / num_devices_;
    auto [in0, in1, out] = getInputsAndReferenceOutputs(MmaLayout::TT, M, N, Ki);
    
    auto mm = at::matmul(in0, in1);
    std::vector<std::vector<at::Tensor>> inputs = {mm.chunk(num_devices_)};
    std::vector<at::Tensor> outputs = {at::empty({Mi, N}, mm.options())};
    auto work = communicator_->getWorld()->reduce_scatter(outputs, inputs);
    work->wait();

    auto start = std::chrono::high_resolution_clock::now();
    for (auto i : c10::irange(repeats_)) {
      mm = at::matmul(in0, in1);
      inputs[0] = mm.chunk(num_devices_);
      work = communicator_->getWorld()->reduce_scatter(outputs, inputs);
      work->wait();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time =  std::chrono::duration_cast< std::chrono::microseconds>(end - start).count();
    double average_time = time / double(repeats_);
    stats entry;
    entry.description = "pytorch eager";
    entry.collective = "ReduceScatter";
    entry.average_time = average_time;
    entry.M_local = M;
    entry.N_local = N;
    entry.K_local = Ki;
    return entry;
  }
};

std::string toString(CommunicatorBackend b) {
  switch(b) {
    case CommunicatorBackend::nccl:
      return "NCCL";
    case CommunicatorBackend::ucc:
      return "UCC";
    default:
      return "Unknown";
  }
}

void writeCSV(std::vector<stats>& data, std::string fname, int64_t deviceId) {
  std::stringstream ss;
  ss << fname << "_" << deviceId << ".csv";
  std::ofstream file(ss.str());
  file << "Description, Collective, Backend, M, N, K, M_local, N_local, K_local, average time (us)\n";
  for (const auto& entry : data) {
    file << entry.description << ", " << entry.collective << ", " << toString(entry.backend) << ", ";
    file << entry.M << ", " << entry.N << ", " << entry.K << ", "; 
    file << entry.M_local << ", " << entry.N_local << ", " << entry.K_local << ", " << entry.average_time << "\n";
  }
  file.close();
}

typedef stats (DistributedMatmulTest::*DistributedMatmulFn) (int M, int N, int K);

int main(int argc, char* argv[]) {
  // Parsing command line options

  std::string versions(argv[1]);
  std::string fname(argv[2]);
  int M_start = std::stoi(argv[3]);
  int M_end = std::stoi(argv[4]);
  int N_start = std::stoi(argv[5]);
  int N_end = std::stoi(argv[6]);
  int K_start = std::stoi(argv[7]);
  int K_end = std::stoi(argv[8]);

  std::vector<DistributedMatmulFn> matmuls;
  if (versions == "all") {
    matmuls = {&DistributedMatmulTest::nvFuser_NoComms,
    &DistributedMatmulTest::nvFuser_AllGather,
    &DistributedMatmulTest::nvFuser_AllReduce,
    &DistributedMatmulTest::nvFuser_ReduceScatter,
    &DistributedMatmulTest::nvFuserATen_NoComms,
    &DistributedMatmulTest::nvFuserATen_AllGather,
    &DistributedMatmulTest::nvFuserATen_AllReduce,
    &DistributedMatmulTest::nvFuserATen_ReduceScatter,
    &DistributedMatmulTest::NoComms_Baseline,
    &DistributedMatmulTest::AllGather_Baseline,
    &DistributedMatmulTest::AllReduce_Baseline,
    &DistributedMatmulTest::ReduceScatter_Baseline};
  } else if (versions == "none") {
    matmuls = {&DistributedMatmulTest::nvFuser_NoComms,
    &DistributedMatmulTest::nvFuserATen_NoComms,
    &DistributedMatmulTest::NoComms_Baseline};
  } else if (versions == "AllGather") {
    matmuls = {&DistributedMatmulTest::nvFuser_AllGather,
    &DistributedMatmulTest::nvFuserATen_AllGather,
    &DistributedMatmulTest::AllGather_Baseline};
  } else if (versions == "AllReduce") {
    matmuls = {&DistributedMatmulTest::nvFuser_AllReduce,
    &DistributedMatmulTest::nvFuserATen_AllReduce,
    &DistributedMatmulTest::AllReduce_Baseline};
  } else if (versions == "ReduceScatter") {
    matmuls = {&DistributedMatmulTest::nvFuser_ReduceScatter,
    &DistributedMatmulTest::nvFuserATen_ReduceScatter,
    &DistributedMatmulTest::ReduceScatter_Baseline};
  }

  DistributedMatmulTest mm;
  std::vector<stats> data;
  auto backends = {CommunicatorBackend::nccl, CommunicatorBackend::ucc};
  for (auto backend : backends) {
    mm.communicator_->setDefaultBackend(backend);
    for (auto matmul : matmuls) {
      for (auto M = M_start; M < M_end; M *= 2) {
        for (auto N = N_start; N < N_end; N *= 2) {
          for (auto K = K_start; K < K_end; K *= 2) {
            stats entry = std::invoke(matmul, mm, M, N, K);
            entry.backend = backend;
            entry.M = M;
            entry.N = N;
            entry.K = K;
            data.push_back(entry);
          }
        }
      }
    }
  }
  writeCSV(data, fname, mm.deviceId());
}