#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/utils.h>
#include <test/validator.h>
namespace nvfuser {

using PersistentBufferTest = NVFuserTest;

TEST_F(PersistentBufferTest, FusionPersistentBufferCalculation1_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);

  auto tv1 = set(tv0);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = set(tv1);
  auto tv5 = add(tv3, tv4);
  fusion.addOutput(tv5);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  NVF_ERROR(buffers.size() == 1);
  NVF_ERROR(resolution.size() == 1 && resolution[0].size() == 1);
  NVF_ERROR(projectable.size() == 1);
  NVF_ERROR(projectable_inputs.size() == 1);

  NVF_ERROR(isTvWithinVec(buffers, tv1));
  NVF_ERROR(isTvWithinVec(projectable, tv1));
  NVF_ERROR(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  NVF_ERROR(tv1_resolution_it != resolution.end())

  NVF_ERROR(isTvWithinVec(*tv1_resolution_it, tv5));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0});
  auto persistent_buffer_size =
      persistentBufferSize(&fusion, runtime_info, persistent_buffer_info);

  NVF_ERROR(
      persistent_buffer_size.persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Float)));
  NVF_ERROR(
      persistent_buffer_size.projected_persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Float)));
}

TEST_F(PersistentBufferTest, FusionPersistentBufferCalculation2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = sum(tv1, {1});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = set(tv1);
  auto tv5 = add(tv3, tv4);
  auto tv6 = castOp(DataType::Half, tv5);
  fusion.addOutput(tv6);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  NVF_ERROR(buffers.size() == 1);
  NVF_ERROR(resolution.size() == 1 && resolution[0].size() == 1);
  NVF_ERROR(projectable.size() == 1);
  NVF_ERROR(projectable_inputs.size() == 1);

  NVF_ERROR(isTvWithinVec(buffers, tv1));
  NVF_ERROR(isTvWithinVec(projectable, tv1));
  NVF_ERROR(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  NVF_ERROR(tv1_resolution_it != resolution.end())

  NVF_ERROR(isTvWithinVec(*tv1_resolution_it, tv5));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0});
  auto persistent_buffer_size =
      persistentBufferSize(&fusion, runtime_info, persistent_buffer_info);

  NVF_ERROR(
      persistent_buffer_size.persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Float)));
  NVF_ERROR(
      persistent_buffer_size.projected_persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Half)));
}

TEST_F(PersistentBufferTest, FusionPersistentBufferCalculation3_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});

  auto tv5 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv5);

  auto tv6 = castOp(DataType::Float, tv5);

  auto tv7 = add(tv6, tv4);
  auto tv8 = set(tv1);
  auto tv9 = add(tv7, tv8);
  auto tv10 = sum(tv9, {1});
  auto tv11 = broadcast(tv10, {false, true});
  auto tv12 = set(tv7);
  auto tv13 = add(tv12, tv11);

  fusion.addOutput(tv13);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  NVF_ERROR(buffers.size() == 2);
  NVF_ERROR(
      resolution.size() == 2 && resolution[0].size() == 1 &&
      resolution[1].size() == 1);
  NVF_ERROR(projectable.size() == 2);
  NVF_ERROR(projectable_inputs.size() == 2);

  NVF_ERROR(isTvWithinVec(buffers, tv1) && isTvWithinVec(buffers, tv7));
  NVF_ERROR(isTvWithinVec(projectable, tv1) && isTvWithinVec(projectable, tv7));

  NVF_ERROR(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  NVF_ERROR(tv1_resolution_it != resolution.end())
  NVF_ERROR(isTvWithinVec(*tv1_resolution_it, tv9));

  auto tv7_resolution_it = tvEntryInVecVec(resolution, buffers, tv7);
  NVF_ERROR(tv7_resolution_it != resolution.end())
  NVF_ERROR(isTvWithinVec(*tv7_resolution_it, tv13));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);
  at::Tensor aten_t5 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0, aten_t5});
  auto persistent_buffer_size =
      persistentBufferSize(&fusion, runtime_info, persistent_buffer_info);

  NVF_ERROR(
      persistent_buffer_size.persistent_buffer_size ==
      static_cast<int64_t>(
          aten_t0.size(1) * dataTypeSize(DataType::Float) * 2));
  NVF_ERROR(
      persistent_buffer_size.projected_persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Half) * 2));
}

TEST_F(PersistentBufferTest, FusionPersistentBufferCalculation4_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = set(tv1);
  auto tv6 = add(tv4, tv5);
  auto tv7 = set(tv2);
  auto tv8 = add(tv7, tv6);
  auto tv9 = castOp(DataType::Half, tv8);

  fusion.addOutput(tv9);

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);

  auto isTvWithinVec = [](std::vector<TensorView*>& vec, TensorView* tv) {
    return std::find(vec.begin(), vec.end(), tv) != vec.end();
  };

  auto tvEntryInVecVec = [](std::vector<std::vector<TensorView*>>& vec_o_vec,
                            std::vector<TensorView*>& buffer_vec,
                            TensorView* tv) {
    auto buffer_it = std::find(buffer_vec.begin(), buffer_vec.end(), tv);
    return vec_o_vec.begin() + std::distance(buffer_vec.begin(), buffer_it);
  };

  auto& buffers = persistent_buffer_info.persistent_buffers;
  auto& resolution = persistent_buffer_info.persistent_buffer_resolution_points;
  auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
  auto& projectable_inputs = persistent_buffer_info.projectable_buffer_inputs;

  NVF_ERROR(buffers.size() == 2);
  NVF_ERROR(
      resolution.size() == 2 && resolution[0].size() == 1 &&
      resolution[1].size() == 1);

  NVF_ERROR(projectable.size() == 2);
  NVF_ERROR(projectable_inputs.size() == 1);

  NVF_ERROR(isTvWithinVec(buffers, tv1) && isTvWithinVec(buffers, tv2));
  NVF_ERROR(isTvWithinVec(projectable, tv1) && isTvWithinVec(projectable, tv2));

  NVF_ERROR(isTvWithinVec(projectable_inputs, tv0));

  auto tv1_resolution_it = tvEntryInVecVec(resolution, buffers, tv1);
  NVF_ERROR(tv1_resolution_it != resolution.end())
  NVF_ERROR(isTvWithinVec(*tv1_resolution_it, tv6));

  auto tv2_resolution_it = tvEntryInVecVec(resolution, buffers, tv2);
  NVF_ERROR(tv2_resolution_it != resolution.end())
  NVF_ERROR(isTvWithinVec(*tv2_resolution_it, tv8));

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  // Schedule through magic scheduler
  SchedulerRuntimeInfo runtime_info(&fusion, {aten_t0});
  auto persistent_buffer_size =
      persistentBufferSize(&fusion, runtime_info, persistent_buffer_info);

  // T1 and T2 are persistent buffers, but T2 can be projected to T1.
  // So, the actual buffer size is just the size to save T1.
  NVF_ERROR(
      persistent_buffer_size.persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Float)));

  NVF_ERROR(
      persistent_buffer_size.projected_persistent_buffer_size ==
      static_cast<int64_t>(aten_t0.size(1) * dataTypeSize(DataType::Half)));
}

TEST_F(PersistentBufferTest, FusionPersistentBufferProjection_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = set(tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = set(tv1);
  auto tv6 = add(tv4, tv5);
  auto tv7 = set(tv2);
  auto tv8 = add(tv7, tv6);
  auto tv9 = castOp(DataType::Half, tv8);

  fusion.addOutput(tv9);

  reduction_scheduler_utils::projectPersistentBuffers(&fusion, true);

  auto tv5_producers = ir_utils::producerTvsOf(tv5);
  auto tv7_producers = ir_utils::producerTvsOf(tv7);

  // Projection should have broken these dependencies

  NVF_ERROR(
      std::find(tv5_producers.begin(), tv5_producers.end(), tv1) ==
      tv5_producers.end());
  NVF_ERROR(
      std::find(tv7_producers.begin(), tv7_producers.end(), tv2) ==
      tv7_producers.end());

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor aten_t0 = at::randn({99, 101}, options);

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs({aten_t0});

  testValidate(&fusion, cg_outputs, {aten_t0}, __LINE__, __FILE__);
}

// Repro of issue #2381
TEST_F(PersistentBufferTest, FusionPersistentBufferProjection2_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv0);
  auto tv1 = makeSymbolicTensor(2, DataType::Half);
  fusion.addInput(tv1);

  auto tv2 = castOp(DataType::Float, tv0);
  auto tv3 = castOp(DataType::Float, tv1);
  auto tv4 = add(tv2, tv3);
  auto tv5 = sum(tv4, {1});
  auto tv6 = broadcast(tv5, {false, true});
  // Cast tv1 again
  auto tv7 = castOp(DataType::Float, tv1);
  // No error if this is done with tv3 rather than tv7
  auto tv8 = sub(tv6, tv7);
  auto tv9 = sub(tv8, tv4);
  auto tv10 = castOp(DataType::Half, tv9);
  fusion.addOutput(tv10);

  std::vector<int64_t> shape({10, 11});

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn(shape, options);
  at::Tensor t1 = at::randn(shape, options);

  // Persistent buffers: tv1, tv4
  // Projectable buffer: tv4
  // Projectable buffer inputs: tv0, tv1

  // tv1 is both a persistent buffer and an input to the projected
  // buffer of tv4. It is NOT considered as projectable.

  auto persistent_info = scheduler_utils::persistentBuffers(&fusion);

  NVF_CHECK(persistent_info.persistent_buffers.size() == 2);
  for (auto tv : persistent_info.persistent_buffers) {
    NVF_CHECK(
        tv == tv4 || tv == tv1,
        "Unexpected persistent buffer: ",
        tv->toString());
  }

  NVF_CHECK(persistent_info.projectable_persistent_buffers.size() == 1);
  for (auto tv : persistent_info.projectable_persistent_buffers) {
    NVF_CHECK(
        tv == tv4,
        "Unexpected projectable persistent buffer: ",
        tv->toString());
  }

  for (auto tv : persistent_info.projectable_buffer_inputs) {
    NVF_CHECK(
        tv == tv0 || tv == tv1,
        "Unexpected projectable buffer input: ",
        tv->toString());
  }

  SchedulerRuntimeInfo runtime_info(&fusion, {t0, t1});
  auto persistent_buffer_size =
      persistentBufferSize(&fusion, runtime_info, persistent_info);

  // Since tv1 is not projectable, it is included in the active mask
  // of projected buffers, even though it is also included in the
  // projectable buffer inputs. Thus, the buffer size would be
  // calculated as the sum of tv1, tv0 and tv1.
  auto projected_size = persistent_buffer_size.projected_persistent_buffer_size;
  auto expected_size =
      static_cast<int64_t>(shape[1] * 2 * dataTypeSize(DataType::Half));
  NVF_CHECK(
      projected_size == expected_size,
      "Buffer projection failure. Expected size: ",
      expected_size,
      ". Actual: ",
      projected_size);
}

// https://github.com/csarofeen/pytorch/issues/2321
TEST_F(
    PersistentBufferTest,
    FusionPersistentBufferProjectionAfterWelfordTranslate_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);
  const float kEps = 1e-5;
  Val* eps_ptr = IrBuilder::create<Val>(kEps);

  DataType dtype = DataType::Half;
  constexpr int64_t dim0 = 2048;
  constexpr int64_t dim1 = 10240;
  std::vector<int64_t> input_shape{dim0, dim1};
  std::vector<int64_t> norm_shape{dim1};
  auto input_half = makeContigTensor(2, dtype);
  auto weight_half = makeContigTensor(1, dtype);
  auto bias_half = makeContigTensor(1, dtype);
  fusion.addInput(input_half);
  fusion.addInput(weight_half);
  fusion.addInput(bias_half);
  auto input = castOp(DataType::Float, input_half);
  auto weight = castOp(DataType::Float, weight_half);
  auto bias = castOp(DataType::Float, bias_half);
  auto result = layer_norm(input, norm_shape, weight, bias, eps_ptr);
  auto result_output = castOp(dtype, result.output);
  fusion.addOutput(result_output);
  fusion.addOutput(result.mean);
  fusion.addOutput(result.invstd);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  c10::optional<at::Tensor> aten_weight = at::randn({input_shape[1]}, options);
  c10::optional<at::Tensor> aten_bias = at::randn({input_shape[1]}, options);
  auto aten_outputs = at::native_layer_norm(
      aten_input, norm_shape, aten_weight, aten_bias, kEps);

  // welford translate
  KernelArgumentHolder runtime_inputs =
      KernelArgumentHolder::createKernelArgumentHolder(
          {aten_input, aten_weight, aten_bias});
  bool isTranslated =
      SegmentCandidateFinder::translateWelfordInFusion(&fusion, runtime_inputs);
  NVF_ERROR(isTranslated);

  // persistent buffer should be projected to input
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);
  NVF_CHECK(
      persistent_buffer_info.projectable_persistent_buffers.size() == 1,
      "should have only one projectable_persistent_buffer!");
  NVF_CHECK(
      persistent_buffer_info.projectable_buffer_inputs.size() == 1,
      "should have only one projectable_buffer_inputs!");
  NVF_CHECK(
      persistent_buffer_info.projectable_buffer_inputs[0] == input_half,
      "persistent buffer should be projected to input!");

  // Check reduction axis is same for all reductions
  // Generate Launch Parameters
  auto reduction_params = getInnerPersistentHeuristics(
      &fusion, {aten_input, aten_weight, aten_bias});
  NVF_CHECK(reduction_params, "Reduction schedule was not generated!");

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs =
      fec.runFusionWithInputs({aten_input, aten_weight, aten_bias});

  testValidate(
      &fusion,
      cg_outputs,
      {aten_input, aten_weight, aten_bias},
      {std::get<0>(aten_outputs),
       std::get<1>(aten_outputs),
       std::get<2>(aten_outputs)},
      __LINE__,
      __FILE__,
      "");
}

// https://github.com/NVIDIA/Fuser/issues/335
// This test is to make sure the benchmark in layer_norm_fused.cpp is correctly
// implemented.
TEST_F(PersistentBufferTest, FusionLayerNormFusedOpsRedundantCast_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const float kEps = 1e-5;
  const int batch_size = 2048 * 8;
  const int hidden_size = 20480;
  DataType dtype = DataType::Half;
  {
    auto tv0 = makeContigTensor(1, dtype);
    auto tv1 = makeContigTensor(2, dtype);
    auto tv2 = makeContigTensor(1, dtype);
    auto tv3 = makeContigTensor(1, dtype);
    auto tv4 = makeContigTensor(1, dtype);

    fusion->addInput(tv0);
    fusion->addInput(tv1);
    fusion->addInput(tv2);
    fusion->addInput(tv3);
    fusion->addInput(tv4);
    auto tv5 = broadcast(tv0, {true, false});
    auto tv6 = castOp(DataType::Float, tv1);
    auto tv7 = castOp(DataType::Float, tv5);
    auto tv8 = add(tv6, tv7);
    auto tv9 = castOp(DataType::Half, tv8);
    auto tv10 = broadcast(tv2, {true, false});
    auto tv11 = castOp(DataType::Float, tv9);
    auto tv12 = castOp(DataType::Float, tv10);
    auto tv13 = add(tv11, tv12);
    auto tv14 = castOp(DataType::Half, tv13);
    auto tv15 = castOp(DataType::Float, tv14);
    auto tv16 = variance(tv15, {1}, false, false);
    auto tv17 = broadcast(tv16, {false, true});
    auto tv18 = sum(tv15, {1}, false);
    auto tv19 = broadcast(tv18, {false, true});

    nvfuser::Val* num_features = IrBuilder::create<Val>(1.0);
    num_features = mul(num_features, tv0->getLeafDomain()[0]->extent());
    auto s20 = num_features;

    auto s21 = reciprocal(s20);
    auto tv22 = mul(tv19, s21);
    auto s23 = IrBuilder::create<Val>(kEps);
    auto tv24 = add(tv17, s23);
    auto tv25 = rsqrt(tv24);
    auto tv26 = broadcast(tv22, {false, false});
    auto tv27 = castOp(DataType::Float, tv14);
    auto tv28 = sub(tv27, tv26);
    auto tv29 = broadcast(tv25, {false, false});
    auto tv30 = mul(tv28, tv29);
    auto tv31 = broadcast(tv4, {true, false});
    auto tv32 = castOp(DataType::Float, tv31);
    auto tv33 = mul(tv30, tv32);
    auto tv34 = broadcast(tv3, {true, false});
    auto tv35 = castOp(DataType::Float, tv34);
    auto tv36 = add(tv33, tv35);
    auto tv37 = castOp(DataType::Half, tv36);
    fusion->addOutput(tv37);
  }

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  std::vector<c10::IValue> inputs;
  std::vector<at::Tensor> outputs;

  {
    auto t0 = at::randn({hidden_size}, options);
    auto t1 = at::randn({batch_size, hidden_size}, options);
    auto t2 = at::randn({hidden_size}, options);
    auto t3 = at::randn({hidden_size}, options);
    auto t4 = at::randn({hidden_size}, options);
    inputs.emplace_back(t0);
    inputs.emplace_back(t1);
    inputs.emplace_back(t2);
    inputs.emplace_back(t3);
    inputs.emplace_back(t4);
    auto t5 = t0.unsqueeze(0).expand({batch_size, hidden_size});
    auto t6 = t1.to(at::kFloat);
    auto t7 = t5.to(at::kFloat);
    auto t8 = at::add(t6, t7);
    auto t9 = t8.to(at::kHalf);
    auto t10 = t2.unsqueeze(0).expand({batch_size, hidden_size});
    auto t11 = t9.to(at::kFloat);
    auto t12 = t10.to(at::kFloat);
    auto t13 = at::add(t11, t12);
    auto t14 = t13.to(at::kHalf);
    auto aten_outputs = at::native_layer_norm(t14, {hidden_size}, t4, t3, kEps);
    auto t33 = std::get<0>(aten_outputs);
    outputs.emplace_back(t33);
  }

  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  NVF_CHECK(
      persistent_buffer_info.persistent_buffers.size() == 2,
      "Before project to other buffers, should have two persistent buffers!");

  // The buffer size should only count 1 buffer because the other one is
  // projected to its producer.
  SchedulerRuntimeInfo runtime_info(fusion, inputs);
  auto persistent_buffer_size =
      persistentBufferSize(fusion, runtime_info, persistent_buffer_info);
  NVF_CHECK(
      persistent_buffer_size.persistent_buffer_size ==
          hidden_size * dataTypeSize(dtype),
      "Persistent buffer size is not correct!");

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs(inputs);
  testValidate(fusion, cg_outputs, inputs, outputs, __LINE__, __FILE__);
}

TEST_F(PersistentBufferTest, FusionRecomputePersistentBuffer_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 1024;
  const int hidden_size = 2048;
  {
    DataType dtype = DataType::Float;
    auto tv0 = makeContigTensor(2, dtype);
    auto tv1 = makeContigTensor(2, dtype);
    fusion->addInput(tv0);
    fusion->addInput(tv1);

    auto tv2 = add(tv0, tv1);
    auto tv3 = castOp(DataType::Half, tv2);

    auto tv4 = castOp(DataType::Float, tv3);
    auto tv5 = sum(tv4, {1});
    auto tv6 = broadcast(tv5, {false, true});
    auto tv7 = add(tv4, tv6);

    auto tv8 = castOp(DataType::Float, tv3);
    auto tv9 = add(tv6, tv8);

    fusion->addOutput(tv7);
    fusion->addOutput(tv9);
  }

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  std::vector<c10::IValue> inputs;
  std::vector<at::Tensor> outputs;

  {
    auto t0 = at::randn({batch_size, hidden_size}, options);
    auto t1 = at::randn({batch_size, hidden_size}, options);
    inputs.emplace_back(t0);
    inputs.emplace_back(t1);

    auto t2 = t0.add(t1);
    auto t3 = t2.to(at::kHalf);
    auto t4 = t3.to(at::kFloat);
    auto t5 = t4.sum({1});
    auto t6 = t5.unsqueeze(1).expand({batch_size, hidden_size});
    auto t7 = t4.add(t6);
    auto t8 = t3.to(at::kFloat);
    auto t9 = t8.add(t6);

    outputs.emplace_back(t7);
    outputs.emplace_back(t9);
  }

  auto persistent_buffer_info1 = scheduler_utils::persistentBuffers(fusion);
  NVF_CHECK(
      persistent_buffer_info1.persistent_buffers.size() == 2,
      "Before project to other buffers, should have two persistent buffers!");

  reduction_scheduler_utils::projectPersistentBuffers(fusion, false);
  auto persistent_buffer_info2 = scheduler_utils::persistentBuffers(fusion);
  NVF_CHECK(
      persistent_buffer_info2.persistent_buffers.size() == 1,
      "After project to other buffers, should have one persistent buffer!");

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs(inputs);
  testValidate(fusion, cg_outputs, inputs, outputs, __LINE__, __FILE__);
}

TEST_F(PersistentBufferTest, ProjectPersistentBufferMultiScopes) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 2048;
  const int hidden_size = 10240;
  DataType input_dtype = DataType::Float;
  auto tv0 = makeContigTensor(2, input_dtype);
  auto tv1 = makeContigTensor(2, input_dtype);
  auto tv2 = makeContigTensor(2, input_dtype);

  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);

  auto tv3 = add(tv0, tv0);
  auto tv4 = sum(tv3, {1});
  auto tv5 = broadcast(tv4, {false, true});
  auto tv6 = add(tv3, tv5);

  auto tv7 = add(tv3, tv3);
  auto tv8 = sum(tv7, {1});
  auto tv9 = broadcast(tv8, {false, true});
  auto tv10 = add(tv7, tv9);

  auto tv11 = add(tv0, tv1);
  auto tv12 = mul(tv11, tv11);
  auto tv13 = sum(tv12, {1});
  auto tv14 = broadcast(tv13, {false, true});
  auto tv15 = add(tv12, tv14);

  auto tv16 = add(tv12, tv2);
  auto tv17 = mul(tv16, tv16);
  auto tv18 = sum(tv17, {1});
  auto tv19 = broadcast(tv18, {false, true});
  auto tv20 = add(tv17, tv19);

  fusion->addOutput(tv6);
  fusion->addOutput(tv10);
  fusion->addOutput(tv15);
  fusion->addOutput(tv20);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({batch_size, hidden_size}, options);
  auto t1 = at::randn({batch_size, hidden_size}, options);
  auto t2 = at::randn({batch_size, hidden_size}, options);
  std::vector<c10::IValue> inputs{t0, t1, t2};

  // The persistent buffers in this fusion are: tv3, tv7, tv12, and tv17. Note
  // that tv7 can be projected back to its producer, tv3. When calculating the
  // total size of persistent buffers ([persistent_buffer_size]), it's important
  // to consider the active scopes of these buffers. Simply subtracting the
  // buffer size of tv7 from the max buffer size may lead to an underestimation.
  // This is because there are two distinct scopes in this computation: (1)
  // During the calculation of tv10, the active persistent buffers are tv3 and
  // tv7. (2) For the calculation of tv20, the active persistent buffers are
  // tv12 and tv17. The max buffer size is based on tv12 and tv17. There is no
  // projectable buffer needs to be deducted in this scope.
  auto persistent_info = scheduler_utils::persistentBuffers(fusion);
  SchedulerRuntimeInfo runtime_info(fusion, inputs);
  auto persistent_buffer_size =
      persistentBufferSize(fusion, runtime_info, persistent_info);
  auto calculated_size = persistent_buffer_size.persistent_buffer_size;
  auto expected_size =
      static_cast<int64_t>(hidden_size * 2 * dataTypeSize(input_dtype));
  NVF_CHECK(
      calculated_size == expected_size,
      "Buffer size calculation failure. Expected size: ",
      expected_size,
      ". Actual: ",
      calculated_size);
  auto persistent_params = getInnerPersistentHeuristics(fusion, inputs);
  NVF_CHECK(persistent_params, "Reduction schedule was not generated!");
  NVF_CHECK(
      !persistent_params->project_persistent_buffers,
      "Shouldn't project persistent buffers to inputs!");
  scheduleInnerPersistentKernel(fusion, *persistent_params);
  FusionExecutor fe;
  fe.compileFusion(fusion, inputs);
  auto cg_outputs = fe.runFusion(inputs);
}

TEST_F(PersistentBufferTest, ChainProjectionToPersistentProducer) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 2048;
  const int hidden_size = 10240;
  DataType input_dtype = DataType::Half;
  auto tv0 = makeContigTensor(2, input_dtype);
  auto tv1 = makeContigTensor(2, input_dtype);
  auto tv2 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = castOp(DataType::Float, tv0);
  auto tv4 = castOp(DataType::Float, tv1);
  auto tv5 = castOp(DataType::Float, tv2);

  // tv7 is persistent
  auto tv6 = add(tv3, tv4);
  auto tv7 = add(tv6, tv5);
  auto tv8 = sum(tv7, {1});
  auto tv9 = broadcast(tv8, {false, true});
  auto tv10 = add(tv7, tv9);

  // tv11 is persistent, and can be projected to tv7
  auto tv11 = add(tv7, tv7);
  auto tv12 = sum(tv11, {1});
  auto tv13 = broadcast(tv12, {false, true});
  auto tv14 = add(tv11, tv13);

  // tv15 is persistent, and can be projected to tv11
  auto tv15 = add(tv11, tv11);
  auto tv16 = sum(tv15, {1});
  auto tv17 = broadcast(tv16, {false, true});
  auto tv18 = add(tv17, tv15);

  fusion->addOutput(tv10);
  fusion->addOutput(tv14);
  fusion->addOutput(tv18);

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({batch_size, hidden_size}, options);
  auto t1 = at::randn({batch_size, hidden_size}, options);
  auto t2 = at::randn({batch_size, hidden_size}, options);
  std::vector<c10::IValue> inputs{t0, t1, t2};
  auto t3 = t0.to(at::kFloat) + t1.to(at::kFloat) + t2.to(at::kFloat);
  auto t4 = at::sum(t3, {1}, true);
  auto t5 = t3 + t4;
  auto t6 = t3 + t3;
  auto t7 = at::sum(t6, {1}, true);
  auto t8 = t6 + t7;
  auto t9 = t6 + t6;
  auto t10 = at::sum(t9, {1}, true);
  auto t11 = t9 + t10;

  // There are 3 persistent buffers: tv7, tv11, and tv15.
  // The PersistentBufferProjector should firstly project
  // tv15 to tv11, then project tv11 to tv7.
  // After projection, tv7 is the only buffer.
  auto persistent_info = scheduler_utils::persistentBuffers(fusion);
  SchedulerRuntimeInfo runtime_info(fusion, inputs);
  auto persistent_buffer_size =
      persistentBufferSize(fusion, runtime_info, persistent_info);
  auto calculated_size = persistent_buffer_size.persistent_buffer_size;
  auto expected_size =
      static_cast<int64_t>(hidden_size * dataTypeSize(DataType::Float));
  NVF_CHECK(
      calculated_size == expected_size,
      "Buffer size calculation failure. Expected size: ",
      expected_size,
      ". Actual: ",
      calculated_size);

  // If project to inputs, there are 3 fp16 tvs, which is larger than 1 fp32.
  // So, shouldn't project to inputs.
  auto persistent_params = getInnerPersistentHeuristics(fusion, inputs);
  NVF_CHECK(persistent_params, "Reduction schedule was not generated!");
  NVF_CHECK(
      !persistent_params->project_persistent_buffers,
      "Shouldn't project persistent buffers to inputs!");
  scheduleInnerPersistentKernel(fusion, *persistent_params);
  FusionExecutor fe;
  fe.compileFusion(fusion, inputs);
  auto cg_outputs = fe.runFusion(inputs);
  testValidate(fusion, cg_outputs, inputs, {t5, t8, t11}, __LINE__, __FILE__);
}

// Test the persistent buffers in softmax are projected back to inputs.
TEST_F(PersistentBufferTest, SoftmaxProjectToInput) {
  auto test_softmax = [](int batch, int feature, DataType dtype) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    const int kReductionAxis = 1;
    std::vector<int64_t> input_shape{batch, feature};
    TensorView* input = makeContigTensor(input_shape.size(), dtype);
    fusion.addInput(input);
    if (dtype == DataType::Half) {
      input = castOp(DataType::Float, input);
    }
    auto output = softmax(input, kReductionAxis);
    if (dtype == DataType::Half) {
      output = castOp(DataType::Half, output);
    }
    fusion.addOutput(output);

    // There should be 2 projectable persistent buffers.
    auto persistent_buffer_info = scheduler_utils::persistentBuffers(&fusion);
    auto& projectable = persistent_buffer_info.projectable_persistent_buffers;
    NVF_ERROR(projectable.size() == 2);

    auto options = at::TensorOptions()
                       .dtype(data_type_to_aten(dtype))
                       .device(at::kCUDA, 0);
    at::Tensor aten_input = at::randn(input_shape, options);
    auto aten_output =
        at::_softmax(aten_input.to(at::kDouble), kReductionAxis, false);

    auto reduction_params = getInnerPersistentHeuristics(&fusion, {aten_input});
    NVF_CHECK(reduction_params, "Reduction schedule was not generated!");
    // 24576 is the threshold to project to inputs. see deriviation in
    // projectBufferToInputs()
    bool should_project_to_input =
        feature * dataTypeSize(DataType::Float) > 24576l;
    NVF_CHECK(
        reduction_params->project_persistent_buffers == should_project_to_input,
        should_project_to_input ? "Should project to inputs!"
                                : "Shouldn't project to inputs!");
    scheduleInnerPersistentKernel(&fusion, *reduction_params);
    auto lparams = reduction_params->lparams;
    nvfuser::FusionExecutor fe;
    fe.compileFusion(&fusion, {aten_input}, lparams);
    auto cg_outputs = fe.runFusion({aten_input}, lparams);

    testValidate(
        &fusion,
        cg_outputs,
        {aten_input},
        {aten_output},
        __LINE__,
        __FILE__,
        "",
        lparams);
  };
  const int batch = 2048;
  std::vector<int> features = {6 * 1024, 10240};
  for (auto feature : features) {
    test_softmax(batch, feature, DataType::Half);
  }
}

// Test projection to inputs when there are three persistent buffers.
TEST_F(PersistentBufferTest, ProjectToInputsAndBroadcastTvs1) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 128;
  const int hidden_size = 10240;
  DataType input_dtype = DataType::Half;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);
  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = add(tv1, tv1);
  auto tv3 = sum(tv2, {1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = div(tv2, tv4);

  auto tv6 = add(tv5, tv5);
  auto tv7 = sum(tv6, {1});
  auto tv8 = broadcast(tv7, {false, true});
  auto tv9 = div(tv6, tv8);

  auto tv10 = add(tv9, tv9);
  auto tv11 = sum(tv10, {1});
  auto tv12 = broadcast(tv11, {false, true});
  auto tv13 = div(tv10, tv12);

  fusion->addOutput(tv5);
  fusion->addOutput(tv9);
  fusion->addOutput(tv13);

  // The persistent buffers in this fusion are: tv2, tv6, and tv10.
  // tv2 is projected to input.
  // tv6 is projected to input and tv4 which is a broadcast tv.
  // tv10 is projected to input, tv4 and tv8 which are broadcast tvs.
  // The only actual persisent buffer is the cached input.
  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({batch_size, hidden_size}, options);
  std::vector<c10::IValue> inputs{t0};

  auto persistent_params = getInnerPersistentHeuristics(fusion, inputs);
  NVF_CHECK(persistent_params, "Reduction schedule was not generated!");
  NVF_CHECK(
      persistent_params->project_persistent_buffers,
      "Should project persistent buffers to inputs!");

  scheduleInnerPersistentKernel(fusion, *persistent_params);
  FusionExecutor fe;
  fe.compileFusion(fusion, inputs);
  auto cg_outputs = fe.runFusion(inputs);
}

// Test projection to inputs when the persistent buffer is a broadcast tv.
TEST_F(PersistentBufferTest, ProjectToInputsAndBroadcastTvs2) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int batch_size = 128;
  const int hidden_size = 8192;
  DataType input_dtype = DataType::Half;
  auto tv0 = makeContigTensor(2, input_dtype);
  fusion->addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = exp(tv1);
  auto tv3 = sum(tv2, {-1});
  auto tv4 = broadcast(tv3, {false, true});
  auto tv5 = add(tv2, tv4);
  fusion->addOutput(tv5);

  auto tv6 = broadcast(tv5, {true, false, false});
  auto tv7 = sum(tv6, {-1});
  auto tv8 = broadcast(tv7, {false, false, true});
  auto tv9 = add(tv6, tv8);
  fusion->addOutput(tv9);

  // In this fusion, tv6 is a persistent buffer with a broadcast dim.
  // Between reduction tv2 and tv6, there are two broadcast tvs: tv4 and tv6.
  // Only tv4 is a valid broadcast tv to project to.
  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  const auto& [can_project, broadcast_tvs] =
      scheduler_utils::canProjectToInputsWithoutReduction(reduction_tvs, tv6);
  NVF_CHECK(
      can_project, "Expect can project to inputs to be true but got false!");
  NVF_CHECK(
      broadcast_tvs.size() == 1,
      "Expect one target broadcast_tv!, Got: ",
      broadcast_tvs.size());
  NVF_CHECK(
      broadcast_tvs.at(0) == tv4,
      "Expect target tv4!, Got: ",
      broadcast_tvs.at(0)->toString());

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({batch_size, hidden_size}, options);
  std::vector<c10::IValue> inputs{t0};

  auto persistent_params = getInnerPersistentHeuristics(fusion, inputs);
  NVF_CHECK(persistent_params, "Reduction schedule was not generated!");
  NVF_CHECK(
      persistent_params->project_persistent_buffers,
      "Should project persistent buffers to inputs!");

  scheduleInnerPersistentKernel(fusion, *persistent_params);
  FusionExecutor fe;
  fe.compileFusion(fusion, inputs, persistent_params->lparams);
  auto cg_outputs = fe.runFusion(inputs, persistent_params->lparams);
}

TEST_F(PersistentBufferTest, ProjectToInputsAndBroadcastTvs3) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  const int dim0 = 128;
  const int dim1 = 32;
  const int dim2 = 256;
  DataType input_dtype = DataType::Half;
  auto tv0 = makeContigTensor(3, input_dtype);
  fusion->addInput(tv0);

  auto tv1 = castOp(DataType::Float, tv0);
  auto tv2 = sum(tv1, {1, 2});
  auto tv3 = broadcast(tv2, {false, true});
  auto tv4 = broadcast(tv3, {false, false, true});
  auto tv5 = add(tv1, tv4);
  fusion->addOutput(tv5);

  auto tv6 = exp(tv5);
  auto tv7 = sum(tv6, {1, 2});
  auto tv8 = broadcast(tv7, {false, true, true});
  auto tv9 = add(tv6, tv8);
  fusion->addOutput(tv9);

  auto tv10 = add(tv5, tv9);
  auto tv11 = sum(tv10, {1, 2});
  auto tv12 = broadcast(tv11, {false, true, true});
  auto tv13 = add(tv10, tv12);
  fusion->addOutput(tv13);

  const auto& reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  // (1) Test projection to inputs when there are two broadcast tvs (tv3 and
  // tv4) between the reduction tv (tv2) and the persistent buffer (tv6). Should
  // only project to tv4.
  const auto& [can_project, broadcast_tvs] =
      scheduler_utils::canProjectToInputsWithoutReduction(reduction_tvs, tv6);
  NVF_CHECK(
      can_project, "Expect can project to inputs to be true but got false!");
  NVF_CHECK(
      broadcast_tvs.size() == 1,
      "Expect one target broadcast_tv!, Got: ",
      broadcast_tvs.size());
  NVF_CHECK(
      broadcast_tvs.at(0) == tv4,
      "Expect target tv4!, Got: ",
      broadcast_tvs.at(0)->toString());

  // (2) Test projection to inputs when the persistent buffer (tv10) depends on
  // two reduction tvs (tv2 and tv7). Should project to tv4 and tv8.
  const auto& [tv10_can_project, tv10_broadcast_tvs] =
      scheduler_utils::canProjectToInputsWithoutReduction(reduction_tvs, tv10);
  NVF_CHECK(
      tv10_can_project,
      "Expect can project to inputs to be true but got false!");
  NVF_CHECK(
      tv10_broadcast_tvs.size() == 2,
      "Expect two target broadcast_tv!, Got: ",
      tv10_broadcast_tvs.size());
  NVF_CHECK(
      tv10_broadcast_tvs.at(0) == tv4,
      "Expect target tv4!, Got: ",
      tv10_broadcast_tvs.at(0)->toString());
  NVF_CHECK(
      tv10_broadcast_tvs.at(1) == tv8,
      "Expect target tv8!, Got: ",
      tv10_broadcast_tvs.at(1)->toString());

  auto options = at::TensorOptions()
                     .dtype(data_type_to_aten(input_dtype))
                     .device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1, dim2}, options);
  std::vector<c10::IValue> inputs{t0};

  auto persistent_params = getInnerPersistentHeuristics(fusion, inputs);
  NVF_CHECK(persistent_params, "Reduction schedule was not generated!");
  NVF_CHECK(
      persistent_params->project_persistent_buffers,
      "Should project persistent buffers to inputs!");
  scheduleInnerPersistentKernel(fusion, *persistent_params);
  FusionExecutor fe;
  fe.compileFusion(fusion, inputs, persistent_params->lparams);
  auto cg_outputs = fe.runFusion(inputs, persistent_params->lparams);
}

} // namespace nvfuser
