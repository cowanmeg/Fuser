// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <cmath>

#include <gtest/gtest.h>

#include <executor.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/interface_nodes.h>
#include <ir/utils.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <preseg_passes/allocation_order_inference.h>
#include <preseg_passes/move_split_cat.h>
#include <preseg_passes/optimization_pass.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

int64_t D = 1, B = 2, E = 768, H = 12, S = 128;
// Note parameters scaled by kParamScale following weight initialization
// recommendations:
// https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.initializer_range
constexpr double kDropoutProb = 0.1, kParamScale = 0.02;

class DistributedTransformerTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<DataType> {
 protected:
  DistributedTransformerTest()
      : optimization_guard_(false), allocation_order_guard_(false) {
    D = communicator_->size();
    // Ensure test runs by setting E and H appropriately
    if ((H % D) != 0) {
      H = D * 2;
    }
    if ((E % D) != 0) {
      E = D * 64;
    }
  }

  void SetUp() {
    MultiDeviceTest::SetUp();
    if (!deviceMajorMinorCheck(8)) {
      GTEST_SKIP() << "Distributed transformer tests require Ampere or newer";
    }
  }

  hir::HostIrExecutorParams executor_params_{
      .use_fusion_executor_cache = true,
      .skip_auto_scheduling = false,
      .cache_fusion_executor = false};

 private:
  // Note: `MoveSplitCat` and `AllocationDomain` preseg passes use ID model.
  // `SdpaFwdOp` currently does not work with ID model since it requires all
  // sibling outputs to have the same root domain.
  //  This will be modified in a future PR.
  preseg_passes::OptimizationPassGuard<preseg_passes::MoveSplitCatPass>
      optimization_guard_;
  preseg_passes::OptimizationPassGuard<preseg_passes::AllocationDomainPass>
      allocation_order_guard_;
};

namespace {
TensorView* replicated_dropout(
    TensorView* x,
    const double kProb,
    Fusion* fusion,
    DeviceMesh mesh) {
  // Need to modify two things before we can use the existing dropout function
  // in composite.cpp (1) Sharding propagation breaks at rand_like because it
  // creates a fresh TV. (2) The philox seed and offset must be set to ensure
  // the masks are identical across processes.
  TensorView* x_float = castOp(DataType::Float, x);
  const double kScale = 1.0 / (1.0 - kProb);
  Val* philox_seed = fusion->zeroVal();
  Val* philox_offset = fusion->zeroVal();
  TensorView* rand_vals = rand_like(x_float, philox_seed, philox_offset);
  TensorView* mask = lt(rand_vals, IrBuilder::create<Val>(1.0 - kProb));
  TensorView* apply_mask = mul(x_float, mask);
  TensorView* dropout = mul(apply_mask, IrBuilder::create<Val>(kScale));
  rand_vals->setDeviceMesh(mesh);
  return dropout;
}

void validate(
    std::vector<at::Tensor> expected_out,
    std::vector<at::Tensor> out) {
  EXPECT_EQ(expected_out.size(), out.size());
  for (auto i : c10::irange(out.size())) {
    // Note: Scaling tolerance up since the error accumulates across ops
    // BFloat16 error is quite high, but the program has been verified with
    // double precision to be logically correct.
    double atol = 0.075 * (i + 1);
    double rtol = 1.6e-2;
    auto all_close = out[i]
                         .to(expected_out[i].dtype())
                         .allclose(
                             expected_out[i],
                             rtol,
                             atol,
                             /*equal_nan=*/true);

    if (!all_close) {
      auto error = (out[i].to(expected_out[i].dtype()) - expected_out[i]).abs();
      auto max_error = error.max().item().to<double>();
      auto max_relative_error =
          (max_error / expected_out[i].abs().max()).item();
      auto error_count =
          at::sum(error >= (atol + expected_out[i].abs() * rtol)).item();
      std::cout << "output[" << i << "] max error: " << max_error << std::endl;
      std::cout << "          max relative error: " << max_relative_error
                << std::endl;
      std::cout << "          failing elements: " << error_count << ", "
                << error_count.to<float>() / at::numel(out[i]) * 100.0
                << "\% of tensor" << std::endl;
    }
    EXPECT_TRUE(all_close);
  }
}
} // namespace

std::vector<at::Tensor> reference_mlp(
    at::Tensor x,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::Tensor b1,
    at::ScalarType at_dtype) {
  at::manual_seed(0);
  auto linear1 = at::matmul(x, w0).add(b0).to(at::kFloat);
  auto gelu = at::gelu(linear1, "tanh");
  auto linear2 = at::matmul(gelu.to(at_dtype), w1).add(b1).to(at::kFloat);
  auto dropout = at::dropout(linear2, kDropoutProb, true);
  return {linear1, gelu, linear2, dropout};
}

std::vector<at::Tensor> reference_mlp_backwards(
    at::Tensor grad,
    at::Tensor x,
    at::Tensor mask,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::ScalarType at_dtype) {
  // recompute activations
  auto linear0 = at::matmul(x, w0).add(b0).to(at::kFloat);
  auto gelu = at::gelu(linear0, "tanh");

  // backwards pass
  auto dropout_grad =
      at::native_dropout_backward(grad, mask, 1.0 / (1.0 - kDropoutProb));
  auto dropout_grad_q = dropout_grad.to(at_dtype);
  auto matmul1_grad = at::matmul(dropout_grad_q, w1.transpose(0, 1));
  auto matmul1_grad_w =
      at::matmul(dropout_grad_q.transpose(0, 1), gelu.to(at_dtype))
          .transpose(0, 1);
  auto matmul1_grad_b = at::sum(dropout_grad, {0});
  auto gelu_grad =
      at::gelu_backward(matmul1_grad.to(at::kFloat), linear0, "tanh");
  auto gelu_grad_q = gelu_grad.to(at_dtype);
  auto matmul0_grad_b = at::sum(gelu_grad, {0});
  auto matmul0_grad = at::matmul(gelu_grad_q, w0.transpose(0, 1));
  auto matmul0_grad_w =
      at::matmul(gelu_grad_q.transpose(0, 1), x).transpose(0, 1);

  std::vector<at::Tensor> grads = {
      dropout_grad,
      matmul1_grad_w,
      matmul1_grad_b,
      gelu_grad,
      matmul0_grad_w,
      matmul0_grad_b,
      matmul0_grad};
  return grads;
}

std::vector<TensorView*> mlp(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    Fusion* fusion,
    DeviceMesh& mesh,
    DataType dtype) {
  // Linear #1
  TensorView* matmul1 = matmul(x, w0);
  TensorView* b0_bcast = broadcast(b0, {false, true, false});
  TensorView* linear1 = add(matmul1, b0_bcast);
  // GeLU
  TensorView* linear1_ = castOp(DataType::Float, linear1);
  TensorView* gelu = tanh_gelu(linear1_);
  TensorView* gelu_ = castOp(dtype, gelu);
  // Linear #2
  TensorView* local_matmul2 = matmul(gelu_, w1);
  TensorView* matmul2 = sum(local_matmul2, {0}); // Allreduce
  TensorView* bcast_bias = broadcast(b1, {true, false});
  TensorView* linear2 = add(matmul2, bcast_bias);
  // Dropout
  TensorView* dropout = replicated_dropout(linear2, kDropoutProb, fusion, mesh);

  // Sharding
  // (TODO) TVs where sharding propagation breaks down:
  // linear_int0: broadcasts where a device dim axis is broadcasted.
  // rand_vals: rand_like creates a fresh new TV.
  // TVs replicated on each device.
  for (auto tv : {x, b1, matmul2, linear2, dropout}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w0, b0, w1, linear1, gelu}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  return {linear1, gelu, linear2, dropout};
}

std::vector<TensorView*> mlp_backwards(
    TensorView* grad,
    TensorView* x,
    TensorView* mask,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    Fusion* fusion,
    DeviceMesh& mesh,
    DataType dtype) {
  // Activation recomputation
  TensorView* matmul0 = matmul(x, w0);
  TensorView* b0_bcast = broadcast(b0, {false, true, false});
  TensorView* linear0 = add(matmul0, b0_bcast);
  linear0 = castOp(DataType::Float, linear0);
  TensorView* gelu = tanh_gelu(linear0);
  gelu = castOp(dtype, gelu);

  // Backwards pass
  constexpr double kScale = 1.0 / (1.0 - kDropoutProb);
  Val* dscale = IrBuilder::create<Val>(kScale);
  TensorView* dropout_grad = dropout_backward(grad, mask, dscale);
  TensorView* dropout_grad_q = castOp(dtype, dropout_grad);

  TensorView* w1_t = transpose(w1, 1, 2);
  TensorView* matmul1_grad_x = matmul(dropout_grad_q, w1_t);
  TensorView* grad_t = transpose(dropout_grad_q, 0, 1);
  TensorView* matmul1_grad_w_t = matmul(grad_t, gelu);
  TensorView* matmul1_grad_w = transpose(matmul1_grad_w_t, 1, 2);
  TensorView* matmul1_grad_b = sum(dropout_grad, {0});

  TensorView* matmul1_grad_x_ = castOp(DataType::Float, matmul1_grad_x);
  TensorView* gelu_grad = tanh_gelu_backward(matmul1_grad_x_, linear0);
  TensorView* gelu_grad_f = castOp(dtype, gelu_grad);

  TensorView* w0_t = transpose(w0, 1, 2);
  TensorView* matmul0_grad_x_partial = matmul(gelu_grad_f, w0_t);
  TensorView* matmul0_grad_x = sum(matmul0_grad_x_partial, {0}); // allreduce
  TensorView* grad_gelu_t = transpose(gelu_grad_f, 1, 2);
  TensorView* matmul0_grad_w_t = matmul(grad_gelu_t, x);
  TensorView* matmul0_grad_w = transpose(matmul0_grad_w_t, 1, 2);
  TensorView* matmul0_grad_b = sum(gelu_grad, {1});

  for (auto tv :
       {x,
        grad,
        mask,
        dropout_grad,
        matmul1_grad_x,
        matmul1_grad_b,
        matmul0_grad_x}) {
    tv->setDeviceMesh(mesh);
  }

  for (auto tv :
       {w0,
        b0,
        w1,
        matmul1_grad_x,
        matmul1_grad_w,
        matmul1_grad_w_t,
        gelu_grad,
        matmul0_grad_w_t,
        matmul0_grad_w,
        matmul0_grad_x_partial,
        matmul0_grad_b}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  std::vector<TensorView*> outputs = {
      dropout_grad,
      matmul1_grad_w,
      matmul1_grad_b,
      gelu_grad,
      matmul0_grad_w,
      matmul0_grad_b,
      matmul0_grad_x};
  return outputs;
}

TEST_P(DistributedTransformerTest, MLP_Layer) {
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* tvx = makeContigTensor(2, dtype);
  TensorView* tvw0 = makeContigTensor(3, dtype);
  TensorView* tvb0 = makeContigTensor(2, dtype);
  TensorView* tvw1 = makeContigTensor(3, dtype);
  TensorView* tvb1 = makeContigTensor(1, dtype);

  fusion->addInput(tvx);
  fusion->addInput(tvw0);
  fusion->addInput(tvb0);
  fusion->addInput(tvw1);
  fusion->addInput(tvb1);

  std::vector<TensorView*> tvsout =
      mlp(tvx, tvw0, tvb0, tvw1, tvb1, fusion.get(), mesh, dtype);

  for (TensorView* tv : tvsout) {
    fusion->addOutput(tv);
  }

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x = at::randn({B * S, E}, options);
  auto w0 = at::randn({E, 4 * E}, options) * kParamScale;
  auto b0 = at::randn({4 * E}, options) * kParamScale;
  auto w1 = at::randn({4 * E, E}, options) * kParamScale;
  auto b1 = at::randn({E}, options) * kParamScale;

  std::vector<at::Tensor> reference_outs =
      reference_mlp(x, w0, b0, w1, b1, at_dtype);

  std::vector<c10::IValue> inputs = {
      x,
      shardTensor(w0, 1, mesh),
      shardTensor(b0, 0, mesh),
      shardTensor(w1, 0, mesh),
      b1};

  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0], 1, mesh),
      shardTensor(reference_outs[1], 1, mesh),
      reference_outs[2],
      reference_outs[3]};

  at::manual_seed(0);
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);
  validate(expected_outputs, outputs);
}

TEST_P(DistributedTransformerTest, MLP_Backward) {
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* grad = makeContigTensor(2, DataType::Float);
  TensorView* x = makeContigTensor(2, dtype);
  TensorView* mask = makeContigTensor(2, DataType::Bool);
  TensorView* w0 = makeContigTensor(3, dtype);
  TensorView* b0 = makeContigTensor(2, dtype);
  TensorView* w1 = makeContigTensor(3, dtype);

  fusion->addInput(grad);
  fusion->addInput(x);
  fusion->addInput(mask);
  fusion->addInput(w0);
  fusion->addInput(b0);
  fusion->addInput(w1);

  std::vector<TensorView*> tv_outs =
      mlp_backwards(grad, x, mask, w0, b0, w1, fusion.get(), mesh, dtype);

  for (TensorView* tv : tv_outs) {
    fusion->addOutput(tv);
  }

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto grad_ = at::randn({B * S, E}, options).to(at::kFloat);
  auto x_ = at::randn({B * S, E}, options);
  auto mask_ = at::randn({B * S, E}, options).lt(1.0 - kDropoutProb);
  auto mlp_w0_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b0_ = at::randn({4 * E}, options) * kParamScale;
  auto mlp_w1_ = at::randn({4 * E, E}, options) * kParamScale;

  std::vector<at::Tensor> outs = reference_mlp_backwards(
      grad_, x_, mask_, mlp_w0_, mlp_b0_, mlp_w1_, at_dtype);

  std::vector<c10::IValue> inputs = {
      grad_,
      x_,
      mask_,
      shardTensor(mlp_w0_, 1, mesh),
      shardTensor(mlp_b0_, 0, mesh),
      shardTensor(mlp_w1_, 0, mesh)};
  std::vector<at::Tensor> expected_outputs = {
      outs[0], // dropout grad
      shardTensor(outs[1], 0, mesh), // linear1 weight grad
      outs[2], // linear1 bias grad
      shardTensor(outs[3], 1, mesh), // gelu grad
      shardTensor(outs[4], 1, mesh), // linear0 weight grad
      shardTensor(outs[5], 0, mesh), // linear0 bias grad
      outs[6]}; // linear0 grad

  at::manual_seed(0);
  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  validate(expected_outputs, outputs);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    DistributedTransformerTest,
    testing::Values(DataType::Half, DataType::BFloat16));
} // namespace nvfuser
