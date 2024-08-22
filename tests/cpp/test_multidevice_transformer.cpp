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
#include <preseg_passes/propagate_shardings.h>
#include <scheduler/mma_utils.h>
#include <scheduler/utils.h>
#include <tests/cpp/multidevice.h>
#include <tests/cpp/validator.h>

namespace nvfuser {

constexpr int64_t B = 2, E = 768, H = 12, S = 128;
// Note parameters scaled by kParamScale following weight initialization
// recommendations:
// https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2Config.initializer_range
// Note: Sdpa probability is set to 0. Since the dropout mask is sharded it
// throws off the seed offset between the sharded nvFuser program and the
// unsharded reference.
constexpr double kDropoutProb = 0.1, kParamScale = 0.02, kSdpaProb = 0.0,
                 kSdpaScale = 1e-3;

class DistributedTransformerTest
    : public MultiDeviceTest,
      public testing::WithParamInterface<DataType> {
 protected:
  DistributedTransformerTest() : D(communicator_->size()) {}

  void SetUp() {
    MultiDeviceTest::SetUp();
    if (H % D != 0) {
      GTEST_SKIP()
          << "Distributed transformer tests require number of devices evenly divide E ";
    }
    if (!deviceMajorMinorCheck(8)) {
      GTEST_SKIP() << "Distributed transformer tests require Ampere or newer";
    }
  }

  hir::HostIrExecutorParams executor_params_{
      .use_fusion_executor_cache = true,
      .skip_auto_scheduling = false,
      .cache_fusion_executor = false};

  const int64_t D; // number of devices
};

namespace {
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

std::vector<at::Tensor> reference_mlp(
    at::Tensor x,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::Tensor b1,
    at::ScalarType at_dtype) {
  auto linear0 = at::matmul(x, w0).add(b0).to(at::kFloat);
  auto gelu = at::gelu(linear0, "tanh");
  auto linear1 = at::matmul(gelu.to(at_dtype), w1).add(b1).to(at::kFloat);
  auto dropout = at::dropout(linear1, kDropoutProb, true);
  return {linear0, gelu, linear1, dropout};
}

std::vector<at::Tensor> reference_mha(
    at::Tensor x,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::Tensor b1,
    at::ScalarType at_dtype) {
  auto m = at::matmul(x, w0).add(b0).view({B, S, 3 * E});
  auto qkv_vec = m.split(E, 2);
  for (auto i = 0; i < 3; i++) {
    qkv_vec[i] =
        qkv_vec[i].reshape({B, S, H, E / H}).transpose(2, 1).to(at_dtype);
  }
  auto sdpa_out = at::_scaled_dot_product_flash_attention(
      qkv_vec[0], qkv_vec[1], qkv_vec[2], kSdpaProb, true, false, kSdpaScale);
  auto sdpa = std::get<0>(sdpa_out);
  // Reassemble heads (B, H, S, E/H) to (B, S, H, E/H) to (B, S, E)
  auto y = sdpa.transpose(1, 2).reshape({B * S, E});
  auto y_proj = at::matmul(y, w1).add(b1);
  auto y_dropout = at::dropout(y_proj.to(at::kFloat), kDropoutProb, true);
  return {m, sdpa, y_proj, y_dropout};
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

std::vector<at::Tensor> reference_mha_backwards(
    at::Tensor grad,
    at::Tensor x,
    at::Tensor mask,
    at::Tensor w0,
    at::Tensor b0,
    at::Tensor w1,
    at::ScalarType at_dtype) {
  // recompute up to sdpa
  auto m = at::matmul(x, w0).add(b0).view({B, S, 3 * E});
  auto qkv_vec = m.split(E, 2);
  for (auto i = 0; i < 3; i++) {
    qkv_vec[i] =
        qkv_vec[i].reshape({B, S, H, E / H}).transpose(2, 1).to(at_dtype);
  }
  auto q = qkv_vec[0];
  auto k = qkv_vec[1];
  auto v = qkv_vec[2];
  auto
      [sdpa_output,
       log_sumexp,
       cum_seq_q,
       cum_seq_k,
       query_seq_len,
       key_seq_len,
       philox_seed,
       philox_offset,
       debug_attn_mask] = at::_scaled_dot_product_flash_attention(
      q,
      k,
      v,
      /*dropout_p=*/kSdpaProb,
      /*is_causal=*/true,
      /*return_debug_mask=*/false,
      /*scale=*/kSdpaScale);

  // backwards pass
  auto dropout_grad =
      at::native_dropout_backward(grad, mask, 1.0 / (1.0 - kDropoutProb));
  auto dropout_grad_q = dropout_grad.to(at_dtype);
  auto matmul1_grad = at::matmul(dropout_grad_q, w1.transpose(0, 1));
  // sdpa output: B, H, S, E/H
  auto sdpa_output_reshape = sdpa_output.transpose(1, 2).view({B*S, E});
  auto matmul1_grad_w =
      at::matmul(dropout_grad_q.transpose(0, 1), sdpa_output_reshape)
          .transpose(0, 1);
  auto matmul1_grad_b = at::sum(dropout_grad, {0});

  // reshape matmul1 grad into sdpa output shape
  auto matmul1_grad_reshape = matmul1_grad.view({B, S, H, E/H}).transpose(1,2); // B, H, S, E/H
  auto [q_grad, k_grad, v_grad] =
      at::_scaled_dot_product_flash_attention_backward(
          matmul1_grad_reshape,
          q,
          k,
          v,
          sdpa_output,
          log_sumexp,
          cum_seq_q,
          cum_seq_k,
          /*max_q=*/*query_seq_len.maybe_as_int(),
          /*max_k=*/*key_seq_len.maybe_as_int(),
          /*dropout_p=*/kSdpaProb,
          /*is_causal=*/true,
          philox_seed,
          philox_offset,
          /*scale=*/kSdpaScale);
  auto q_grad_ = q_grad.transpose(1, 2).view({B*S, E});
  auto k_grad_ = k_grad.transpose(1, 2).view({B*S, E});
  auto v_grad_ = v_grad.transpose(1, 2).view({B*S, E});
  auto qkv_grad = at::cat({q_grad_,k_grad_,v_grad_}, -1);
  auto matmul0_grad_b = at::sum(qkv_grad.to(at::kFloat), {0});
  auto matmul0_grad = at::matmul(qkv_grad, w0.transpose(0, 1));
  auto matmul0_grad_w =
      at::matmul(qkv_grad.transpose(0, 1), x).transpose(0, 1);

  // Note: sdpa_output, sdpa_logsumexp are saved for the backwards pass
  // and become inputs to the nvfuser mha backwards pass
  std::vector<at::Tensor> tensors = {sdpa_output,
      log_sumexp,
      philox_seed, philox_offset,
      dropout_grad,
      matmul1_grad_w,
      matmul1_grad_b,
      q_grad, k_grad, v_grad,
      matmul0_grad_w,
      matmul0_grad_b,
      matmul0_grad
  };
  return tensors;
}

std::vector<TensorView*> mlp(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    const DeviceMesh& mesh,
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
  Val* prob = IrBuilder::create<Val>(1.0 - kDropoutProb);
  Val* scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  auto dropout_result = dropout(linear2, prob, scale).output;

  // Manual sharding annotations
  for (auto tv : {x, b1, matmul2, linear2, dropout_result}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w0, b0, w1, linear1, gelu}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  return {linear1, gelu, linear2, dropout_result};
}

std::vector<TensorView*> mha(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* b1,
    const DeviceMesh& mesh,
    DataType dtype) {
  // Linear 1
  TensorView* mm = matmul(x, w0);
  TensorView* proj_bias_bcast = broadcast(b0, {false, true, false});
  TensorView* qkv1 = add(mm, proj_bias_bcast);
  // Forming the q,k,v vectors:
  auto D = w0->axis(0)->extent()->value().as<int64_t>();
  TensorView* qkv = reshape(qkv1, {D, B * S, 3 * E / D}, {D, B, S, 3 * E / D});
  std::vector<TensorView*> qkv_reshaped = {};
  for (auto i : c10::irange(3)) {
    TensorView* tv_slice =
        slice(qkv, {0, 0, 0, E / D * i}, {D, B, S, E / D * (i + 1)});
    TensorView* tv_reshape =
        reshape(tv_slice, {D, B, S, E / D}, {D, B, S, H / D, E / H});
    TensorView* tv_trans = transpose(tv_reshape, 2, 3);
    TensorView* tv_cast = castOp(dtype, tv_trans);
    qkv_reshaped.push_back(tv_cast);
    // Explicitly shard qkv before calling SDPA node
    for (auto tv : {tv_slice, tv_reshape, tv_trans, tv_cast}) {
      tv->setDeviceMesh(mesh);
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }
  }
  // SDPA
  SdpfaFwdResult sdpa = sdpfa_fwd(
      qkv_reshaped[0],
      qkv_reshaped[1],
      qkv_reshaped[2],
      IrBuilder::create<Val>(kSdpaProb),
      IrBuilder::create<Val>(true),
      IrBuilder::create<Val>(kSdpaScale));
  TensorView* sdpa_output = sdpa.output;
  // Linear projection
  TensorView* sdpa_transpose = transpose(sdpa_output, 2, 3);
  TensorView* sdpa_reshape =
      reshape(sdpa_transpose, {D, B, S, H / D, E / H}, {D, B * S, E / D});
  TensorView* mm2 = matmul(sdpa_reshape, w1);
  TensorView* mm2_ar = sum(mm2, {0}); // allreduce
  TensorView* b1_bcast = broadcast(b1, {true, false});
  TensorView* linear2 = add(mm2_ar, b1_bcast);
  // Dropout
  Val* prob = IrBuilder::create<Val>(1.0 - kDropoutProb);
  Val* scale = IrBuilder::create<Val>(1.0 / (1.0 - kDropoutProb));
  auto dropout_result = dropout(linear2, prob, scale).output;

  for (auto tv : {x, b1, mm2_ar, linear2, dropout_result}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w0, b0, w1, mm2, qkv, sdpa_output}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  return {qkv, sdpa_output, linear2, dropout_result};
}

// TODO: These linear_backwards helper functions can be merged once 
// (1) improved sharding propagation pass and (2) insert resharding expr
// pass handles matmul tasks are complete.
// struct LinearBackwardsResult {
//   TensorView* grad;
//   TensorView* grad_w;
//   TensorView* grad_b;
// };

// x format: [i0, i1] dtype
// weight format: [DID(D), i1, i2/D] dtype
// grad format: [DID(D) i0, i2/D] float
// outputs: grad_x [i0, i1] dtype
// grad_w [DID i1/D, i2] dtype
// grad_b [i2] float
std::vector<TensorView*> linear_backwards(TensorView* x,
  TensorView* w,
  TensorView* grad, 
  DataType dtype) {
  TensorView* grad_q;
  if (grad->dtype() == dtype) {
    grad_q = grad;
    grad = castOp(DataType::Float, grad_q);
  } else {
    grad_q = castOp(dtype, grad);
  }
  TensorView* w_t = transpose(w, 1, 2);
  // Note: Depending how grad_x is sharded, an allreduce may
  // be automatically inserted.
  TensorView* grad_x_partials = matmul(grad_q, w_t);
  TensorView* grad_x = sum(grad_x_partials, {0}); // allreduce
  TensorView* grad_q_t = transpose(grad_q, 1, 2);
  TensorView* grad_w_t = matmul(grad_q_t, x);
  TensorView* grad_w = transpose(grad_w_t, 1, 2);
  TensorView* grad_b = sum(grad, {1});

  return {grad_x, grad_w, grad_b};
}

// x format: [i0, i1] dtype
// weight format: [DID, i1, i2/D] dtype
// grad format: [i0, i2] float
// outputs: grad_x [i0, i1] dtype
// grad_w [DID i1, i2] dtype
// grad_b [i2] float
std::vector<TensorView*> sharded_linear_backwards(TensorView* x,
  TensorView* w,
  TensorView* grad, 
  DataType dtype) {
  TensorView* grad_q = castOp(dtype, grad);
  TensorView* w_t = transpose(w, 1, 2);
  TensorView* grad_x = matmul(grad_q, w_t);
  TensorView* grad_t = transpose(grad_q, 0, 1);
  TensorView* grad_w_t = matmul(grad_t, x);
  TensorView* grad_w = transpose(grad_w_t, 1, 2);
  TensorView* grad_b = sum(grad, {0});

  return {grad_x, grad_w, grad_b};
}

std::vector<TensorView*> mlp_backwards(
    TensorView* grad,
    TensorView* x,
    TensorView* mask,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    const DeviceMesh& mesh,
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

  std::vector<TensorView*> linear1_grads = sharded_linear_backwards(gelu, w1, dropout_grad, dtype);

  TensorView* matmul1_grad_x_ = castOp(DataType::Float, linear1_grads[0]);
  TensorView* gelu_grad = tanh_gelu_backward(matmul1_grad_x_, linear0);
  
  std::vector<TensorView*> linear0_grads = linear_backwards(x, w0, gelu_grad, dtype);

  // Manaul sharding annotations
  for (auto tv :
       {x,
        grad,
        mask,
        dropout_grad,
        linear1_grads[2],
        linear0_grads[0]}) {
    tv->setDeviceMesh(mesh);
  }

  for (auto tv :
       {w0,
        b0,
        w1,
        linear1_grads[0],
        linear1_grads[1],
        gelu_grad,
        linear0_grads[1],
        linear0_grads[2]}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  std::vector<TensorView*> outputs = {
      dropout_grad,
      linear1_grads[1],
      linear1_grads[2],
      gelu_grad,
      linear0_grads[1],
      linear0_grads[2],
      linear0_grads[0]};
  return outputs;
}

std::vector<TensorView*> mha_backwards(
    TensorView* x,
    TensorView* w0,
    TensorView* b0,
    TensorView* w1,
    TensorView* mask, 
    TensorView* sdpa_output, 
    TensorView* sdpa_log_sumexp,
    TensorView* sdpa_seed,
    TensorView* sdpa_offset,
    TensorView* grad,
    const DeviceMesh& mesh,
    DataType dtype) {
  // Recompute: Linear 1, QKV vectors
  TensorView* mm = matmul(x, w0);
  TensorView* proj_bias_bcast = broadcast(b0, {false, true, false});
  TensorView* qkv1 = add(mm, proj_bias_bcast);
  // Forming the q,k,v vectors:
  auto D = w0->axis(0)->extent()->value().as<int64_t>();
  TensorView* qkv = reshape(qkv1, {D, B * S, 3 * E / D}, {D, B, S, 3 * E / D});
  std::vector<TensorView*> qkv_reshaped = {};
  for (auto i : c10::irange(3)) {
    TensorView* tv_slice =
        slice(qkv, {0, 0, 0, E / D * i}, {D, B, S, E / D * (i + 1)});
    TensorView* tv_reshape =
        reshape(tv_slice, {D, B, S, E / D}, {D, B, S, H / D, E / H});
    TensorView* tv_trans = transpose(tv_reshape, 2, 3);
    TensorView* tv_cast = castOp(dtype, tv_trans);
    qkv_reshaped.push_back(tv_cast);
    // Explicitly shard qkv before calling SDPA node
    for (auto tv : {tv_slice, tv_reshape, tv_trans, tv_cast}) {
      tv->setDeviceMesh(mesh);
      tv->axis(0)->parallelize(ParallelType::DIDx);
    }
  }

  // Backwards
  constexpr double kScale = 1.0 / (1.0 - kDropoutProb);
  Val* dscale = IrBuilder::create<Val>(kScale);
  TensorView* dropout_grad = dropout_backward(grad, mask, dscale);

  // linear1 backwards
  TensorView* sdpa_output_reshape = transpose(sdpa_output, 2, 3); // D, B, S, H/D, E/H
  sdpa_output_reshape = reshape(sdpa_output_reshape, {D, B, S, H/D, E/H}, {D, B*S, E/D});
  std::vector<TensorView*> linear1_backwards = sharded_linear_backwards(sdpa_output_reshape, w1, dropout_grad, dtype);

  // SDPA backwards
  TensorView* linear1_grad_x = reshape(linear1_backwards[0], {D, B*S, E/D}, {D, B, S, H/D, E/H});
  linear1_grad_x = transpose(linear1_grad_x, 2, 3); // D, B, H/D, S, E/H
  // Explicitly shard inputs into SDPA backward node
  for (auto tv : {linear1_grad_x, sdpa_output, sdpa_log_sumexp}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  auto sdpa_grad = sdpfa_bwd(
      linear1_grad_x,
      qkv_reshaped[0],
      qkv_reshaped[1],
      qkv_reshaped[2],
      sdpa_output,
      sdpa_log_sumexp,
      /*dropout_p=*/IrBuilder::create<Val>(kSdpaProb),
      /*is_causal=*/IrBuilder::create<Val>(true),
      sdpa_seed,
      sdpa_offset,
      /*scale=*/IrBuilder::create<Val>(kSdpaScale));

  TensorView* q_grad = transpose(sdpa_grad.grad_query, 2, 3);
  q_grad = reshape(q_grad, {D, B, S, H/D, E/H}, {D, B*S, E/D});
  TensorView* v_grad = transpose(sdpa_grad.grad_value, 2, 3);
  v_grad = reshape(v_grad, {D, B, S, H/D, E/H}, {D, B*S, E/D});
  TensorView* k_grad = transpose(sdpa_grad.grad_key, 2, 3);
  k_grad = reshape(k_grad, {D, B, S, H/D, E/H}, {D, B*S, E/D});
  TensorView* kqv_grad = cat({k_grad, q_grad, v_grad}, -1);
  std::vector<TensorView*> linear0_backwards = linear_backwards(x, w0, kqv_grad, dtype);

  for (auto tv : {x, mask, grad, dropout_grad, linear1_backwards[2], linear0_backwards[0]}) {
    tv->setDeviceMesh(mesh);
  }
  for (auto tv : {w0, b0, w1, sdpa_output, mm,
    linear1_backwards[0], linear1_backwards[1], 
    linear0_backwards[1], linear0_backwards[2],
    sdpa_grad.grad_query, sdpa_grad.grad_key, sdpa_grad.grad_value}) {
    tv->setDeviceMesh(mesh);
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }
  return {dropout_grad, linear1_backwards[1], linear1_backwards[2], 
     sdpa_grad.grad_query, sdpa_grad.grad_key, sdpa_grad.grad_value,
     linear0_backwards[1], linear0_backwards[2], linear0_backwards[0]};
}
} // namespace

TEST_P(DistributedTransformerTest, MLP_Layer) {
  preseg_passes::OptimizationPassGuard<preseg_passes::PropagateShardingsPass>
      guard(false);
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

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
      mlp(tvx, tvw0, tvb0, tvw1, tvb1, mesh, dtype);

  for (TensorView* tv : tvsout) {
    fusion->addOutput(tv);
  }
  shardBetween({tvw0}, {tvsout[3]});

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x = at::randn({B * S, E}, options);
  auto w0 = at::randn({E, 4 * E}, options) * kParamScale;
  auto b0 = at::randn({4 * E}, options) * kParamScale;
  auto w1 = at::randn({4 * E, E}, options) * kParamScale;
  auto b1 = at::randn({E}, options) * kParamScale;

  // Note: resetting the seed before reference and nvFuser
  // execution so that random vals are the same.
  at::manual_seed(getATenRandomSeed());
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

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  at::manual_seed(getATenRandomSeed());
  auto outputs = runtime.runWithInput(inputs);
  validate(expected_outputs, outputs);
}

TEST_P(DistributedTransformerTest, Multiheaded_Attention) {
  preseg_passes::OptimizationPassGuard<preseg_passes::PropagateShardingsPass>
      guard(false);
  auto dtype = GetParam();
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);
  at::ScalarType at_dtype = data_type_to_aten(dtype);

  TensorView* tvx = makeContigConcreteTensor({B * S, E}, dtype);
  TensorView* tvw0 = makeContigConcreteTensor({D, E, 3 * E / D}, dtype);
  TensorView* tvb0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* tvw1 = makeContigConcreteTensor({D, E / D, E}, dtype);
  TensorView* tvb1 = makeContigConcreteTensor({E}, dtype);

  fusion->addInput(tvx);
  fusion->addInput(tvw0);
  fusion->addInput(tvb0);
  fusion->addInput(tvw1);
  fusion->addInput(tvb1);

  auto tv_outs = mha(tvx, tvw0, tvb0, tvw1, tvb1, mesh, dtype);

  for (auto tv : tv_outs) {
    fusion->addOutput(tv);
  }

  shardBetween({tvw0}, {tv_outs[3]});

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x = at::randn({B * S, E}, options);
  auto w0 = at::randn({E, 3 * E}, options) * kParamScale;
  auto b0 = at::randn({3 * E}, options) * kParamScale;
  auto w1 = at::randn({E, E}, options) * kParamScale;
  auto b1 = at::randn({E}, options) * kParamScale;

  at::manual_seed(getATenRandomSeed());
  auto reference_outs = reference_mha(x, w0, b0, w1, b1, at_dtype);
  std::vector<c10::IValue> inputs = {
      x,
      shardTensor(w0.view({E, 3, E}), 2, mesh).view({1, E, 3 * E / D}),
      shardTensor(b0.view({3, E}), 1, mesh).view({1, 3 * E / D}),
      shardTensor(w1, 0, mesh),
      b1};
  std::vector<at::Tensor> expected_outputs = {
      shardTensor(reference_outs[0].view({B, S, 3, E}), 3, mesh)
          .view({1, B, S, 3 * E / D}),
      shardTensor(reference_outs[1], 1, mesh),
      reference_outs[2],
      reference_outs[3]};

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  at::manual_seed(getATenRandomSeed());
  auto out = runtime.runWithInput(inputs);
  validate(expected_outputs, out);
}

TEST_P(DistributedTransformerTest, MLP_Backward) {
  preseg_passes::OptimizationPassGuard<preseg_passes::PropagateShardingsPass>
      guard(false);
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

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
      mlp_backwards(grad, x, mask, w0, b0, w1, mesh, dtype);

  for (TensorView* tv : tv_outs) {
    fusion->addOutput(tv);
  }

  // Sharded: matmul1_grad_w, gelu_grad, matmul0_grad_w, matmul0_grad_b
  shardBetween({w0, w1, b0}, {tv_outs[1], tv_outs[3], tv_outs[4], tv_outs[5]});
  // Unsharded: dropout_grad, matmul1_grad_b, matmul0_grad_x
  shardBetween({x, mask, grad}, {tv_outs[0], tv_outs[2], tv_outs[6]});

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

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  auto outputs = runtime.runWithInput(inputs);

  validate(expected_outputs, outputs);
}

TEST_P(DistributedTransformerTest, MHA_Backward) {
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* tvx = makeContigConcreteTensor({B * S, E}, dtype);
  TensorView* tvw0 = makeContigConcreteTensor({D, E, 3 * E / D}, dtype);
  TensorView* tvb0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* tvw1 = makeContigConcreteTensor({D, E / D, E}, dtype);
  TensorView* tvgrad = makeContigConcreteTensor({B * S, E}, DataType::Float);
  TensorView* tvmask = makeContigConcreteTensor({B * S, E}, DataType::Bool);
  TensorView* tvsdpa_out = makeContigConcreteTensor({D, B, H/D, S, E/H}, dtype);
  TensorView* tvsdpa_log_sumexp = makeContigConcreteTensor({D, B, H/D, S}, DataType::Float);
  TensorView* tvsdpa_seed = makeSymbolicTensor({}, DataType::Int);
  TensorView* tvspda_offset = makeSymbolicTensor({}, DataType::Int);

  fusion->addInput(tvx);
  fusion->addInput(tvw0);
  fusion->addInput(tvb0);
  fusion->addInput(tvw1);
  fusion->addInput(tvgrad);
  fusion->addInput(tvmask);
  fusion->addInput(tvsdpa_out);
  fusion->addInput(tvsdpa_log_sumexp);
  fusion->addInput(tvsdpa_seed);
  fusion->addInput(tvspda_offset);

  auto tvouts = mha_backwards(tvx, tvw0, tvb0, tvw1, tvmask, tvsdpa_out, 
    tvsdpa_log_sumexp, tvsdpa_seed, tvspda_offset, tvgrad, mesh, dtype);

  for (auto tv : tvouts) {
    fusion->addOutput(tv);
  }

  shardBetween({tvw1}, {tvouts[1], tvouts[2]});
  shardBetween({tvw0, tvb0}, {tvouts[3], tvouts[4], tvouts[5], tvouts[6], tvouts[7]});
  shardBetween({tvx, tvmask, tvgrad}, {tvouts[0], tvouts[8]});

  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x = at::randn({B * S, E}, options);
  auto w0 = at::randn({E, 3 * E}, options) * kParamScale;
  auto b0 = at::randn({3 * E}, options) * kParamScale;
  auto w1 = at::randn({E, E}, options) * kParamScale;
  auto grad = at::randn({B * S, E}, options).to(at::kFloat);
  auto mask = at::randn({B * S, E}, options).lt(1.0 - kDropoutProb);

  at::manual_seed(getATenRandomSeed());
  auto reference_outs = reference_mha_backwards(grad, x, mask, w0, b0, w1, at_dtype);
  std::vector<c10::IValue> inputs = {
      x,
      shardTensor(w0.view({E, 3, E}), 2, mesh).view({1, E, 3 * E / D}),
      shardTensor(b0.view({3, E}), 1, mesh).view({1, 3 * E / D}),
      shardTensor(w1, 0, mesh),
      grad,
      mask,
      shardTensor(reference_outs[0], 1, mesh), // sdpa.output
      shardTensor(reference_outs[1], 1, mesh), // sdpa.log_sumexp
      reference_outs[2],
      reference_outs[3]
      };
  std::vector<at::Tensor> expected_outputs = {
      reference_outs[4], // dropout grad
      shardTensor(reference_outs[5], 0, mesh), // matmul1 weight grad
      reference_outs[6], // matmul1 bias grad
      shardTensor(reference_outs[7], 1, mesh), // q grad
      shardTensor(reference_outs[8], 1, mesh), // k grad
      shardTensor(reference_outs[9], 1, mesh),  // v grad
      shardTensor(reference_outs[10].view({E, 3, E}), 2, mesh).view({1, E, 3 * E / D}),
      shardTensor(reference_outs[11].view({3, E}), 1 , mesh).view({1, 3 * E / D}),
      reference_outs[12]
      };

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  at::manual_seed(getATenRandomSeed());
  runtime.runWithInput(inputs);
  auto out = runtime.runWithInput(inputs);
  validate(expected_outputs, out);
}

TEST_P(DistributedTransformerTest, Forward) {
  preseg_passes::OptimizationPassGuard<preseg_passes::PropagateShardingsPass>
      guard(false);
  auto dtype = GetParam();
  at::ScalarType at_dtype = data_type_to_aten(dtype);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  const auto mesh = DeviceMesh::createForNumDevices(D);

  TensorView* x = makeContigConcreteTensor({B * S, E}, DataType::Float);
  TensorView* mha_w0 = makeContigConcreteTensor({D, E, 3 * E / D}, dtype);
  TensorView* mha_b0 = makeContigConcreteTensor({D, 3 * E / D}, dtype);
  TensorView* mha_w1 = makeContigConcreteTensor({D, E / D, E}, dtype);
  TensorView* mha_b1 = makeContigConcreteTensor({E}, dtype);
  TensorView* mlp_w0 = makeContigTensor(3, dtype);
  TensorView* mlp_b0 = makeContigTensor(2, dtype);
  TensorView* mlp_w1 = makeContigTensor(3, dtype);
  TensorView* mlp_b1 = makeContigTensor(1, dtype);

  fusion->addInput(x);
  fusion->addInput(mha_w0);
  fusion->addInput(mha_b0);
  fusion->addInput(mha_w1);
  fusion->addInput(mha_b1);
  fusion->addInput(mlp_w0);
  fusion->addInput(mlp_b0);
  fusion->addInput(mlp_w1);
  fusion->addInput(mlp_b1);

  constexpr float kEps = 1e-5;
  auto eps = IrBuilder::create<Val>(kEps);
  std::vector<int64_t> norm_shape{E};

  auto ln_1 =
      layer_norm(x, norm_shape, /*weight=*/nullptr, /*bias=*/nullptr, eps);
  auto mha_in = castOp(dtype, ln_1.output);
  auto mha_out = mha(mha_in, mha_w0, mha_b0, mha_w1, mha_b1, mesh, dtype)[3];
  auto resid_1 = add(x, mha_out);
  auto ln_2 = layer_norm(
      resid_1, norm_shape, /*weight=*/nullptr, /*bias=*/nullptr, eps);
  auto mlp_in = castOp(dtype, ln_2.output);
  auto mlp_out = mlp(mlp_in, mlp_w0, mlp_b0, mlp_w1, mlp_b1, mesh, dtype)[3];
  auto resid_2 = add(mha_out, mlp_out);

  fusion->addOutput(ln_1.output);
  fusion->addOutput(mha_out);
  fusion->addOutput(ln_2.output);
  fusion->addOutput(mlp_out);
  fusion->addOutput(resid_2);

  for (auto tv : {x, ln_1.output, ln_2.output, resid_2}) {
    tv->setDeviceMesh(mesh);
  }

  shardBetween({mlp_w0}, {resid_2});
  shardBetween({mha_w0}, {mlp_in});
  shardBetween({x}, {mha_in});


  const auto options =
      at::TensorOptions().dtype(at_dtype).device(communicator_->device());
  auto x_ = at::randn({B * S, E}, options).to(at::kFloat);
  auto mha_w0_ = at::randn({E, 3 * E}, options) * kParamScale;
  auto mha_b0_ = at::randn({3 * E}, options) * kParamScale;
  auto mha_w1_ = at::randn({E, E}, options) * kParamScale;
  auto mha_b1_ = at::randn({E}, options) * kParamScale;

  auto mlp_w0_ = at::randn({E, 4 * E}, options) * kParamScale;
  auto mlp_b0_ = at::randn({4 * E}, options) * kParamScale;
  auto mlp_w1_ = at::randn({4 * E, E}, options) * kParamScale;
  auto mlp_b1_ = at::randn({E}, options) * kParamScale;

  at::manual_seed(getATenRandomSeed());
  auto ln_1_ = at::native_layer_norm(
      x_, norm_shape, /*weight=*/std::nullopt, /*bias=*/std::nullopt, kEps);
  auto ln_1_out_ = std::get<0>(ln_1_).to(at_dtype);

  auto mha_out_ =
      reference_mha(ln_1_out_, mha_w0_, mha_b0_, mha_w1_, mha_b1_, at_dtype)[3];
  auto resid1_ = mha_out_ + x_;
  auto ln_2_ = at::native_layer_norm(
      resid1_,
      norm_shape,
      /*weight=*/std::nullopt,
      /*bias=*/std::nullopt,
      kEps);
  auto ln_2_out_ = std::get<0>(ln_2_).to(at_dtype);

  auto mlp_out_ =
      reference_mlp(ln_2_out_, mlp_w0_, mlp_b0_, mlp_w1_, mlp_b1_, at_dtype)[3];
  auto at_out = mha_out_ + mlp_out_;

  std::vector<c10::IValue> inputs = {
      x_,
      shardTensor(mha_w0_.view({E, 3, E}), 2, mesh).view({1, E, 3 * E / D}),
      shardTensor(mha_b0_.view({3, E}), 1, mesh).view({1, 3 * E / D}),
      shardTensor(mha_w1_, 0, mesh),
      mha_b1_,
      shardTensor(mlp_w0_, 1, mesh),
      shardTensor(mlp_b0_, 0, mesh),
      shardTensor(mlp_w1_, 0, mesh),
      mlp_b1_};

  std::vector<at::Tensor> expected_outputs = {
      ln_1_out_, mha_out_, ln_2_out_, mlp_out_, at_out};

  MultiDeviceExecutor runtime(
      std::move(fusion), *communicator_, executor_params_);
  at::manual_seed(getATenRandomSeed());
  auto outputs = runtime.runWithInput(inputs);
  validate(expected_outputs, outputs);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    DistributedTransformerTest,
    testing::Values(DataType::Half, DataType::BFloat16));
} // namespace nvfuser
