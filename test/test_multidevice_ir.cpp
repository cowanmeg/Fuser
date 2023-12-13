// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <gtest/gtest.h>
#include <disjoint_set.h>
#include <test/multidevice.h>
#include <test/validator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ops/all_ops.h>
#include <multidevice/pipeline_ir.h>
#include <multidevice/runtime.h>
#include <multidevice/utils.h>

namespace nvfuser {

TEST_F(MultiDeviceTest, ShardOuterAxis) {
  int sharded_dim = 0;
  Fusion fusion;
  FusionGuard fg(&fusion);
  DeviceMesh mesh({0, 1});
  int num_devices = 2;

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv0_ = add(tv0, tv0);
  TensorView* tv1 = set(tv0_);
  TensorView* tv2 = add(tv1, tv1);
  TensorView* tv3 = set(tv2);
  TensorView* tv4 = sum(tv3, {0});
  fusion.addInput(tv0);
  fusion.addOutput(tv4);

  tv1->split(sharded_dim, num_devices, false);
  tv1->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  tv2->split(sharded_dim, num_devices, false);
  tv2->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  
  PipelineStageDescriptor stage0(false), stage1(false), stage2(false);
  stage0.addVal({tv0, tv0_});
  stage1.addVal({tv1, tv2});
  stage2.addVal({tv3, tv4});
  stage0.mesh = mesh;
  stage1.mesh = mesh;
  stage2.mesh = mesh;

  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1), std::move(stage2)}};
  Pipeline pipeline(&fusion, std::move(descriptor));

  fusion.printKernel();

  auto x = at::randn({num_devices*2, 3}, tensor_options);
  std::vector<c10::IValue> inputs = {x};
  auto ref_outputs = at::sum(x*4, {0});
  // if (communicator->deviceId() == 0) {
  //   std::cout << "Inputs " << x << std::endl;
  //   std::cout << "Expected " << ref_outputs << std::endl;
  // }
  
  MultiDeviceRuntime runtime(&pipeline, *communicator);
  auto outputs = runtime.runWithInput(inputs);
  // std::cout << "Outputs:" << std::endl << outputs << std::endl;

  testValidate(&fusion, outputs, inputs, {ref_outputs}, __LINE__, __FILE__);

  }

TEST_F(MultiDeviceTest, ShardInnerAxis) {
  int sharded_dim = 1;
  Fusion fusion;
  FusionGuard fg(&fusion);
  DeviceMesh mesh({0, 1});
  int num_devices = 2;

  TensorView* tv0 = makeContigTensor(2);
  TensorView* tv0_ = add(tv0, tv0);
  TensorView* tv1 = set(tv0_);
  TensorView* tv2 = add(tv1, tv1);
  TensorView* tv3 = set(tv2);
  TensorView* tv4 = sum(tv3, {0});
  fusion.addInput(tv0);
  fusion.addOutput(tv4);

  // TODO: Split on non-outermost axis
  // tv1->split(sharded_dim, num_devices, false);
  tv1->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  // tv2->split(sharded_dim, num_devices, false);
  tv2->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  
  PipelineStageDescriptor stage0(false), stage1(false), stage2(false);
  stage0.addVal({tv0, tv0_});
  stage1.addVal({tv1, tv2});
  stage2.addVal({tv3, tv4});
  stage0.mesh = mesh;
  stage1.mesh = mesh;
  stage2.mesh = mesh;


  PipelineDescriptor descriptor{
      .stage_descriptors{std::move(stage0), std::move(stage1), std::move(stage2)}};
  Pipeline pipeline(&fusion, std::move(descriptor));

  fusion.printKernel();

  auto x = at::randn({3, num_devices}, tensor_options);
  std::vector<c10::IValue> inputs = {x};
  auto ref_outputs = at::sum(x*4, {0});
  if (communicator->deviceId() == 0) {
    std::cout << "Inputs " << x << std::endl;
    std::cout << "Expected " << ref_outputs << std::endl;
  }
  
  MultiDeviceRuntime runtime(&pipeline, *communicator);
  auto outputs = runtime.runWithInput(inputs);
  std::cout << "Outputs " << outputs << std::endl;

  testValidate(&fusion, outputs, inputs, {ref_outputs}, __LINE__, __FILE__);
}

// TEST_F(MultiDeviceTest, ShardReorder) {
//   Fusion fusion;
//   FusionGuard fg(&fusion);
//   DeviceMesh mesh({0, 1});
//   int num_devices = 2;

//   TensorView* tv0 = makeContigTensor(2);
//   TensorView* tv0_ = add(tv0, tv0);
//   TensorView* tv1 = set(tv0_);
//   TensorView* tv2 = add(tv1, tv1);
//   TensorView* tv3 = set(tv2);
//   TensorView* tv4 = sum(tv3, {0});
//   fusion.addInput(tv0);
//   fusion.addOutput(tv4);

//   // TODO: enforce the outermost axis are the only ones allowed to be parallelized. 
//   // TODO: split.
//   // tv1->split(0, num_devices, false);
//   tv1->setAllocationDomain({tv1->axis(1), tv1->axis(0)}, true);
//   tv1->reorder({{1, 0}});
//   tv1->axis(0)->parallelize(ParallelType::DIDx);
//   std::cout << "TV1 " << tv1->toString();
//   // tv2->split(0, num_devices, false);
//   // tv2->setAllocationDomain({tv2->axis(1), tv2->axis(0)}, true); // This is implied?
//   // tv2->reorder({{1, 0}});
//   tv2->axis(0)->parallelize(ParallelType::DIDx);
//   std::cout << "TV2 " << tv2->toString();
  
//   PipelineStageDescriptor stage0(false), stage1(false), stage2(false);
//   stage0.addVal({tv0, tv0_});
//   stage1.addVal({tv1, tv2});
//   stage2.addVal({tv3, tv4});
//   stage0.mesh = mesh;
//   stage1.mesh = mesh;
//   stage2.mesh = mesh;


//   PipelineDescriptor descriptor{
//       .stage_descriptors{std::move(stage0), std::move(stage1), std::move(stage2)}};
//   Pipeline pipeline(&fusion, std::move(descriptor));

//   fusion.printKernel();

//   auto x = at::randn({4, num_devices}, tensor_options);
//   std::vector<c10::IValue> inputs = {x};
//   auto ref_outputs = at::sum(x*4, {0});
//   if (communicator->deviceId() == 0) {
//     std::cout << "Inputs " << x << std::endl;
//     std::cout << "Expected " << ref_outputs << std::endl;
//   }
  
//   MultiDeviceRuntime runtime(&pipeline, *communicator);
//   auto outputs = runtime.runWithInput(inputs);
//   std::cout << "Outputs " << outputs << std::endl;

//   testValidate(&fusion, outputs, inputs, {ref_outputs}, __LINE__, __FILE__);
// }

} //nvfuser
#endif
