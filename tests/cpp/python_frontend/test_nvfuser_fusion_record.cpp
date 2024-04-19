// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <python_frontend/fusion_record.h>
#include <tests/cpp/utils.h>
#include <tests/cpp/validator.h>

namespace nvfuser {
using namespace nvfuser::python_frontend;

// RUN CMD: bin/test_jit --gtest_filter="NVFuserTest*RecordFunctorEquality*"
TEST_F(NVFuserTest, RecordFunctorEquality_CUDA) {
  // Getting the std::function matching correct is error prone so providing
  // checks for OpRecord, CastOp, and ReductionOp that employ std::function
  // matching.

  // OpRecord Equality Check
  {
    auto t0 = State(0, serde::StateType::Tensor);
    auto s1 = State(1, serde::StateType::Scalar);
    auto out = State(2, serde::StateType::Tensor);
    std::unique_ptr<RecordFunctor> test_record1(
        new OpRecord<TensorView*, TensorView*, Val*>(
            {t0, s1},
            {out},
            "ops.mul",
            serde::RecordType::Binary_TV_VAL,
            static_cast<TensorView* (*)(TensorView*, Val*)>(mul)));
    std::unique_ptr<RecordFunctor> test_record2(
        new OpRecord<TensorView*, TensorView*, Val*>(
            {t0, s1},
            {out},
            "ops.mul",
            serde::RecordType::Binary_TV_VAL,
            static_cast<TensorView* (*)(TensorView*, Val*)>(mul)));
    std::unique_ptr<RecordFunctor> test_record3(
        new OpRecord<TensorView*, TensorView*, Val*>(
            {t0, s1},
            {out},
            "ops.mul",
            serde::RecordType::Binary_TV_VAL,
            static_cast<TensorView* (*)(TensorView*, Val*)>(mul)));

    EXPECT_TRUE(*test_record1 == *test_record2);
    EXPECT_TRUE(*test_record1 == *test_record3);
    EXPECT_TRUE(*test_record2 == *test_record3);
  }

  // CastOpRecord Equality Check
  {
    auto t0 = State(0, serde::StateType::Tensor);
    auto out = State(1, serde::StateType::Tensor);
    std::unique_ptr<RecordFunctor> test_record1(
        new CastOpRecord<TensorView*, TensorView*>(
            {t0},
            {out},
            "ops.cast",
            serde::RecordType::CastTv,
            static_cast<TensorView* (*)(DataType, TensorView*)>(castOp),
            DataType::Half));
    std::unique_ptr<RecordFunctor> test_record2(
        new CastOpRecord<TensorView*, TensorView*>(
            {t0},
            {out},
            "ops.cast",
            serde::RecordType::CastTv,
            static_cast<TensorView* (*)(DataType, TensorView*)>(castOp),
            DataType::Half));
    std::unique_ptr<RecordFunctor> test_record3(
        new CastOpRecord<TensorView*, TensorView*>(
            {t0},
            {out},
            "ops.cast",
            serde::RecordType::CastTv,
            static_cast<TensorView* (*)(DataType, TensorView*)>(castOp),
            DataType::Half));

    EXPECT_TRUE(*test_record1 == *test_record2);
    EXPECT_TRUE(*test_record1 == *test_record3);
    EXPECT_TRUE(*test_record2 == *test_record3);
  }

  // ReductionOpRecord Equality Check
  {
    auto t0 = State(0, serde::StateType::Tensor);
    auto out = State(1, serde::StateType::Tensor);
    std::unique_ptr<RecordFunctor> test_record1(new ReductionOpRecord(
        {t0},
        {out},
        "ops.sum",
        serde::RecordType::ReductionSum,
        static_cast<TensorView* (*)(TensorView*,
                                    const std::vector<int64_t>&,
                                    bool,
                                    DataType)>(sum),
        {0},
        false,
        DataType::Float));
    std::unique_ptr<RecordFunctor> test_record2(new ReductionOpRecord(
        {t0},
        {out},
        "ops.sum",
        serde::RecordType::ReductionSum,
        static_cast<TensorView* (*)(TensorView*,
                                    const std::vector<int64_t>&,
                                    bool,
                                    DataType)>(sum),
        {0},
        false,
        DataType::Float));
    std::unique_ptr<RecordFunctor> test_record3(new ReductionOpRecord(
        {t0},
        {out},
        "ops.sum",
        serde::RecordType::ReductionSum,
        static_cast<TensorView* (*)(TensorView*,
                                    const std::vector<int64_t>&,
                                    bool,
                                    DataType)>(sum),
        {0},
        false,
        DataType::Float));

    EXPECT_TRUE(*test_record1 == *test_record2);
    EXPECT_TRUE(*test_record1 == *test_record3);
    EXPECT_TRUE(*test_record2 == *test_record3);
  }
}

} // namespace nvfuser
