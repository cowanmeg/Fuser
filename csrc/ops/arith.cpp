// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <expr_evaluator.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/iostream.h>
#include <ir/utils.h>
#include <ops/alias.h>
#include <ops/arith.h>
#include <ops/utils.h>
#include <type.h>
#include <type_promotion.h>

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Half.h>
#include <c10/util/irange.h>

#include <cfloat>

namespace nvfuser {

Val* castOp(DataType dtype, Val* v1) {
  auto orig_dtype = v1->getDataType().value();
  if (dtype == orig_dtype) {
    return set(v1);
  }

  if (cast_func_str(std::make_pair(orig_dtype, dtype)) == std::nullopt) {
    NVF_CHECK(
        false,
        "Illegal Cast value from  DataType: ",
        orig_dtype,
        " to DataType: ",
        dtype);
  }

  if (isComplexType(orig_dtype) && !isComplexType(dtype)) {
    TORCH_WARN(
        "Casting from ",
        orig_dtype,
        " to ",
        dtype,
        " discards the imaginary part.");
  }

  Val* out = ops::newValLike(v1, dtype);
  IrBuilder::create<UnaryOp>(UnaryOpType::Cast, out, v1);
  return out;
}

Val* maybeCastOp(DataType dtype, Val* v1) {
  if (v1->isScalar()) {
    return SimplifyingIrBuilder::maybeCastExpr(dtype, v1);
  }
  if (v1->dtype() != dtype) {
    return castOp(dtype, v1);
  }
  return v1;
}

TensorView* castOp(DataType dtype, TensorView* v1) {
  return castOp(dtype, v1->as<Val>())->as<TensorView>();
}

TensorView* maybeCastOp(DataType dtype, TensorView* v1) {
  if (v1->dtype() != dtype) {
    return castOp(dtype, v1);
  }
  return v1;
}

Val* bitCastOp(DataType dtype, Val* v1) {
  if (v1->getDataType().value() == dtype) {
    return v1;
  }

  NVF_CHECK(
      dataTypeSize(v1->getDataType().value()) == dataTypeSize(dtype),
      "BitCast only works for types of the same size");

  Val* out = ops::newValLike(v1, dtype);
  IrBuilder::create<UnaryOp>(UnaryOpType::BitCast, out, v1);
  return out;
}

TensorView* bitCastOp(DataType dtype, TensorView* v1) {
  return bitCastOp(dtype, v1->as<Val>())->as<TensorView>();
}

Val* unaryOp(UnaryOpType type, Val* v1) {
  Val* out = ops::newValLike(v1, v1->getDataType().value());
  IrBuilder::create<UnaryOp>(type, out, v1);
  return out;
}

TensorView* unaryOp(UnaryOpType type, TensorView* v1) {
  return unaryOp(type, v1->as<Val>())->as<TensorView>();
}

Val* unaryIsOp(UnaryOpType type, Val* v) {
  Val* out = ops::newValLike(v, DataType::Bool);
  IrBuilder::create<UnaryOp>(type, out, v);
  return out;
}

TensorView* unaryIsOp(UnaryOpType type, TensorView* v) {
  return unaryIsOp(type, v->asVal())->as<TensorView>();
}

Val* unaryOp(UnaryOpType type, Val* v1, const TypePromotionConfig& config) {
  auto cast_v1 = promoteValues(config, {v1}).front();
  return unaryOp(type, cast_v1);
}

TensorView* unaryOp(
    UnaryOpType type,
    TensorView* v1,
    const TypePromotionConfig& config) {
  auto cast_v1 = promoteValues(config, {v1}).front();
  return unaryOp(type, cast_v1)->as<TensorView>();
}

static TensorView* factoryOutput(
    const std::vector<Val*>& shape,
    DataType dtype,
    bool maybe_symbolic = true) {
  // For concrete dimensions, set IterType to Broadcast or Iteration. If we
  // cannot determine the IterType, set it to Symbolic so that it can be set
  // later during dynamic shape concretization.
  std::vector<IterDomain*> out_root;
  out_root.reserve(shape.size());
  ExpressionEvaluator ee;
  for (Val* shi : shape) {
    IterType iter_type =
        maybe_symbolic ? IterType::Symbolic : IterType::Iteration;
    PolymorphicValue ext = ee.evaluate(shi);
    if (ext.hasValue()) {
      NVF_CHECK(
          ext.is<int64_t>(),
          "Expected int extent argument to factory function but found constant value ",
          shi->toInlineString());
      iter_type =
          ext.as<int64_t>() == 1 ? IterType::Broadcast : IterType::Iteration;
    }
    out_root.push_back(
        IterDomainBuilder(
            shi->fusion()->zeroVal(),
            SimplifyingIrBuilder::maybeCastExpr(DataType::Index, shi))
            .iter_type(iter_type)
            .build());
  }
  auto* out_td = IrBuilder::create<TensorDomain>(
      out_root, TensorDomain::getContiguityFilledWith(out_root, true));
  auto* out = IrBuilder::create<TensorView>(out_td, dtype);
  return out;
}

// TENSOR FACTORIES
TensorView* rand(
    const std::vector<Val*>& shape,
    DataType dtype,
    Val* philox_seed,
    Val* philox_offset,
    bool maybe_symbolic) {
  TensorView* out = factoryOutput(shape, dtype, maybe_symbolic);
  IrBuilder::create<RNGOp>(
      RNGOpType::Uniform,
      out,
      dtype,
      std::vector<Val*>{},
      philox_seed,
      philox_offset);
  return out;
}

// TENSOR FACTORIES
TensorView* uniform(
    const std::vector<Val*>& shape,
    Val* low,
    Val* high,
    DataType dtype,
    Val* philox_seed,
    Val* philox_offset,
    bool maybe_symbolic) {
  TensorView* out = factoryOutput(shape, dtype, maybe_symbolic);
  IrBuilder::create<RNGOp>(
      RNGOpType::UniformRange,
      out,
      dtype,
      std::vector<Val*>{low, high},
      philox_seed,
      philox_offset);
  return out;
}

TensorView* normal(
    const std::vector<Val*>& shape,
    Val* mean,
    Val* std,
    DataType dtype,
    Val* philox_seed,
    Val* philox_offset,
    bool maybe_symbolic) {
  TensorView* out = factoryOutput(shape, dtype, maybe_symbolic);
  IrBuilder::create<RNGOp>(
      RNGOpType::NormalGeneral,
      out,
      dtype,
      std::vector<Val*>{mean, std},
      philox_seed,
      philox_offset);
  return out;
}

TensorView* randn(
    const std::vector<Val*>& shape,
    DataType dtype,
    Val* philox_seed,
    Val* philox_offset,
    bool maybe_symbolic) {
  TensorView* out = factoryOutput(shape, dtype, maybe_symbolic);
  IrBuilder::create<RNGOp>(
      RNGOpType::NormalStandard,
      out,
      dtype,
      std::vector<Val*>{},
      philox_seed,
      philox_offset);
  return out;
}

TensorView* randn_like(TensorView* tv, Val* philox_seed, Val* philox_offset) {
  NVF_CHECK(
      isFloatingPointType(tv->dtype()),
      "input must have floating point type, but got ",
      tv->dtype());
  // Create a new output TV manually so that we carry over IterTypes, instead
  // of inferring them from the shape as we would if we used randn().
  TensorView* out = ops::newOutputTV({tv}, tv->dtype());
  IrBuilder::create<RNGOp>(
      RNGOpType::NormalStandard,
      out,
      tv->dtype(),
      std::vector<Val*>{},
      philox_seed,
      philox_offset);
  return out;
}
TensorView* randn_like(TensorView* tv) {
  return randn_like(tv, nullptr, nullptr);
}
Val* randn_like(Val* v, Val* philox_seed, Val* philox_offset) {
  return randn_like(v->as<TensorView>(), philox_seed, philox_offset);
}
Val* randn_like(Val* v) {
  return randn_like(v->as<TensorView>(), nullptr, nullptr);
}

TensorView* rand_like(TensorView* tv, Val* philox_seed, Val* philox_offset) {
  NVF_CHECK(
      isFloatingPointType(tv->dtype()),
      "input must have floating point type, but got ",
      tv->dtype());
  // Create a new output TV manually so that we carry over IterTypes, instead
  // of inferring them from the shape as we would if we used rand().
  TensorView* out = ops::newOutputTV({tv}, tv->dtype());
  IrBuilder::create<RNGOp>(
      RNGOpType::Uniform,
      out,
      tv->dtype(),
      std::vector<Val*>{},
      philox_seed,
      philox_offset);
  return out;
}
TensorView* rand_like(TensorView* tv) {
  return rand_like(tv, nullptr, nullptr);
}
Val* rand_like(Val* v, Val* philox_seed, Val* philox_offset) {
  return rand_like(v->as<TensorView>(), philox_seed, philox_offset);
}
Val* rand_like(Val* v) {
  return rand_like(v->as<TensorView>(), nullptr, nullptr);
}

TensorView* full(
    const std::vector<Val*>& shape,
    Val* fill_value,
    DataType dtype,
    bool maybe_symbolic) {
  fill_value = maybeCastOp(dtype, fill_value);
  // Create a new output TV manually so that we carry over IterTypes, instead
  // of inferring them from the shape as we would if we used full().
  TensorView* out = factoryOutput(shape, dtype, maybe_symbolic);
  IrBuilder::create<FullOp>(out, fill_value);
  return out;
}

TensorView* full_like(TensorView* tv, Val* fill_value, DataType dtype) {
  fill_value = maybeCastOp(dtype, fill_value);
  TensorView* out = ops::newOutputTV({tv}, dtype);
  IrBuilder::create<FullOp>(out, fill_value);
  return out;
}

TensorView* full_like(TensorView* tv, Val* fill_value) {
  return full_like(tv, fill_value, tv->dtype());
}

Val* full_like(Val* v, Val* fill_value) {
  return full_like(v->as<TensorView>(), fill_value);
}

TensorView* zeros(
    const std::vector<Val*>& shape,
    DataType dtype,
    bool maybe_symbolic) {
  return full(
      shape,
      FusionGuard::getCurFusion()->zeroVal(dtype),
      dtype,
      maybe_symbolic);
}

TensorView* zeros_like(TensorView* tv) {
  return full_like(tv, FusionGuard::getCurFusion()->zeroVal(tv->dtype()));
}

Val* zeros_like(Val* v) {
  return zeros_like(v->as<TensorView>());
}

TensorView* ones(
    const std::vector<Val*>& shape,
    DataType dtype,
    bool maybe_symbolic) {
  return full(
      shape, FusionGuard::getCurFusion()->oneVal(dtype), dtype, maybe_symbolic);
}

TensorView* ones_like(TensorView* tv) {
  return full_like(tv, FusionGuard::getCurFusion()->oneVal(tv->dtype()));
}

Val* ones_like(Val* v) {
  return ones_like(v->as<TensorView>());
}

TensorView* iota(Val* length, Val* start, Val* step, DataType dtype) {
  if (start == nullptr) {
    start = IrBuilder::create<Val>(0L, dtype);
  }
  if (step == nullptr) {
    step = IrBuilder::create<Val>(1L, dtype);
  }
  NVF_CHECK(
      isIntegralType(*length->getDataType()),
      "length must be integer, but get dtype ",
      *length->getDataType());
  NVF_CHECK(
      !isComplexType(*start->getDataType()) &&
          isIntegralType(*start->getDataType()) == isIntegralType(dtype) &&
          isFloatingPointType(*start->getDataType()) ==
              isFloatingPointType(dtype),
      "iota: start dtype does not match specified dtype argument, should be ",
      dtype,
      " but get ",
      *start->getDataType());
  NVF_CHECK(
      !isComplexType(*step->getDataType()) &&
          isIntegralType(*step->getDataType()) == isIntegralType(dtype) &&
          isFloatingPointType(*step->getDataType()) ==
              isFloatingPointType(dtype),
      "iota: step dtype does not match specified dtype argument, should be ",
      dtype,
      " but get ",
      *step->getDataType());

  start = maybeCastOp(dtype, start);
  step = maybeCastOp(dtype, step);

  if (start->isConst() && start->isFloatingPointScalar()) {
    NVF_ERROR(
        std::isfinite(start->value().as<double>()),
        "iota: length, start, step must be finite numbers.");
  }

  if (step->isConst() && step->isFloatingPointScalar()) {
    NVF_ERROR(
        std::isfinite(step->value().as<double>()),
        "iota: length, start, step must be finite numbers.");
  }

  NVF_ERROR(
      !step->isConstScalar() || !step->isZero(),
      "iota: step value must not equal zero.");

  TensorView* out = factoryOutput({length}, dtype);
  IrBuilder::create<IotaOp>(out, length, start, step);
  return out;
}

TensorView* arange(Val* end, DataType dtype) {
  return arange(FusionGuard::getCurFusion()->zeroVal(dtype), end, dtype);
}

TensorView* arange(Val* start, Val* end, DataType dtype) {
  return arange(start, end, FusionGuard::getCurFusion()->oneVal(dtype), dtype);
}

TensorView* arange(Val* start, Val* end, Val* step, DataType dtype) {
  Val* start_for_size_computation = start;
  Val* end_for_size_computation = end;
  Val* step_for_size_computation = step;
  if (isIntegralType(dtype)) {
    start_for_size_computation = maybeCastOp(DataType::Int, start);
    end_for_size_computation = maybeCastOp(DataType::Int, end);
    step_for_size_computation = maybeCastOp(DataType::Int, step);
  } else if (isFloatingPointType(dtype)) {
    start_for_size_computation = maybeCastOp(DataType::Double, start);
    end_for_size_computation = maybeCastOp(DataType::Double, end);
    step_for_size_computation = maybeCastOp(DataType::Double, step);
  }
  start = maybeCastOp(dtype, start);
  step = maybeCastOp(dtype, step);
  // Make sure no negative value is passed to ceilDiv as the device
  // implementation of ceilDiv assumes positive inputs
  auto distance =
      abs(sub(end_for_size_computation, start_for_size_computation));
  auto abs_step = abs(step_for_size_computation);
  auto length = ceilDiv(distance, abs_step);
  if (!isIntegralType(length->dtype())) {
    length = maybeCastOp(DataType::Index, length);
  }
  return iota(length, start, step, dtype);
}

TensorView* eye(Val* rows, Val* cols, DataType dtype) {
  NVF_CHECK(rows->getDataType() == DataType::Int, "rows must have type Int");
  NVF_CHECK(cols->getDataType() == DataType::Int, "cols must have type Int");
  TensorView* out = factoryOutput({rows, cols}, dtype);
  IrBuilder::create<EyeOp>(out, dtype);
  return out;
}

TensorView* eye(Val* size, DataType dtype) {
  return eye(size, size, dtype);
}

// UNARY OPERATIONS

#define NVFUSER_DEFINE_UNARY_OP(operator_name, operator_type) \
  Val* operator_name(Val* value) {                            \
    return unaryOp(UnaryOpType::operator_type, value);        \
  }                                                           \
  TensorView* operator_name(TensorView* tv) {                 \
    return unaryOp(UnaryOpType::operator_type, tv);           \
  }

NVFUSER_DEFINE_UNARY_OP(ceil, Ceil)
NVFUSER_DEFINE_UNARY_OP(floor, Floor)
NVFUSER_DEFINE_UNARY_OP(frac, Frac)
NVFUSER_DEFINE_UNARY_OP(relu, Relu)
NVFUSER_DEFINE_UNARY_OP(round, Round)
NVFUSER_DEFINE_UNARY_OP(silu, Silu)
NVFUSER_DEFINE_UNARY_OP(trunc, Trunc)
NVFUSER_DEFINE_UNARY_OP(print, Print)
#undef NVFUSER_DEFINE_UNARY_OP

// As a workaround to #1541, we promote half types to single for neg.
// Eventually, `neg` should probably be defined using NVFUSER_DEFINE_UNARY_OP.
// However, currently, nvFuser codegen misses certain header files for half
// types and therefore has no access to data types like `__nv_bfloat16`  and
// intrinsics like `__hneg`.
//
// Note: TypePromotion::float_op_config is a wrong config to use here, because
// it "promotes" integers to float and loses precision (see
// UnaryTests/UnaryTest.Neg/int64_t). TypePromotion::float_only_op_config is
// also wrong, because it doesn't allow integers at all.
Val* neg(Val* v) {
  return unaryOp(UnaryOpType::Neg, v, TypePromotion::default_op_config);
}

TensorView* neg(TensorView* tv) {
  return unaryOp(UnaryOpType::Neg, tv, TypePromotion::default_op_config);
}

Val* logical_not(Val* v) {
  v = maybeCastOp(DataType::Bool, v);
  return unaryOp(UnaryOpType::LogicalNot, v);
}

TensorView* logical_not(TensorView* tv) {
  tv = maybeCastOp(DataType::Bool, tv);
  return unaryOp(UnaryOpType::LogicalNot, tv);
}

Val* bitwise_not(Val* v) {
  if (!isIntegralType(v->dtype())) {
    NVF_CHECK(
        isBooleanType(v->dtype()),
        "input must have integral or boolean type, but got ",
        v->dtype());
    v = castOp(DataType::Int, v);
    return logical_not(v);
  }
  return unaryOp(UnaryOpType::BitwiseNot, v);
}

TensorView* bitwise_not(TensorView* tv) {
  if (!isIntegralType(tv->dtype())) {
    NVF_CHECK(
        isBooleanType(tv->dtype()),
        "input must have integral or boolean type, but got ",
        tv->dtype());
    tv = castOp(DataType::Int, tv);
    return logical_not(tv);
  }
  return unaryOp(UnaryOpType::BitwiseNot, tv);
}

// The output of abs(complex_tensor) are real numbers
Val* abs(Val* v) {
  if (v->getDataType() == DataType::ComplexDouble) {
    Val* out = ops::newValLike(v, DataType::Double);
    IrBuilder::create<UnaryOp>(UnaryOpType::Abs, out, v);
    return out;
  }
  if (v->getDataType() == DataType::ComplexFloat) {
    Val* out = ops::newValLike(v, DataType::Float);
    IrBuilder::create<UnaryOp>(UnaryOpType::Abs, out, v);
    return out;
  }
  return unaryOp(UnaryOpType::Abs, v);
}

TensorView* abs(TensorView* tv) {
  return abs(tv->as<Val>())->as<TensorView>();
}

// The output of signbit(tensor) are boolean values
Val* signbit(Val* v) {
  auto cast_v = promoteValues(TypePromotion::default_op_config, {v}).front();
  Val* out = ops::newValLike(v, DataType::Bool);
  IrBuilder::create<UnaryOp>(UnaryOpType::Signbit, out, cast_v);
  return out;
}

TensorView* signbit(TensorView* tv) {
  return signbit(tv->as<Val>())->as<TensorView>();
}

// The output of real(complex_tensor) are real numbers
Val* real(Val* v) {
  if (v->getDataType() == DataType::ComplexDouble) {
    Val* out = ops::newValLike(v, DataType::Double);
    IrBuilder::create<UnaryOp>(UnaryOpType::Real, out, v);
    return out;
  }
  if (v->getDataType() == DataType::ComplexFloat) {
    Val* out = ops::newValLike(v, DataType::Float);
    IrBuilder::create<UnaryOp>(UnaryOpType::Real, out, v);
    return out;
  }
  // We use LoadStoreOp instead of UnaryOpType::Real to support non-complex
  // tensors
  return set(v);
}

TensorView* real(TensorView* tv) {
  return real(tv->as<Val>())->as<TensorView>();
}

// The output of imag(complex_tensor) are real numbers
Val* imag(Val* v) {
  if (v->getDataType() == DataType::ComplexDouble) {
    Val* out = ops::newValLike(v, DataType::Double);
    IrBuilder::create<UnaryOp>(UnaryOpType::Imag, out, v);
    return out;
  }
  if (v->getDataType() == DataType::ComplexFloat) {
    Val* out = ops::newValLike(v, DataType::Float);
    IrBuilder::create<UnaryOp>(UnaryOpType::Imag, out, v);
    return out;
  }
  NVF_CHECK(false, "imag not supported for non-complex tensors");
}

TensorView* imag(TensorView* tv) {
  return imag(tv->as<Val>())->as<TensorView>();
}

// construct complex tensor from real and imag tensors
Val* complex(Val* r, Val* i) {
  DataType dtype = r->getDataType().value();
  NVF_CHECK(
      dtype == i->getDataType().value(),
      "real and imag data type should be same in complex().");
  Val* out = ops::newValLike(r, getComplexTypeFromType(dtype));
  IrBuilder::create<BinaryOp>(BinaryOpType::Complex, out, r, i);
  return out;
}

TensorView* complex(TensorView* tv_r, TensorView* tv_i) {
  return complex(tv_r->as<Val>(), tv_i->as<Val>())->as<TensorView>();
}
// UNARY FLOAT CAST OPERATIONS

#define NVFUSER_DEFINE_UNARY_FLOAT_OP(op_name, op_type)                       \
  Val* op_name(Val* v) {                                                      \
    return unaryOp(UnaryOpType::op_type, v, TypePromotion::float_op_config);  \
  }                                                                           \
  TensorView* op_name(TensorView* tv) {                                       \
    return unaryOp(UnaryOpType::op_type, tv, TypePromotion::float_op_config); \
  }

NVFUSER_DEFINE_UNARY_FLOAT_OP(acos, Acos)
NVFUSER_DEFINE_UNARY_FLOAT_OP(acosh, Acosh)
NVFUSER_DEFINE_UNARY_FLOAT_OP(asin, Asin)
NVFUSER_DEFINE_UNARY_FLOAT_OP(asinh, Asinh)
NVFUSER_DEFINE_UNARY_FLOAT_OP(atan, Atan)
NVFUSER_DEFINE_UNARY_FLOAT_OP(atanh, Atanh)
NVFUSER_DEFINE_UNARY_FLOAT_OP(cos, Cos)
NVFUSER_DEFINE_UNARY_FLOAT_OP(cosh, Cosh)
NVFUSER_DEFINE_UNARY_FLOAT_OP(exp, Exp)
NVFUSER_DEFINE_UNARY_FLOAT_OP(exp2, Exp2)
NVFUSER_DEFINE_UNARY_FLOAT_OP(expm1, Expm1)
NVFUSER_DEFINE_UNARY_FLOAT_OP(erf, Erf)
NVFUSER_DEFINE_UNARY_FLOAT_OP(erfc, Erfc)
NVFUSER_DEFINE_UNARY_FLOAT_OP(erfinv, Erfinv)
NVFUSER_DEFINE_UNARY_FLOAT_OP(erfcinv, Erfcinv)
NVFUSER_DEFINE_UNARY_FLOAT_OP(lgamma, Lgamma)
NVFUSER_DEFINE_UNARY_FLOAT_OP(log, Log)
NVFUSER_DEFINE_UNARY_FLOAT_OP(log10, Log10)
NVFUSER_DEFINE_UNARY_FLOAT_OP(log1p, Log1p)
NVFUSER_DEFINE_UNARY_FLOAT_OP(log2, Log2)
NVFUSER_DEFINE_UNARY_FLOAT_OP(reciprocal, Reciprocal)
NVFUSER_DEFINE_UNARY_FLOAT_OP(rsqrt, Rsqrt)
NVFUSER_DEFINE_UNARY_FLOAT_OP(sigmoid, Sigmoid)
NVFUSER_DEFINE_UNARY_FLOAT_OP(sin, Sin)
NVFUSER_DEFINE_UNARY_FLOAT_OP(sinh, Sinh)
NVFUSER_DEFINE_UNARY_FLOAT_OP(sqrt, Sqrt)
NVFUSER_DEFINE_UNARY_FLOAT_OP(tan, Tan)
NVFUSER_DEFINE_UNARY_FLOAT_OP(tanh, Tanh)
#undef NVFUSER_DEFINE_UNARY_FLOAT_OP

#define NVFUSER_DEFINE_UNARY_IS_OP(operator_name, operator_type) \
  Val* operator_name(Val* value) {                               \
    return unaryIsOp(UnaryOpType::operator_type, value);         \
  }                                                              \
  TensorView* operator_name(TensorView* tv) {                    \
    return unaryIsOp(UnaryOpType::operator_type, tv);            \
  }

NVFUSER_DEFINE_UNARY_IS_OP(isfinite, IsFinite)
NVFUSER_DEFINE_UNARY_IS_OP(isinf, IsInf)
NVFUSER_DEFINE_UNARY_IS_OP(isnan, IsNan)
NVFUSER_DEFINE_UNARY_IS_OP(isneginf, IsNegInf)
NVFUSER_DEFINE_UNARY_IS_OP(isposinf, IsPosInf)
NVFUSER_DEFINE_UNARY_IS_OP(isreal, IsReal)
#undef NVFUSER_DEFINE_UNARY_IS_OP

// BINARY OPERATIONS

namespace {
// Helper function to reduce repetitive code
template <typename T1, typename T2>
TensorView* arithOpOverloads(Val* (*func)(Val*, Val*), T1* v1, T2* v2) {
  Val* out = func(v1->template as<Val>(), v2->template as<Val>());
  NVF_ERROR(out->isA<TensorView>());
  return out->as<TensorView>();
}

template <typename T1, typename T2>
TensorView* arithOpOverloads(
    BinaryOpType type,
    T1* v1,
    T2* v2,
    DataType common_dtype) {
  Val* out = binaryOp(
      type, v1->template as<Val>(), v2->template as<Val>(), common_dtype);
  NVF_ERROR(out->isA<TensorView>());
  return out->as<TensorView>();
}

template <typename T1, typename T2, typename T3>
TensorView* arithOpOverloads(
    Val* (*func)(Val*, Val*, Val*),
    T1* v1,
    T2* v2,
    T3* v3) {
  auto vals = ops::maybeBroadcast({v1, v2, v3});
  Val* out = func(
      vals[0]->template as<Val>(),
      vals[1]->template as<Val>(),
      vals[2]->template as<Val>());
  NVF_ERROR(out->isA<TensorView>());
  return out->as<TensorView>();
}

template <typename T1, typename T2, typename T3, typename T4>
TensorView* arithOpOverloads(
    Val* (*func)(Val*, Val*, Val*, Val*),
    T1* v1,
    T2* v2,
    T3* v3,
    T4* v4) {
  auto vals = ops::maybeBroadcast({v1, v2, v3, v4});
  Val* out = func(
      vals[0]->template as<Val>(),
      vals[1]->template as<Val>(),
      vals[2]->template as<Val>(),
      vals[3]->template as<Val>());
  NVF_ERROR(out->isA<TensorView>());
  return out->as<TensorView>();
}

// Output type promotion logic for binary operators
DataType getOutputType(
    BinaryOpType op_type,
    Val* v1,
    Val* v2,
    DataType common_dtype) {
  if (isLogicalOp(op_type)) {
    return DataType::Bool;
  } else if (common_dtype == DataType::Null) {
    return promoteType(v1->getDataType().value(), v2->getDataType().value());
  } else {
    return common_dtype;
  }
}

} // namespace

Val* binaryOp(BinaryOpType type, Val* v1, Val* v2, DataType common_dtype) {
  const auto out_dtype = getOutputType(type, v1, v2, common_dtype);
  const auto out_vtype =
      promoteType(v1->getValType().value(), v2->getValType().value());
  auto vals = ops::maybeBroadcast({v1, v2});
  Val* out = nullptr;
  if (out_vtype == ValType::TensorView) {
    out = ops::newOutputTV(vals, out_dtype);
  } else {
    out = ops::newScalar(out_vtype, out_dtype);
  }
  IrBuilder::create<BinaryOp>(type, out, vals[0], vals[1]);
  return out;
}

TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    Val* v2,
    DataType common_dtype) {
  return arithOpOverloads(type, v1, v2, common_dtype);
}

TensorView* binaryOp(
    BinaryOpType type,
    Val* v1,
    TensorView* v2,
    DataType common_dtype) {
  return arithOpOverloads(type, v1, v2, common_dtype);
}

TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    TensorView* v2,
    DataType common_dtype) {
  return arithOpOverloads(type, v1, v2, common_dtype);
}

Val* binaryOp(
    BinaryOpType type,
    Val* v1,
    Val* v2,
    const TypePromotionConfig& config) {
  std::vector<Val*> operands = {v1, v2};
  auto common_dtype = computeTypes(config, operands);
  auto cast_values = promoteValues(operands, common_dtype);
  return binaryOp(type, cast_values.front(), cast_values.back(), common_dtype);
}

TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    Val* v2,
    const TypePromotionConfig& config) {
  std::vector<Val*> operands = {v1, v2};
  auto common_dtype = computeTypes(config, operands);
  auto cast_values = promoteValues(operands, common_dtype);
  return binaryOp(
      type,
      cast_values.front()->as<TensorView>(),
      cast_values.back(),
      common_dtype);
}

TensorView* binaryOp(
    BinaryOpType type,
    Val* v1,
    TensorView* v2,
    const TypePromotionConfig& config) {
  std::vector<Val*> operands = {v1, v2};
  auto common_dtype = computeTypes(config, operands);
  auto cast_values = promoteValues(operands, common_dtype);
  return binaryOp(
      type,
      cast_values.front(),
      cast_values.back()->as<TensorView>(),
      common_dtype);
}

TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    TensorView* v2,
    const TypePromotionConfig& config) {
  std::vector<Val*> operands = {v1, v2};
  auto common_dtype = computeTypes(config, operands);
  auto cast_values = promoteValues(operands, common_dtype);
  return binaryOp(
      type,
      cast_values.front()->as<TensorView>(),
      cast_values.back()->as<TensorView>(),
      common_dtype);
}

#define NVFUSER_DEFINE_BINARY_FLOAT_OP(op_name, op_type)                \
  Val* op_name(Val* v1, Val* v2) {                                      \
    return binaryOp(                                                    \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_op_config); \
  }                                                                     \
  TensorView* op_name(TensorView* v1, Val* v2) {                        \
    return binaryOp(                                                    \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_op_config); \
  }                                                                     \
  TensorView* op_name(Val* v1, TensorView* v2) {                        \
    return binaryOp(                                                    \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_op_config); \
  }                                                                     \
  TensorView* op_name(TensorView* v1, TensorView* v2) {                 \
    return binaryOp(                                                    \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_op_config); \
  }

NVFUSER_DEFINE_BINARY_FLOAT_OP(truediv, Div)
NVFUSER_DEFINE_BINARY_FLOAT_OP(atan2, Atan2)
#undef NVFUSER_DEFINE_BINARY_FLOAT_OP

// These ops require full-precision floating point types (after float type
// promotion)
#define NVFUSER_DEFINE_BINARY_FLOAT_ONLY_OP(op_name, op_type)                \
  Val* op_name(Val* v1, Val* v2) {                                           \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_only_op_config); \
  }                                                                          \
  TensorView* op_name(TensorView* v1, Val* v2) {                             \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_only_op_config); \
  }                                                                          \
  TensorView* op_name(Val* v1, TensorView* v2) {                             \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_only_op_config); \
  }                                                                          \
  TensorView* op_name(TensorView* v1, TensorView* v2) {                      \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_only_op_config); \
  }
NVFUSER_DEFINE_BINARY_FLOAT_ONLY_OP(nextafter, Nextafter)
#undef NVFUSER_DEFINE_BINARY_FLOAT_ONLY_OP

#define NVFUSER_DEFINE_BINARY_CAST_OP(op_name, op_type)                   \
  Val* op_name(Val* v1, Val* v2) {                                        \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(TensorView* v1, Val* v2) {                          \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(Val* v1, TensorView* v2) {                          \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(TensorView* v1, TensorView* v2) {                   \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }

// Integer binary ops
NVFUSER_DEFINE_BINARY_CAST_OP(div, Div)
NVFUSER_DEFINE_BINARY_CAST_OP(mod, Mod)
NVFUSER_DEFINE_BINARY_CAST_OP(ceilDiv, CeilDiv)
NVFUSER_DEFINE_BINARY_CAST_OP(add, Add)
NVFUSER_DEFINE_BINARY_CAST_OP(fmod, Fmod)
NVFUSER_DEFINE_BINARY_CAST_OP(mul, Mul)
NVFUSER_DEFINE_BINARY_CAST_OP(pow, Pow)
NVFUSER_DEFINE_BINARY_CAST_OP(remainder, Remainder)
NVFUSER_DEFINE_BINARY_CAST_OP(sub, Sub)
#undef NVFUSER_DEFINE_BINARY_CAST_OP

#define NVFUSER_DEFINE_LOGICAL_OP(op_name, op_type)                       \
  Val* op_name(Val* v1, Val* v2) {                                        \
    v1 = maybeCastOp(DataType::Bool, v1);                                 \
    v2 = maybeCastOp(DataType::Bool, v2);                                 \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(TensorView* v1, Val* v2) {                          \
    v1 = maybeCastOp(DataType::Bool, v1);                                 \
    v2 = maybeCastOp(DataType::Bool, v2);                                 \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(Val* v1, TensorView* v2) {                          \
    v1 = maybeCastOp(DataType::Bool, v1);                                 \
    v2 = maybeCastOp(DataType::Bool, v2);                                 \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(TensorView* v1, TensorView* v2) {                   \
    v1 = maybeCastOp(DataType::Bool, v1);                                 \
    v2 = maybeCastOp(DataType::Bool, v2);                                 \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }

NVFUSER_DEFINE_LOGICAL_OP(logical_and, LogicalAnd)
NVFUSER_DEFINE_LOGICAL_OP(logical_or, LogicalOr)
#undef NVFUSER_DEFINE_LOGICAL_OP

#define NVFUSER_DEFINE_BITWISE_OP(op_name, op_type, bool_alternative)     \
  Val* op_name(Val* v1, Val* v2) {                                        \
    if (isBooleanType(v1->dtype()) && isBooleanType(v2->dtype())) {       \
      return bool_alternative(v1, v2);                                    \
    }                                                                     \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(TensorView* v1, Val* v2) {                          \
    if (isBooleanType(v1->dtype()) && isBooleanType(v2->dtype())) {       \
      return bool_alternative(v1, v2);                                    \
    }                                                                     \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(Val* v1, TensorView* v2) {                          \
    if (isBooleanType(v1->dtype()) && isBooleanType(v2->dtype())) {       \
      return bool_alternative(v1, v2);                                    \
    }                                                                     \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(TensorView* v1, TensorView* v2) {                   \
    if (isBooleanType(v1->dtype()) && isBooleanType(v2->dtype())) {       \
      return bool_alternative(v1, v2);                                    \
    }                                                                     \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }

NVFUSER_DEFINE_BITWISE_OP(bitwise_and, BitwiseAnd, logical_and)
NVFUSER_DEFINE_BITWISE_OP(bitwise_or, BitwiseOr, logical_or)
NVFUSER_DEFINE_BITWISE_OP(bitwise_xor, BitwiseXor, ne)
#undef NVFUSER_DEFINE_BITWISE_OP

#define NVFUSER_DEFINE_INT_ONLY_OP(op_name, op_type)                      \
  Val* op_name(Val* v1, Val* v2) {                                        \
    NVF_CHECK(                                                            \
        isIntegralType(v1->dtype()) && isIntegralType(v2->dtype()),       \
        "input must have integral type, but got ",                        \
        v1->dtype(),                                                      \
        " and ",                                                          \
        v2->dtype());                                                     \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(TensorView* v1, Val* v2) {                          \
    NVF_CHECK(                                                            \
        isIntegralType(v1->dtype()) && isIntegralType(v2->dtype()),       \
        "input must have integral type, but got ",                        \
        v1->dtype(),                                                      \
        " and ",                                                          \
        v2->dtype());                                                     \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(Val* v1, TensorView* v2) {                          \
    NVF_CHECK(                                                            \
        isIntegralType(v2->dtype()) && isIntegralType(v2->dtype()),       \
        "input must have integral type, but got ",                        \
        v1->dtype(),                                                      \
        " and ",                                                          \
        v2->dtype());                                                     \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(TensorView* v1, TensorView* v2) {                   \
    NVF_CHECK(                                                            \
        isIntegralType(v1->dtype()) && isIntegralType(v2->dtype()),       \
        "input must have integral type, but got ",                        \
        v1->dtype(),                                                      \
        " and ",                                                          \
        v2->dtype());                                                     \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }

NVFUSER_DEFINE_INT_ONLY_OP(bitwise_left_shift, Lshift)
NVFUSER_DEFINE_INT_ONLY_OP(bitwise_right_shift, Rshift)
NVFUSER_DEFINE_INT_ONLY_OP(gcd, Gcd)
#undef NVFUSER_DEFINE_INT_ONLY_OP

// The logical_right_shift operation shifts the value's bits to the right.
// If the value is negative, it appends zeros to the front of the value.
// The sign is preserved with arithmetic_right_shift, so ones are pushed to the
// most significant bit.
//
// An alternate approach is to cast the value to an unsigned integer, perform
// the right shift, and then cast back to the original value. In C++, unsigned
// integers are shifted with logical right shift.
template <typename LHS, typename RHS>
typename std::conditional<
    std::is_same<LHS, TensorView*>::value ||
        std::is_same<RHS, TensorView*>::value,
    TensorView*,
    Val*>::type
logical_right_shift_helper(LHS x, RHS shift) {
  auto sizeof_int_dtype = (x->dtype() == PrimDataType::Int) ? 64L : 32L;

  auto neg_one = IrBuilder::create<Val>(x->container(), -1L);
  auto one = IrBuilder::create<Val>(x->container(), 1L);
  auto two = IrBuilder::create<Val>(x->container(), 2L);
  auto num_bits_scalar =
      IrBuilder::create<Val>(x->container(), sizeof_int_dtype);

  auto mask =
      where(ge(shift, num_bits_scalar), neg_one, sub(pow(two, shift), one));
  auto shifted_mask = bitwise_left_shift(mask, sub(num_bits_scalar, shift));
  auto right_shift_value = bitwise_right_shift(x, shift);
  return where(
      signbit(x),
      bitwise_xor(shifted_mask, right_shift_value),
      right_shift_value);
}

TensorView* logical_right_shift(TensorView* x, TensorView* shift) {
  return logical_right_shift_helper(x, shift);
}
TensorView* logical_right_shift(TensorView* x, Val* shift) {
  return logical_right_shift_helper(x, shift);
}
TensorView* logical_right_shift(Val* x, TensorView* shift) {
  return logical_right_shift_helper(x, shift);
}
Val* logical_right_shift(Val* x, Val* shift) {
  return logical_right_shift_helper(x, shift);
}

#define NVFUSER_DEFINE_BINARY_COMPARE_OP(op_name, op_type)                   \
  Val* op_name(Val* v1, Val* v2) {                                           \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::comparison_op_config); \
  }                                                                          \
  TensorView* op_name(TensorView* v1, Val* v2) {                             \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::comparison_op_config); \
  }                                                                          \
  TensorView* op_name(Val* v1, TensorView* v2) {                             \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::comparison_op_config); \
  }                                                                          \
  TensorView* op_name(TensorView* v1, TensorView* v2) {                      \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::comparison_op_config); \
  }

// Logical binary ops
NVFUSER_DEFINE_BINARY_COMPARE_OP(eq, Eq)
NVFUSER_DEFINE_BINARY_COMPARE_OP(ge, GE)
NVFUSER_DEFINE_BINARY_COMPARE_OP(gt, GT)
NVFUSER_DEFINE_BINARY_COMPARE_OP(le, LE)
NVFUSER_DEFINE_BINARY_COMPARE_OP(lt, LT)
NVFUSER_DEFINE_BINARY_COMPARE_OP(ne, NE)
#undef NVFUSER_DEFINE_BINARY_COMPARE_OP

// REDUCTION OPERATIONS

// TODO: How do we adjust this so we can reduce to a single scalar value?
static TensorView* newForReduction(
    TensorView* tv,
    const std::vector<unsigned int>& axes,
    DataType data_type = DataType::Null) {
  auto orig_domain = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  std::set<unsigned int> axes_set(axes.begin(), axes.end());

  std::vector<IterDomain*> new_domain;

  NVF_ERROR(
      !axes_set.empty(),
      "Asked for output of reduction, but no reduction axis provided.");

  NVF_ERROR(
      (*(axes_set.rbegin())) < orig_domain.size(),
      "Error setting up reduction, reduction axis (",
      *(axes_set.rbegin()),
      ") is outside nDims (",
      orig_domain.size(),
      "). Keep in mind reductions are relative to root domains, not modified views.");

  auto axis_iter = axes_set.begin();
  for (const auto dim : c10::irange(orig_domain.size())) {
    bool isReduction = false;
    if (axis_iter != axes_set.end() && *axis_iter == dim) {
      isReduction = true;
      axis_iter++;
    }

    const IterDomain* id = orig_domain[dim];

    NVF_CHECK(
        !(isReduction && id->isBroadcast() && !id->isImplicitBroadcast()),
        "Cannot reduce an axis that is marked as broadcasted as it has an undetermined size. Tried to reduce ID = ",
        id,
        " of tensor ",
        tv);

    new_domain.push_back(
        IterDomainBuilder(id)
            // If the domain is being reduced, but it's coming in as an expanded
            // extent, we need to realize the expand.
            .extent(
                isReduction && id->hasExpandedExtent() ? id->expandedExtent()
                                                       : id->extent())
            .resetSchedulingParams()
            .iter_type(isReduction ? IterType::Reduction : id->getIterType())
            .build());
  }

  TensorDomain* td = IrBuilder::create<TensorDomain>(
      new_domain, TensorDomain::getContiguityFilledWith(new_domain, true));

  data_type =
      data_type == DataType::Null ? tv->getDataType().value() : data_type;
  return IrBuilder::create<TensorView>(td, data_type);
}

namespace {

// PyTorch accepts reductions of zero-dimensional tensors, which are
// just ignored.
TensorView* reductionOpZeroDimTensor(TensorView* inp) {
  NVF_ERROR(inp->domain()->noReductions().empty());
  return set(inp);
}

} // namespace

TensorView* reductionOpRaw(
    BinaryOpType reduction_op_type,
    const std::vector<int>& axes,
    Val* init,
    TensorView* tv,
    bool keep_dim /*=false*/,
    DataType dtype /*  DataType::Null */) {
  // TODO: should we use squeeze for size 1 broadcast dim?

  NVF_CHECK(
      init->isConstScalar(),
      "Cannot create a reduction operation where the initial value is not a const scalar.");

  NVF_CHECK(
      TensorDomain::sameAs(tv->getMaybeRFactorDomain(), tv->getLeafDomain()),
      "Reducing a tensor once it's gone under transformations is not permitted at this time. \n",
      "Please set reductions before calling split/merge/computeAt.\n  RFactor: ",
      tv->getMaybeRFactorDomain(),
      "\n  Domain: ",
      tv->domain()->toString());

  NVF_CHECK(!axes.empty(), "No reduction axis specified");

  // PyTorch allows reduction of 0-dim tensors
  if (tv->domain()->noReductions().empty()) {
    return reductionOpZeroDimTensor(tv);
  }

  std::vector<unsigned int> uint_axes =
      ops::canonicalizeAxes(axes, tv->domain()->noReductions().size());

  TensorView* out = newForReduction(tv, uint_axes, dtype);
  const auto out_type = out->getDataType().value();
  const auto init_type = init->getDataType().value();
  NVF_CHECK(
      (isFloatingPointType(out_type) && isFloatingPointType(init_type)) ||
          (isComplexType(out_type) && isComplexType(init_type)) ||
          (isIntegralType(out_type) && isIntegralType(init_type)) ||
          (isBooleanType(out_type) && isBooleanType(init_type)),
      "Types should match for reduction ops but received: ",
      out_type,
      " and ",
      init_type);
  IrBuilder::create<ReductionOp>(reduction_op_type, init, out, tv);

  if (keep_dim) {
    auto tv_root = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
    std::vector<bool> is_broadcast(tv_root.size(), false);
    for (auto axis : uint_axes) {
      is_broadcast.at(axis) = true;
    }
    out = broadcast(out, is_broadcast);
  }
  return out;
}

namespace {

TensorView* maybeFullInsteadOfReduction(
    const std::vector<unsigned int>& axes, // sorted
    Val* init,
    TensorView* tv,
    bool keep_dim,
    DataType dtype) {
  auto tv_root = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  const auto ndims = tv_root.size();
  for (auto i : axes) {
    if (tv_root.at(i)->extent()->isZeroInt()) {
      std::vector<IterDomain*> new_root;
      new_root.reserve(keep_dim ? ndims : ndims - axes.size());
      int cur_pos = 0;
      for (auto j : c10::irange(ndims)) {
        bool is_reduction = cur_pos < (int)axes.size() && axes.at(cur_pos) == j;
        if (is_reduction) {
          cur_pos++;
          if (keep_dim) {
            auto id = IterDomainBuilder(
                          tv->fusion()->zeroVal(), tv->fusion()->oneVal())
                          .iter_type(IterType::Broadcast)
                          .build();
            new_root.push_back(id);
          }
        } else {
          new_root.push_back(tv_root.at(j)->cloneWithoutRFactor());
        }
      }

      TensorDomain* td = IrBuilder::create<TensorDomain>(
          new_root, TensorDomain::getContiguityFilledWith(new_root, true));

      dtype = (dtype == DataType::Null ? tv->getDataType().value() : dtype);
      auto output = IrBuilder::create<TensorView>(td, dtype);
      init = maybeCastOp(dtype, init);
      IrBuilder::create<FullOp>(output, init);
      return output;
    }
  }
  return nullptr;
}

} // namespace

TensorView* reductionOp(
    BinaryOpType reduction_op_type,
    const std::vector<int>& axes,
    Val* init,
    TensorView* tv,
    bool keep_dim /*=false*/,
    DataType dtype /* DataType::Null */) {
  NVF_CHECK(
      init->isConstScalar(),
      "Cannot create a reduction operation where the initial value is not a const scalar.");

  NVF_CHECK(
      TensorDomain::sameAs(tv->getMaybeRFactorDomain(), tv->getLeafDomain()),
      "Reducing a tensor once it's gone under transformations is not permitted at this time. \n",
      "Please set reductions before calling split/merge/computeAt.\n  RFactor: ",
      tv->getMaybeRFactorDomain(),
      "\n  Domain: ",
      tv->domain()->toString());

  NVF_CHECK(!axes.empty(), "No reduction axis specified");

  auto tv_root = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  const auto ndims = tv_root.size();

  // PyTorch allows reduction of 0-dim tensors
  if (ndims == 0) {
    return reductionOpZeroDimTensor(tv);
  }

  std::vector<unsigned int> uint_axes = ops::canonicalizeAxes(axes, ndims);
  std::sort(uint_axes.begin(), uint_axes.end());

  // In PyTorch, reduction of a size-0 tensor is effectively creating a tensor
  // filled with the init value.
  auto maybe_full =
      maybeFullInsteadOfReduction(uint_axes, init, tv, keep_dim, dtype);
  if (maybe_full != nullptr) {
    return maybe_full;
  }

  // [Trivial reductions]
  // When we reduce a simple broadcast axis like bS0{1} the effect is just to
  // squeeze out that broadcast axis. When the axis is expanded, such as bS1{1
  // ex i0}, then the effect depends on the op type. We have the following
  // mappings from op_type to expanded reduction equivalent:
  //   Add -> multiplication by i0
  //   Mul -> raise to power i0
  //   Min/Max -> squeeze
  //   {Logical,Bitwise}{And,Or} -> squeeze
  //   BitwiseXor -> 0 if i0 is even, else squeeze
  //   Eq -> squeeze
  //   Gcd -> squeeze
  // Other op-types are non-commutative, so we ignore them here as they should
  // not be used in reductions. We can see that the only two common ops that
  // require special consideration are Add and Mul. Currently Xor is not
  // supported for expanded reduction. We treat all others as trivial (i.e.
  // squeeze).
  std::vector<int> reduction_axes;
  std::vector<bool> is_squeeze(ndims, false);
  bool expand_reductions_are_trivial = reduction_op_type != BinaryOpType::Add &&
      reduction_op_type != BinaryOpType::Mul &&
      reduction_op_type != BinaryOpType::BitwiseXor;
  int offset = 0;
  for (unsigned int axis : uint_axes) {
    auto id = tv_root[axis];
    if (id->isBroadcast()) {
      is_squeeze[axis] = true;
      offset--;
    } else {
      reduction_axes.push_back((int)axis + offset);
    }
  }

  TensorView* squeezed = tv;
  if (offset < 0) {
    // There are some broadcast dims being reduced. We squeeze them all first.
    squeezed = squeeze(tv, is_squeeze, /*squeeze_expanded=*/true);
  }

  TensorView* out = squeezed;
  if (!reduction_axes.empty()) {
    out = reductionOpRaw(
        reduction_op_type, reduction_axes, init, squeezed, keep_dim, dtype);
  }

  if (!expand_reductions_are_trivial) {
    Val* factor = nullptr;
    for (auto axis : uint_axes) {
      IterDomain* id = tv_root[axis];
      if (id->isBroadcast() && id->hasExpandedExtent()) {
        factor =
            SimplifyingIrBuilder::mulExpr(factor, id->getMaybeExpandedExtent());
      }
    }
    if (factor != nullptr) {
      factor = SimplifyingIrBuilder::maybeCastExpr(out->dtype(), factor);
      if (reduction_op_type == BinaryOpType::Add) {
        out = mul(out, factor);
      } else if (reduction_op_type == BinaryOpType::Mul) {
        out = pow(out, factor);
      } else {
        NVF_ERROR(
            false,
            "Add and Mul are the only non-trivial expand reductions allowed");
      }
    }
  }

  if (keep_dim && offset < 0) {
    // There were squeezed dimension removed from squeeze that will not be
    // restored by reductionOpRaw above, so we restore them here
    out = broadcast(out, is_squeeze);
  }

  if (out == tv) {
    // makes sure that a new tensor is created
    return set(tv);
  }

  return out;
}

TensorView* sum(
    TensorView* v1,
    const std::vector<int>& axes,
    bool keep_dim /*=false*/,
    DataType dtype /* DataType::Null */) {
  if (dtype == DataType::Null) {
    auto initial_v1_dtype = v1->getDataType().value();
    if (isBooleanType(initial_v1_dtype) || isIntegralType(initial_v1_dtype)) {
      dtype = DataType::Int;
    }
  }

  // Cast input tensor to dtype before the operation is performed
  if (dtype != DataType::Null) {
    v1 = optionalCastStrict(dtype, v1)->as<TensorView>();
  }

  auto init = FusionGuard::getCurFusion()->zeroVal(v1->getDataType().value());
  return reductionOp(BinaryOpType::Add, axes, init, v1, keep_dim, dtype);
}

TensorView* prod(
    TensorView* v1,
    const std::vector<int>& axes,
    bool keep_dim /*=false*/,
    DataType dtype /* DataType::Null */) {
  if (dtype == DataType::Null) {
    auto initial_v1_dtype = v1->getDataType().value();
    if (isBooleanType(initial_v1_dtype) || isIntegralType(initial_v1_dtype)) {
      dtype = DataType::Int;
    }
  }

  // Cast input tensor to dtype before the operation is performed
  if (dtype != DataType::Null) {
    v1 = optionalCastStrict(dtype, v1)->as<TensorView>();
  }

  auto init = FusionGuard::getCurFusion()->oneVal(v1->getDataType().value());
  return reductionOp(BinaryOpType::Mul, axes, init, v1, keep_dim, dtype);
}

TensorView* max(
    TensorView* v1,
    const std::vector<int>& axes,
    bool keep_dim /*=false*/,
    DataType dtype /* DataType::Null */) {
  NVF_CHECK(
      dtype == DataType::Null,
      "A dtype other than Null is not currently supported.");
  Val* init = ops::getMinimumValue(v1->getDataType().value());
  NVF_CHECK(init != nullptr, "Missing initial value");
  return reductionOp(BinaryOpType::Max, axes, init, v1, keep_dim);
}

TensorView* min(
    TensorView* v1,
    const std::vector<int>& axes,
    bool keep_dim /*=false*/,
    DataType dtype /* DataType::Null */) {
  NVF_CHECK(
      dtype == DataType::Null,
      "A dtype other than Null is not currently supported.");
  Val* init = ops::getMaximumValue(v1->getDataType().value());
  NVF_CHECK(init != nullptr, "Missing initial value");
  return reductionOp(BinaryOpType::Min, axes, init, v1, keep_dim);
}

TensorView* broadcast(
    TensorView* inp,
    const std::vector<bool>& is_broadcast_dim) {
  auto nBCastDims = is_broadcast_dim.size();
  // Validate is_broadcast_dim
  unsigned int n_broadcasts = 0;
  for (auto ent : is_broadcast_dim) {
    if (ent) {
      n_broadcasts++;
    }
  }

  NVF_CHECK(
      nBCastDims - n_broadcasts ==
          TensorDomain::noReductions(inp->getMaybeRFactorDomain()).size(),
      "Invalid broadcast, number of false entries in is_broadcast_dim expected to be ",
      TensorDomain::noReductions(inp->getMaybeRFactorDomain()).size(),
      " but received ",
      nBCastDims - n_broadcasts);

  if (n_broadcasts == 0) {
    auto identity = set(inp);
    NVF_ERROR(
        identity->getValType().value() == ValType::TensorView,
        "Expected identity op, but didn't get a TensorView back.");
    return identity->as<TensorView>();
  }

  std::vector<IterDomain*> out_domain;
  // Don't propagate reduction IDs through arith ops.
  auto inp_domain = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  size_t iinp = 0, ibdim = 0;
  while (ibdim < is_broadcast_dim.size()) {
    if (is_broadcast_dim[ibdim]) {
      out_domain.push_back(IterDomainBuilder(
                               FusionGuard::getCurFusion()->zeroVal(),
                               FusionGuard::getCurFusion()->oneVal())
                               .iter_type(IterType::Broadcast)
                               .build());
    } else {
      out_domain.push_back(
          IterDomainBuilder(inp_domain[iinp]).resetSchedulingParams().build());
      iinp++;
    }
    ibdim++;
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      inp->getDataType().value());
  IrBuilder::create<BroadcastOp>(out_tensor, inp, is_broadcast_dim);
  return out_tensor;
}

TensorView* expand(TensorView* inp, const std::vector<Val*>& expanded_sizes) {
  auto inp_domain = TensorDomain::noReductions(inp->getMaybeRFactorDomain());

  NVF_CHECK(
      expanded_sizes.size() >= inp_domain.size(),
      "Invalid expand, number of sizes provided is expected to be at least ",
      inp_domain.size(),
      " but received ",
      expanded_sizes.size());

  inp = ops::maybe_broadcast_inner_to_rank(inp, expanded_sizes.size());
  inp_domain = TensorDomain::noReductions(inp->getMaybeRFactorDomain());

  std::vector<Val*> maybe_expanded_sizes;
  maybe_expanded_sizes.resize(inp_domain.size(), nullptr);

  // Might a dimension actually get expanded? This will be true if any input
  // IterDomains are Symbolic, since these may or may not be Broadcast.
  bool expanded = false;

  std::vector<IterDomain*> out_domain;
  for (auto i : c10::irange(inp_domain.size())) {
    auto inp_id = inp_domain[i];
    auto out_id_builder = IterDomainBuilder(inp_id);
    maybe_expanded_sizes[i] = inp_domain[i]->extent();

    auto expanded_size_int = expanded_sizes[i]->value();

    // If the expanded size is -1, let the input extent be propagated
    // as is
    if (expanded_size_int.hasValue() && expanded_size_int == -1) {
      // This is just done for clarity. It isn't necessary as it's
      // already done when constructing out_id_builder.
      out_id_builder.extent(inp_id->extent());
    } else if (
        // special patch for Symbolic IterDomain with a static size-1 extent
        // since we know it will become broadcast at concretization
        // See Issue: https://github.com/NVIDIA/Fuser/pull/1393
        (inp_id->extent()->isConstInt() && inp_id->extent()->evaluate() == 1) &&
        (!expanded_size_int.hasValue() || expanded_size_int != 1)) {
      // When input id is a broadcast, expand the extent to the given
      // size, which can be concrete or symbolic.
      expanded = true;
      auto expanded_extent = maybeCastOp(DataType::Index, expanded_sizes[i]);
      out_id_builder.expanded_extent(expanded_extent);
      // need to mark iter type as Broadcast for Symbolic input domains
      out_id_builder.iter_type(IterType::Broadcast);
      maybe_expanded_sizes[i] = expanded_extent;
    } else if (
        inp_id->isSymbolic() &&
        (!inp_id->extent()->isConstInt() &&
         !inp_id->extent()->sameAs(expanded_sizes[i]))) {
      // need to mark iter type as Symbolic since this might not be an expand
      // after concretization
      expanded = true;
      out_id_builder.iter_type(IterType::Symbolic);
      auto expanded_extent = maybeCastOp(DataType::Index, expanded_sizes[i]);
      // We set the extent instead of the expanded extent on a Symbolic
      // IterDomain. At concretization, if the IterType is determined to be
      // Broadcast, we will replace this with 1 and use the old extent as
      // expandedExtent.
      out_id_builder.extent(expanded_extent);
      maybe_expanded_sizes[i] = expanded_extent;
    } else if (!inp_id->extent()->isConstInt()) {
      // Input id is non-broadcast and its extent is symbolic. Promote
      // the extent to the given expanded size.
      // Note that expansion to 1 just means its extent becomes 1 and
      // does not mean the ID becomes a broadcast.
      out_id_builder.extent(maybeCastOp(DataType::Index, expanded_sizes[i]));
    } else {
      // Input id is non-expand and its extent is concrete. Nothing
      // to expand, but the input and expanded sizes should match if
      // the expanded size is also concrete.
      auto inp_id_size_int = inp_id->extent()->evaluate();
      if (expanded_size_int.is<int64_t>()) {
        NVF_CHECK(
            inp_id_size_int == expanded_size_int,
            "Invalid expand size, ",
            expanded_sizes[i]->toString(),
            ", for ",
            inp_id->toString());
      }
    }
    out_domain.push_back(out_id_builder.build());
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      inp->getDataType().value());
  if (!expanded) {
    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out_tensor, inp);
  } else {
    IrBuilder::create<ExpandOp>(out_tensor, inp, maybe_expanded_sizes);
  }
  return out_tensor;
}

TensorView* expand_as(TensorView* inp, TensorView* other) {
  auto inp_domain = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  auto other_domain =
      TensorDomain::noReductions(other->getMaybeRFactorDomain());

  NVF_CHECK(
      inp_domain.size() <= other_domain.size(),
      "Invalid expand_as, dimensions of inp is higher than dimensions of other, expected other to be at least ",
      inp_domain.size(),
      " but received ",
      other_domain.size());

  inp = ops::maybe_broadcast_inner_to_rank(inp, other_domain.size());
  inp_domain = TensorDomain::noReductions(inp->getMaybeRFactorDomain());

  std::vector<IterDomain*> out_domain;
  std::vector<Val*> maybe_expanded_sizes;
  bool expanded = false;
  for (auto i : c10::irange(inp_domain.size())) {
    auto inp_id = inp_domain[i];
    auto other_id = other_domain[i];

    auto out_id_builder = IterDomainBuilder(inp_id);
    Val* maybe_expanded_size = inp_id->extent();

    if (!inp_id->isBroadcast()) {
      NVF_ERROR(
          !other_id->isBroadcast(),
          "Cannot expand as a tensor if other has broadcast dimensions that don't map to broadcast dimensions in the input.");
      if (!inp_id->isConstInt() && other_id->isConstInt()) {
        out_id_builder.extent(
            ops::promoteSize(inp_id->extent(), other_id->extent()));
      }
    } else {
      if (!other_id->isBroadcast()) {
        expanded = true;
        out_id_builder.expanded_extent(other_id->extent());
        maybe_expanded_size = other_id->extent();
      } else if (other_id->isBroadcast() && other_id->hasExpandedExtent()) {
        expanded = true;
        out_id_builder.expanded_extent(other_id->expandedExtent());
        maybe_expanded_size = other_id->expandedExtent();
      }
    }
    out_domain.push_back(out_id_builder.build());
    maybe_expanded_sizes.push_back(maybe_expanded_size);
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      inp->getDataType().value());
  if (!expanded) {
    IrBuilder::create<LoadStoreOp>(LoadStoreOpType::Set, out_tensor, inp);
  } else {
    IrBuilder::create<ExpandOp>(out_tensor, inp, maybe_expanded_sizes);
  }
  return out_tensor;
}

std::vector<Val*> tensor_sizes(TensorView* inp) {
  auto iter_domains = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  std::vector<Val*> sizes(iter_domains.size(), nullptr);

  for (auto idx : c10::irange(iter_domains.size())) {
    sizes[idx] = iter_domains[idx]->getMaybeExpandedExtent();
  }

  return sizes;
}

std::vector<Val*> shape(TensorView* inp) {
  auto iter_domains = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  std::vector<Val*> shape;

  shape.reserve(iter_domains.size());
  for (auto id : iter_domains) {
    shape.push_back(id->getMaybeExpandedExtent());
  }

  return shape;
}

Val* size(TensorView* inp, int64_t dim) {
  auto iter_domains = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  auto idx = dim;
  if (idx < 0) {
    idx = static_cast<int64_t>(iter_domains.size()) + idx;
  }
  NVF_CHECK(
      (idx >= 0) && (static_cast<size_t>(idx) < iter_domains.size()),
      __FUNCTION__,
      ": The dimension requested is beyond the bounds of the shape of the indexed tensor!",
      " Tensor Dims: ",
      iter_domains.size(),
      " Dim: ",
      dim);
  return iter_domains.at(idx)->getMaybeExpandedExtent();
}

Val* at(const std::vector<Val*>& inp, int64_t index) {
  auto idx = index;
  if (idx < 0) {
    idx = static_cast<int64_t>(inp.size()) + idx;
  }
  NVF_CHECK(
      (idx >= 0) && (static_cast<size_t>(idx) < inp.size()),
      __FUNCTION__,
      ": The index requested is beyond the bounds of the indexed vector!",
      " Vector Size: ",
      inp.size(),
      " Index: ",
      index);
  return inp.at(idx);
}

WelfordResult WelfordRaw(
    TensorView* tv,
    const std::vector<int>& axes,
    TensorView* init_avg,
    TensorView* init_var,
    Val* init_N) {
  NVF_CHECK(
      TensorDomain::sameAs(tv->getMaybeRFactorDomain(), tv->getLeafDomain()),
      "Reducing a tensor once it's gone under transformations is not permitted at this time. \n",
      "Please set reductions before calling split/merge/computeAt.\n  RFactor: ",
      tv->getMaybeRFactorDomain(),
      "\n  Domain: ",
      tv->domain()->toString());

  NVF_CHECK(tv->nDims() > 0, "Tried to reduce a 0-dim tensor");
  NVF_CHECK(!axes.empty(), "No reduction axis specified");

  if (init_N == nullptr) {
    init_N = FusionGuard::getCurFusion()->zeroVal();
  }

  // Initial values for welford op are tensors, so their dims have to match the
  // output dim,
  // i.e. original_dims - dims_to_be_reduced
  Val* init_avg_val = nullptr;
  Val* init_var_val = nullptr;
  if (!init_N->isZeroInt()) {
    NVF_CHECK(
        init_avg != nullptr && init_var != nullptr && init_N != nullptr,
        "welford op: all init values need to be provided");
    NVF_CHECK(
        (axes.size() + init_avg->getRootDomain().size()) ==
            tv->getRootDomain().size(),
        "welford op: initial tensor mismatch");
    NVF_CHECK(
        (axes.size() + init_var->getRootDomain().size()) ==
            tv->getRootDomain().size(),
        "welford op: initial tensor mismatch");
    init_avg_val = init_avg;
    init_var_val = init_var;
  } else {
    init_avg_val = IrBuilder::create<Val>(0.0);
    init_var_val = IrBuilder::create<Val>(0.0);
  }

  // Check and collect reduction axes
  std::vector<unsigned int> uint_axes =
      ops::canonicalizeAxes(axes, tv->domain()->noReductions().size());
  // Create tensor outputs
  TensorView* out_avg = newForReduction(tv, uint_axes);
  TensorView* out_var = newForReduction(tv, uint_axes);
  TensorView* out_N = newForReduction(tv, uint_axes, DataType::Index);

  IrBuilder::create<WelfordOp>(
      out_avg,
      out_var,
      out_N, /*out avg/var/count */
      tv, /*in avg/var/count */
      FusionGuard::getCurFusion()->zeroVal(),
      FusionGuard::getCurFusion()->oneVal(),
      init_avg_val,
      init_var_val,
      init_N); /*init avg/var/count */
  return WelfordResult(out_avg, out_var, out_N);
}

WelfordResult Welford(
    TensorView* tv,
    const std::vector<int>& axes,
    TensorView* init_avg,
    TensorView* init_var,
    Val* init_N) {
  NVF_CHECK(
      TensorDomain::sameAs(tv->getMaybeRFactorDomain(), tv->getLeafDomain()),
      "Reducing a tensor once it's gone under transformations is not permitted at this time. \n",
      "Please set reductions before calling split/merge/computeAt.\n  RFactor: ",
      tv->getMaybeRFactorDomain(),
      "\n  Domain: ",
      tv->domain()->toString());

  NVF_CHECK(tv->nDims() > 0, "Tried to reduce a 0-dim tensor");
  NVF_CHECK(!axes.empty(), "No reduction axis specified");

  // Check and collect reduction axes
  auto tv_root = tv->domain()->noReductions();
  const auto ndims = tv_root.size();
  std::vector<unsigned int> uint_axes = ops::canonicalizeAxes(axes, ndims);
  std::sort(uint_axes.begin(), uint_axes.end());

  // Squeeze before reduction
  std::vector<int> reduction_axes;
  std::vector<bool> is_trivial_reduction(ndims, false);
  int offset = 0;
  for (auto axis : uint_axes) {
    auto id = tv_root[axis];
    is_trivial_reduction[axis] = id->isBroadcast() &&
        !id->hasExpandedExtent() && id->extent()->isOneInt();
    if (!is_trivial_reduction[axis]) {
      reduction_axes.push_back((int)axis + offset);
    } else {
      offset--;
    }
  }

  TensorView* squeezed = tv;
  if (offset < 0) {
    squeezed = squeeze(tv, is_trivial_reduction, /*squeeze_expanded=*/true);
  }

  if (!reduction_axes.empty()) {
    DataType dtype = tv->getDataType().value();
    if (isComplexType(dtype)) {
      // var of complex number is a real number, calculate real part and image
      // part
      WelfordResult real_part =
          Welford(real(squeezed), reduction_axes, init_avg, init_var, init_N);
      WelfordResult imag_part =
          Welford(imag(squeezed), reduction_axes, init_avg, init_var, init_N);
      TensorView* out_avg = complex(real_part.avg, imag_part.avg);
      TensorView* out_var = add(real_part.var_sum, imag_part.var_sum);
      TensorView* out_N = real_part.n;
      return WelfordResult(out_avg, out_var, out_N, false);
    } else {
      return WelfordRaw(squeezed, reduction_axes, init_avg, init_var, init_N);
    }
  }

  // if squeeze only

  if (init_N == nullptr) {
    init_N = FusionGuard::getCurFusion()->zeroVal();
  }
  TensorView* out_N = full_like(
      squeezed,
      add(init_N, FusionGuard::getCurFusion()->oneVal(init_N->dtype())),
      DataType::Index);

  // Initial values for welford op are tensors, so their dims have to match the
  // output dim
  if (!init_N->isZeroInt()) {
    NVF_CHECK(
        init_var != nullptr,
        "welford op: init variance value need to be provided");
    NVF_CHECK(
        squeezed->getRootDomain().size() == init_var->getRootDomain().size(),
        "welford op: initial tensor mismatch");
    return WelfordResult(squeezed, init_var, out_N, false);
  } else {
    return WelfordResult(
        squeezed,
        full_like(squeezed, IrBuilder::create<Val>(0.0)),
        out_N,
        false);
  }
}

WelfordResult::WelfordResult(
    TensorView* in_avg,
    TensorView* in_var_sum,
    TensorView* in_n,
    const bool check_definition)
    : avg(in_avg), var_sum(in_var_sum), n(in_n) {
  if (!check_definition) {
    // For squeeze-only and complex welford, the definition of outputs does not
    // have to be the same.
    return;
  }
  NVF_ERROR(avg->definition()->sameAs(var_sum->definition()));
  NVF_ERROR(avg->definition()->sameAs(n->definition()));
}

// COMPOUND OPERATIONS

// add_alpha
Val* add_alpha(Val* v1, Val* v2, Val* s) {
  NVF_CHECK(
      s->getValType().value() == ValType::Others,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  std::vector<Val*> operands = {v1, v2};
  auto common_dtype = computeTypes(TypePromotion::default_op_config, operands);
  auto cast_values = promoteValues({v1, v2, s}, common_dtype);
  auto vals = ops::maybeBroadcast(cast_values);
  Val* intrm = mul(vals[1], vals[2]);
  return add(vals[0], intrm);
}
TensorView* add_alpha(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(add_alpha, v1, v2, v3);
}
TensorView* add_alpha(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(add_alpha, v1, v2, v3);
}
TensorView* add_alpha(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(add_alpha, v1, v2, v3);
}
// sub_alpha
Val* sub_alpha(Val* v1, Val* v2, Val* s) {
  NVF_CHECK(
      s->getValType().value() == ValType::Others,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  std::vector<Val*> operands = {v1, v2};
  auto common_dtype = computeTypes(TypePromotion::default_op_config, operands);
  auto cast_values = promoteValues({v1, v2, s}, common_dtype);
  auto vals = ops::maybeBroadcast(cast_values);
  Val* intrm = mul(vals[1], vals[2]);
  return sub(vals[0], intrm);
}
TensorView* sub_alpha(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(sub_alpha, v1, v2, v3);
}
TensorView* sub_alpha(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(sub_alpha, v1, v2, v3);
}
TensorView* sub_alpha(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(sub_alpha, v1, v2, v3);
}
// lerp
Val* lerp(Val* start, Val* end, Val* weight) {
  auto cast_values =
      promoteValues(TypePromotion::default_op_config, {start, end, weight});
  start = cast_values[0];
  end = cast_values[1];
  weight = cast_values[2];

  auto out_dtype =
      promoteType(start->getDataType().value(), end->getDataType().value());
  auto out_vtype =
      promoteType(start->getValType().value(), end->getValType().value());

  auto vals = ops::maybeBroadcast({start, end, weight});
  Val* out = nullptr;
  if (out_vtype == ValType::TensorView) {
    out = ops::newOutputTV(vals, out_dtype);
  } else {
    out = ops::newScalar(out_vtype, out_dtype);
  }

  IrBuilder::create<TernaryOp>(
      TernaryOpType::Lerp, out, vals[0], vals[1], vals[2]);
  return out;
}
TensorView* lerp(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(Val* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(TensorView* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(Val* v1, TensorView* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(TensorView* v1, TensorView* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}

// addcmul
Val* addcmul(Val* v1, Val* v2, Val* v3, Val* s) {
  NVF_CHECK(
      s->getValType().value() == ValType::Others,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  std::vector<Val*> operands = {v1, v2, v3};
  auto common_dtype = computeTypes(TypePromotion::default_op_config, operands);
  auto cast_values = promoteValues({v1, v2, v3, s}, common_dtype);
  auto vals = ops::maybeBroadcast(cast_values);
  Val* intrm1 = mul(vals[2], vals[3]);
  Val* intrm2 = mul(vals[1], intrm1);
  return add(vals[0], intrm2);
}
TensorView* addcmul(TensorView* v1, Val* v2, Val* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(Val* v1, TensorView* v2, Val* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(Val* v1, Val* v2, TensorView* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(TensorView* v1, TensorView* v2, Val* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(TensorView* v1, Val* v2, TensorView* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(Val* v1, TensorView* v2, TensorView* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(TensorView* v1, TensorView* v2, TensorView* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}

// TERNARY OPERATIONS
// where (c ? v1 : v2)
Val* where(Val* c, Val* v1, Val* v2) {
  NVF_CHECK(
      c->getDataType().value() == DataType::Bool,
      "Condition should be of DataType Bool, not ",
      c->getDataType().value());

  std::vector<Val*> operands = {v1, v2};
  auto common_dtype =
      computeTypes(TypePromotion::default_op_config, operands, false);
  auto cast_values = promoteValues(operands, common_dtype);
  v1 = cast_values[0];
  v2 = cast_values[1];

  NVF_CHECK(c->getDataType().value() == DataType::Bool);
  const auto& out_dtype = common_dtype;
  auto out_vtype =
      promoteType(v1->getValType().value(), v2->getValType().value());
  // Even when v1 and v2 are scalar, the output is a tensor if the
  // conditional input is a tensor.
  if (c->getValType() == ValType::TensorView) {
    out_vtype = ValType::TensorView;
  }
  auto vals = ops::maybeBroadcast({c, v1, v2});
  Val* out = nullptr;
  if (out_vtype == ValType::TensorView) {
    out = ops::newOutputTV(vals, out_dtype);
  } else {
    out = ops::newScalar(out_vtype, out_dtype);
  }
  IrBuilder::create<TernaryOp>(
      TernaryOpType::Where, out, vals[0], vals[1], vals[2]);
  return out;
}

TensorView* where(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(Val* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(TensorView* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(Val* v1, TensorView* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(TensorView* v1, TensorView* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}

// TERNARY OPERATIONS

Val* threshold(Val* in, Val* thresh, Val* value) {
  NVF_CHECK(
      (thresh->getValType().value() == ValType::Others ||
       thresh->getValType().value() == ValType::NamedScalar) &&
          (value->getValType().value() == ValType::Others ||
           value->getValType().value() == ValType::NamedScalar),
      "For Threshold operation: Thresh and Value values should be Scalars.");

  thresh = optionalCast(in->getDataType().value(), thresh);
  value = optionalCast(in->getDataType().value(), value);
  Val* out = ops::newValLike(in, in->getDataType().value());

  IrBuilder::create<TernaryOp>(
      TernaryOpType::Threshold, out, in, thresh, value);
  return out;
}

TensorView* threshold(TensorView* in, Val* thresh, Val* value) {
  return threshold(in->as<Val>(), thresh, value)->as<TensorView>();
}

Val* clamp(Val* in, Val* min_val, Val* max_val) {
  NVF_CHECK(
      (min_val == nullptr || min_val->getValType().value() == ValType::Others ||
       min_val->getValType().value() == ValType::NamedScalar) &&
          (max_val == nullptr ||
           max_val->getValType().value() == ValType::Others ||
           max_val->getValType().value() == ValType::NamedScalar),
      "For Clamp operation: Min and Max values should be Scalars.");

  min_val = (min_val == nullptr)
      ? ops::getMinimumValue(in->getDataType().value())
      : optionalCast(in->getDataType().value(), min_val);
  NVF_CHECK(min_val != nullptr, "Missing minimum value");

  max_val = (max_val == nullptr)
      ? ops::getMaximumValue(in->getDataType().value())
      : optionalCast(in->getDataType().value(), max_val);
  NVF_CHECK(max_val != nullptr, "Missing maximum value");

  Val* out = ops::newValLike(in, in->getDataType().value());
  IrBuilder::create<TernaryOp>(TernaryOpType::Clamp, out, in, min_val, max_val);
  return out;
}

TensorView* clamp(TensorView* in, Val* min_val, Val* max_val) {
  return clamp(in->as<Val>(), min_val, max_val)->as<TensorView>();
}

// sum_to operator

TensorView* sum_to(TensorView* in, const std::vector<Val*>& sum_to_size) {
  const auto& root = TensorDomain::noReductions(in->getMaybeRFactorDomain());

  NVF_CHECK(
      root.size() >= sum_to_size.size(),
      "sum_to: Error trying to reduce",
      in,
      "into a shape of size",
      sum_to_size.size());

  // If no reduction is needed sum_to returns the input tv
  TensorView* out = in;

  const auto leading_dims = root.size() - sum_to_size.size();

  // Generate reduction axes for leading dims
  std::vector<int> reduce_dims(leading_dims);
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);

  // Generate reduction axes for dims within sum_to_size
  std::vector<bool> inner_red_dims(sum_to_size.size(), false);
  bool reduction_within_shape = false;

  // Reduce rest of the dims with keep_dim
  for (const auto i : c10::irange(leading_dims, root.size())) {
    if (sum_to_size[i - leading_dims]->isOneInt() &&
        !root[i]->extent()->isOneInt()) {
      inner_red_dims[i - leading_dims] = true;
      reduce_dims.push_back((int)i);
      reduction_within_shape = true;
    }
  }

  // Reduction step
  if (!reduce_dims.empty()) {
    out = sum(in, reduce_dims);
  }

  // Broadcast back reduced dims within shape
  if (reduction_within_shape) {
    out = broadcast(out, inner_red_dims);
  }

  return out;
}

TensorView* sum_to(TensorView* in, const std::vector<int64_t>& sum_to_size) {
  const auto& root = TensorDomain::noReductions(in->getMaybeRFactorDomain());

  NVF_CHECK(
      root.size() >= sum_to_size.size(),
      "sum_to: Error trying to reduce",
      in,
      "into a shape of size",
      sum_to_size.size());

  // If no reduction is needed sum_to returns the input tv
  TensorView* out = in;

  const auto leading_dims = root.size() - sum_to_size.size();

  // Generate reduction axes for leading dims
  std::vector<int> reduce_dims(leading_dims);
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);

  // Generate reduction axes for dims within sum_to_size
  std::vector<bool> inner_red_dims(sum_to_size.size(), false);
  bool reduction_within_shape = false;

  // Reduce rest of the dims with keep_dim
  for (const auto i : c10::irange(leading_dims, root.size())) {
    if (sum_to_size[i - leading_dims] == 1 && !root[i]->extent()->isOneInt()) {
      inner_red_dims[i - leading_dims] = true;
      reduce_dims.push_back((int)i);
      reduction_within_shape = true;
    }
  }

  // Reduction step
  if (!reduce_dims.empty()) {
    out = sum(in, reduce_dims);
  }

  // Broadcast back reduced dims within shape
  if (reduction_within_shape) {
    out = broadcast(out, inner_red_dims);
  }

  return out;
}

TensorView* shift(TensorView* inp, const std::vector<int>& offsets, bool pad) {
  // When pad is false, no padding is given. When it is true, padding
  // sizes are set so that output domains have the same extents as
  // input domains.
  std::vector<int> pad_width(offsets.size(), 0);
  if (pad) {
    for (const auto i : c10::irange(offsets.size())) {
      pad_width[i] = std::abs(offsets[i]);
    }
  }
  return shift(inp, offsets, pad_width);
}

TensorView* shift(
    TensorView* inp,
    const std::vector<int>& offsets,
    const std::vector<int>& pad_width_param) {
  auto inp_dom = TensorDomain::noReductions(inp->getRootDomain());
  const auto ndims = inp_dom.size();

  auto pad_width = pad_width_param;
  // Default padding is set so that the extent is kept unchanged
  if (pad_width.empty()) {
    pad_width = offsets;
    for (auto& p : pad_width) {
      p = std::abs(p);
    }
  }

  NVF_CHECK(
      ndims == offsets.size(),
      "Invalid shift offsets, number of entries in offsets expected to be ",
      ndims,
      " but received ",
      offsets.size());

  NVF_CHECK(
      ndims == pad_width.size(),
      "Invalid padding width list, number of entries in pad_width expected to be ",
      ndims,
      " but received ",
      pad_width.size());

  std::for_each(pad_width.begin(), pad_width.end(), [](const auto& pad) {
    NVF_CHECK(pad >= 0, "Padding width must be >= 0: ", pad);
  });

  TensorView* out = nullptr;

  std::vector<IterDomain*> out_dom;
  for (const auto i : c10::irange(ndims)) {
    const auto inp_axis = inp_dom[i];
    const auto offset = offsets[i];
    const auto pad = pad_width[i];

    if (offset == 0) {
      out_dom.push_back(inp_axis->cloneWithoutRFactor());
      continue;
    }

    Val* current_start_offset = dynamic_cast<Val*>(inp_axis->start());
    NVF_ERROR(
        current_start_offset != nullptr && current_start_offset->isConst(),
        "Invalid IterDomain start value:",
        current_start_offset);

    Val* current_stop_offset = dynamic_cast<Val*>(inp_axis->stopOffset());
    NVF_ERROR(
        current_stop_offset != nullptr && current_stop_offset->isConst(),
        "Invalid IterDomain stop offset value:",
        current_stop_offset);

    const auto cur_start_offset_value = current_start_offset->value();
    const auto cur_stop_offset_value = current_stop_offset->value();

    PolymorphicValue out_start_offset = 0L;
    PolymorphicValue out_stop_offset = 0L;

    if (offset > 0) {
      using namespace PolymorphicValue_functions;
      // shift to right; extent remains the same, start and stop
      // positions are moved right
      out_start_offset = cur_start_offset_value + offset - pad;
      out_stop_offset = max(cur_stop_offset_value - offset, int64_t(0));
      // If pad > offset, the extent of the output ID could be larger than the
      // input, and the start offset of the output domain could become
      // negative, which is not supported.
      NVF_CHECK(
          out_start_offset >= 0,
          "Invalid shift offset and padding. Padding must not be larger than the absolute extent of shift offset. Padding: ",
          pad,
          ". Shift: ",
          offset,
          ".");
    } else {
      using namespace PolymorphicValue_functions;
      // shift to left; extent remains the same, start and stop
      // positions are moved left
      out_start_offset = max(cur_start_offset_value + offset, int64_t(0));
      out_stop_offset = cur_stop_offset_value - offset - pad;
      // Similar to the above case whwere offset is positive, if pad >
      // -offset (note offset is negative), the extent of the output
      // ID could be larger than the input, and the stop offset of the
      // output domain could become negative.
      NVF_CHECK(
          out_stop_offset >= 0,
          "Invalid shift offset and padding. Padding must not be larger than the absolute extent of shift offset. Padding: ",
          pad,
          ". Shift: ",
          offset,
          ".");
    }

    out_dom.push_back(
        IterDomainBuilder(
            IrBuilder::create<Val>(out_start_offset, DataType::Index),
            inp_axis->extent())
            .stop_offset(
                IrBuilder::create<Val>(out_stop_offset, DataType::Index))
            .iter_type(inp_axis->getIterType())
            .build());
  }

  out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_dom, TensorDomain::getContiguityFilledWith(out_dom, true)),
      inp->getDataType().value());

  IrBuilder::create<ShiftOp>(out, inp, offsets, pad_width);
  return out;
}

namespace {

// Return a new TensorDomain with given root domains. Apply
// strides if necessary. With non-unit strides, strided domains become an
// rfactor domain.
TensorDomain* generateTensorDomainWithStrides(
    const std::vector<IterDomain*>& root_domains,
    const std::vector<int>& strides,
    bool skip_unit_stride) {
  std::vector<IterDomain*> strided_domains;

  // If strides are just unit strides, don't apply striding
  if (strides.empty() ||
      (skip_unit_stride &&
       std::all_of(
           strides.begin(), strides.end(), [](int s) { return s == 1; }))) {
    return IrBuilder::create<TensorDomain>(
        root_domains,
        TensorDomain::getContiguityFilledWith(root_domains, true));
  }

  for (const auto i : c10::irange(root_domains.size())) {
    auto root_dom = root_domains.at(i);

    if (i >= strides.size() || (skip_unit_stride && strides[i] == 1)) {
      strided_domains.push_back(root_dom);
      continue;
    }

    // Split the root domain by the stride
    auto split_out = root_dom->stridedSplit(strides[i]);
    strided_domains.push_back(split_out.first);
    strided_domains.push_back(split_out.second);
  }

  auto strided_td = IrBuilder::create<TensorDomain>(
      root_domains,
      strided_domains,
      strided_domains,
      TensorDomain::getContiguityFilledWith(strided_domains, true));

  return strided_td;
}

} // namespace

TensorView* gather(
    TensorView* inp,
    const std::vector<int>& window_shape,
    const std::vector<std::vector<int>>& pad_width,
    const std::vector<int>& strides,
    bool trim_out_of_bounds) {
  auto inp_dom = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  const auto ndims = inp_dom.size();

  NVF_CHECK(
      ndims == window_shape.size(),
      "Invalid window shape: number of entries expected to be ",
      ndims,
      " but received ",
      window_shape.size());

  std::for_each(window_shape.begin(), window_shape.end(), [](const auto& w) {
    NVF_CHECK(w > 0, "Window size must be > 0: ", w);
  });

  NVF_CHECK(
      ndims == pad_width.size(),
      "Invalid pad width: number of entries expected to be ",
      ndims,
      " but received ",
      pad_width.size());

  std::for_each(pad_width.begin(), pad_width.end(), [](const auto& p) {
    NVF_CHECK(
        p.size() == 2,
        "Each entry of pad_width must have two non-negative integers.");
    std::for_each(p.begin(), p.end(), [](const auto& p_left_or_right) {
      NVF_CHECK(
          p_left_or_right >= 0, "Padding must be >= 0: ", p_left_or_right);
    });
  });

  NVF_CHECK(
      strides.empty() || ndims == strides.size(),
      "Invalid strides: number of entries expected to be ",
      ndims,
      " but received ",
      strides.size());

  std::for_each(strides.begin(), strides.end(), [](const auto& s) {
    NVF_CHECK(s > 0, "Stride must be > 0: ", s);
  });

  std::vector<IterDomain*> out_root_domains;
  std::vector<IterDomain*> out_gather_dom;

  for (const auto i : c10::irange(ndims)) {
    const auto inp_axis = inp_dom[i];
    const auto window_dim = window_shape[i];
    const auto pad_left = pad_width[i][0];
    const auto pad_right = pad_width[i][1];
    // This may be over-conservative
    NVF_ERROR(inp_axis->start()->isZeroInt());
    NVF_ERROR(
        inp_axis->stopOffset()->isConstInt(),
        "Dynamic stop offset not supported: ",
        inp_axis);
    const auto inp_stop_offset = inp_axis->stopOffset()->evaluate();
    const auto extent_adjustment = window_dim - 1 - pad_left - pad_right;
    NVF_CHECK(
        extent_adjustment >= 0,
        "Invalid gather window and padding as output extent would be larger than input.",
        " Window: ",
        window_dim,
        ". Padding left: ",
        pad_left,
        ". Padding right: ",
        pad_right);
    const auto out_stop_offset = inp_stop_offset + extent_adjustment;
    out_root_domains.push_back(
        IterDomainBuilder(
            FusionGuard::getCurFusion()->zeroVal(), inp_axis->extent())
            .stop_offset(
                IrBuilder::create<Val>(out_stop_offset, DataType::Index))
            .iter_type(inp_axis->getIterType())
            .build());
    // create a new axis for the gathered domain
    out_gather_dom.push_back(
        IterDomainBuilder(
            FusionGuard::getCurFusion()->zeroVal(),
            IrBuilder::create<Val>((int64_t)window_dim, DataType::Index))
            .iter_type(IterType::Gather)
            .build());
  }

  out_root_domains.insert(
      out_root_domains.end(), out_gather_dom.begin(), out_gather_dom.end());

  TensorDomain* out_td = nullptr;

  if (trim_out_of_bounds) {
    // If no stride vector is given, just use stride 1. It does not do
    // any striding effect, but out-of-bounds values are trimmed.
    auto s = strides.empty() ? std::vector<int>(ndims, 1) : strides;
    out_td = generateTensorDomainWithStrides(out_root_domains, strides, false);
  } else {
    out_td = generateTensorDomainWithStrides(out_root_domains, strides, true);
  }

  auto out_tv =
      IrBuilder::create<TensorView>(out_td, inp->getDataType().value());

  IrBuilder::create<GatherOp>(out_tv, inp, window_shape, pad_width);
  return out_tv;
}

TensorView* viewAsScalar(TensorView* inp) {
  auto inp_type = inp->getDataType().value();
  auto vec_size = std::get<ArrayType>(inp_type.type).size;
  auto out_type = *std::get<ArrayType>(inp_type.type).type;

  std::vector<IterDomain*> out_domain;
  auto inp_domain = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  out_domain.reserve(inp_domain.size());
  for (auto d : inp_domain) {
    out_domain.push_back(d->cloneWithoutRFactor());
  }

  IterDomain* id =
      IterDomainBuilder(
          inp_domain[0]->container()->zeroVal(),
          IrBuilder::create<Val>((int64_t)vec_size, DataType::Index))
          .iter_type(IterType::VectorComponent)
          .build();
  out_domain.push_back(id);

  auto out = IrBuilder::create<TensorView>(
      inp->container(),
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      out_type);

  IrBuilder::create<ViewAsScalar>(inp->container(), out, inp, id);

  return out;
}

namespace {

//! Create new output for mma
static TensorView* newForMma(
    TensorView* tv_a,
    TensorView* tv_b,
    const std::vector<unsigned int>& axes,
    DataType data_type = DataType::Float) {
  auto orig_domain_a =
      TensorDomain::noReductions(tv_a->getMaybeRFactorDomain());
  auto orig_domain_b =
      TensorDomain::noReductions(tv_b->getMaybeRFactorDomain());

  NVF_ERROR(
      orig_domain_a.size() == orig_domain_b.size(),
      "MMA op: need matching dim input");

  std::set<unsigned int> axes_set(axes.begin(), axes.end());
  std::vector<IterDomain*> new_domain;

  NVF_ERROR(
      !axes_set.empty(),
      "Asked for output of reduction, but no reduction axis provided.");

  NVF_ERROR(
      (*(axes_set.rbegin())) < orig_domain_a.size(),
      "Error setting up reduction, reduction axis (",
      *(axes_set.rbegin()),
      ") is outside nDims (",
      orig_domain_a.size(),
      "). Keep in mind reductions are relative to root domains, not modified views.");

  auto axis_iter = axes_set.begin();
  for (const auto dim : c10::irange(orig_domain_a.size())) {
    bool isReduction = false;
    if (axis_iter != axes_set.end() && *axis_iter == dim) {
      isReduction = true;
      axis_iter++;
    }

    const IterDomain* id = orig_domain_a[dim]->isBroadcast()
        ? orig_domain_b[dim]
        : orig_domain_a[dim];

    NVF_CHECK(
        !(isReduction && id->isBroadcast() && !id->isImplicitBroadcast()),
        "Cannot reduce an axis that is marked as broadcasted as it has an undetermined size. Tried to reduce ID = ",
        id,
        " of tensor ",
        tv_a,
        "and",
        tv_b);

    new_domain.push_back(
        IterDomainBuilder(id->start(), id->extent())
            .stop_offset(id->stopOffset())
            .iter_type(isReduction ? IterType::Reduction : id->getIterType())
            .build());
  }

  TensorDomain* td = IrBuilder::create<TensorDomain>(
      new_domain, TensorDomain::getContiguityFilledWith(new_domain, true));

  return IrBuilder::create<TensorView>(td, data_type);
}

} // namespace

TensorView* fusedMultiplySum(
    TensorView* tv_a,
    TensorView* tv_b,
    const std::vector<int>& axes,
    Val* init) {
  // TODO:
  //  Validate axis relationships between a and b
  NVF_CHECK(tv_a->nDims() > 0, "Tried to reduce a 0-dim tensor");

  // TODO:
  //  Add tf32 and other mma data types
  //  Add fallback path for non-mma data types.
  NVF_CHECK(
      tv_a->getDataType().value() == DataType::Half ||
      tv_a->getDataType().value() == DataType::BFloat16);
  NVF_CHECK(tv_a->getDataType().value() == tv_b->getDataType().value());

  NVF_CHECK(!axes.empty(), "No reduction axis specified");

  // TODO:
  //  will lift this in a follow up when we have a
  //  more generic axes matching.
  NVF_CHECK(
      axes.size() == 1, "Single axis reduction only for mma op instantiation.")

  std::vector<unsigned int> uint_axes =
      ops::canonicalizeAxes(axes, tv_a->domain()->noReductions().size());

  TensorView* out = newForMma(tv_a, tv_b, uint_axes);

  if (init == nullptr) {
    init = IrBuilder::create<Val>(0.0, out->dtype());
  }

  // TODO:
  //  We will want to support initialize and rfactor with
  //  mma as well, for maybe fusing bias in prolog.
  NVF_CHECK(
      init->isConstScalar(),
      "Cannot create a reduction operation where the initial value is not a const scalar.");
  NVF_CHECK(
      init->dtype() == out->dtype(),
      "Init value dtype for fusedMultiplySum must match output.");

  IrBuilder::create<MmaOp>(out, tv_a, tv_b, init);

  return out;
}

TensorView* tensor(Val* val) {
  auto dtype = val->dtype();
  if (std::holds_alternative<PrimDataType>(dtype.type)) {
    // scalar tensor
    return full({}, val, dtype);
  }
  std::vector<int64_t> sizes;
  while (std::holds_alternative<ArrayType>(dtype.type)) {
    sizes.push_back((int64_t)std::get<ArrayType>(dtype.type).size);
    dtype = *std::get<ArrayType>(dtype.type).type;
  }
  NVF_ERROR(
      std::holds_alternative<PrimDataType>(dtype.type),
      "Expected an array of scalar or nested array of scalar");

  std::vector<IterDomain*> out_domain;
  out_domain.reserve(sizes.size());
  for (auto size : sizes) {
    IterDomain* id = IterDomainBuilder(
                         val->container()->zeroVal(),
                         IrBuilder::create<Val>(size, DataType::Index))
                         .build();
    out_domain.push_back(id);
  }

  auto out = IrBuilder::create<TensorView>(
      val->container(),
      IrBuilder::create<TensorDomain>(
          out_domain, TensorDomain::getContiguityFilledWith(out_domain, true)),
      dtype);

  IrBuilder::create<TensorConstruct>(val->container(), out, val);
  return out;
}

} // namespace nvfuser
