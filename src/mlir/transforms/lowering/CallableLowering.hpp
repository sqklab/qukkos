/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#pragma once
#include "quantum_to_llvm.hpp"

namespace qukkos {
class TupleUnpackOpLowering : public ConversionPattern {
protected:
public:
  explicit TupleUnpackOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::TupleUnpackOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class CreateCallableOpLowering : public ConversionPattern {
protected:
public:
  inline static const std::string qir_create_callable =
      "__quantum__rt__callable_create";
  explicit CreateCallableOpLowering(MLIRContext *context)
      : ConversionPattern(mlir::quantum::CreateCallableOp::getOperationName(),
                          1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qukkos