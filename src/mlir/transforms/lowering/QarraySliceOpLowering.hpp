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
class QarraySliceOpLowering : public ConversionPattern {
protected:
  // Constant string for runtime function name
  inline static const std::string qir_qubit_array_slice =
      "__quantum__rt__array_slice";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;

public:
  // Constructor, store seen variables
  explicit QarraySliceOpLowering(MLIRContext *context,
                                 std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::ArraySliceOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace qukkos