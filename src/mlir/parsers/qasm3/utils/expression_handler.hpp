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

#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "qasm3BaseVisitor.h"
#include "qasm3Parser.h"
#include "qasm3_utils.hpp"
#include "symbol_table.hpp"

static constexpr double pi = 3.141592653589793238;
using namespace qasm3;

namespace qukkos {
class qasm3_expression_generator : public qasm3::qasm3BaseVisitor {
 protected:
  mlir::OpBuilder& builder;
  mlir::ModuleOp m_module;
  std::string file_name = "";

  std::string indexed_variable_name = "";
  
  bool builtin_math_func_treat_ints_as_float = false;
  bool casting_indexed_integer_to_bool = false;
  bool found_negation_unary_op = false;
  mlir::Value indexed_variable_value;

  mlir::Type internal_value_type;


  ScopedSymbolTable& symbol_table;

  void update_current_value(mlir::Value v);

  template <typename OpTy, typename... Args>
  OpTy createOp(Args... args) {
    OpTy value = builder.create<OpTy>(args...);
    update_current_value(value);
    return value;
  }

 public:
  mlir::Value current_value;
  mlir::Value last_current_value;

  qasm3_expression_generator(mlir::OpBuilder& b, ScopedSymbolTable& table,
                             std::string& fname);
  qasm3_expression_generator(mlir::OpBuilder& b, ScopedSymbolTable& table,
                             std::string& fname, mlir::Type t);

  antlrcpp::Any visitTerminal(antlr4::tree::TerminalNode* node) override;
  antlrcpp::Any visitExpression(qasm3Parser::ExpressionContext* ctx) override;

  // antlrcpp::Any visitIncrementor(qasm3Parser::IncrementorContext* ctx)
  // override;

  antlrcpp::Any visitExpressionTerminator(
      qasm3Parser::ExpressionTerminatorContext* ctx) override;

  antlrcpp::Any visitMultiplicativeExpression(
      qasm3Parser::MultiplicativeExpressionContext* ctx) override;
  antlrcpp::Any visitAdditiveExpression(
      qasm3Parser::AdditiveExpressionContext* ctx) override;
  antlrcpp::Any visitIncrementor(qasm3Parser::IncrementorContext* ctx) override;

  antlrcpp::Any visitXOrExpression(
      qasm3Parser::XOrExpressionContext* ctx) override;

  antlrcpp::Any visitBooleanExpression(
      qasm3Parser::BooleanExpressionContext* ctx) override;
  antlrcpp::Any visitComparsionExpression(
      qasm3Parser::ComparsionExpressionContext* ctx) override;
  antlrcpp::Any visitUnaryExpression(
      qasm3Parser::UnaryExpressionContext* ctx) override;
};
}  // namespace qukkos