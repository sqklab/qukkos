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
#include "qasm3_utils.hpp"
#include "symbol_table.hpp"

#include <iostream>

#include "Quantum/QuantumOps.h"
#include "exprtk.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using symbol_table_t = exprtk::symbol_table<double>;
using expression_t = exprtk::expression<double>;
using parser_t = exprtk::parser<double>;

namespace qukkos {

void ScopedSymbolTable::replace_symbol(mlir::Value old_value,
                                       mlir::Value new_value) {
  auto key = replacement_helper[old_value.getAsOpaquePointer()];
  for (int i = current_scope; i >= 0; i--) { // again, nasty bug. MUST BE int NOT auto
    if (scoped_symbol_tables[i].find(key) != scoped_symbol_tables[i].end()) {
      scoped_symbol_tables[i].at(key) = new_value;
      replacement_helper[new_value.getAsOpaquePointer()] = key;
    }
  }
}

std::string ScopedSymbolTable::get_symbol_var_name(mlir::Value value) {
  auto iter = replacement_helper.find(value.getAsOpaquePointer());
  return iter != replacement_helper.end() ? iter->second : "";
}

std::optional<int64_t>
ScopedSymbolTable::try_evaluate_constant_integer_expression(
    const std::string expr_str) {
  auto all_constants = get_constant_integer_variables();
  std::vector<std::string> variable_names;
  std::vector<double> variable_values;
  for (auto [n, v] : all_constants) {
    variable_names.push_back(n);
    variable_values.push_back(v);
  }

  for (auto &[name, global_value] : global_constants) {
    if (global_constant_memref_types.count(name) &&
        global_constant_memref_types[name].isa<mlir::IntegerType>()) {
      variable_names.push_back(name);
      variable_values.push_back((int64_t)global_value);
    }
  }

  double ref = 0.0;

  symbol_table_t exprtk_symbol_table;
  exprtk_symbol_table.add_constants();
  for (int i = 0; i < variable_names.size(); i++) {
    exprtk_symbol_table.add_variable(variable_names[i], variable_values[i]);
  }

  expression_t expr;
  expr.register_symbol_table(exprtk_symbol_table);
  parser_t parser;
  if (parser.compile(expr_str, expr)) {
    ref = expr.value();
  } else {
    // Cannot eval to a const int.
    return std::nullopt;
  }

  return (int64_t)ref;
}

int64_t ScopedSymbolTable::evaluate_constant_integer_expression(
    const std::string expr_str) {
  const std::optional<int64_t> try_eval =
      try_evaluate_constant_integer_expression(expr_str);
  double ref = 0.0;
  if (try_eval.has_value()) {
    ref = try_eval.value();
  } else {
    printErrorMessage("Failed to evaluate constant integer expression: " +
                      expr_str + ". Must be a constant integer type.");
  }

  return (int64_t)ref;
}

void ScopedSymbolTable::evaluate_const_global(const std::string variable_name,
                                              const std::string expr_str,
                                              mlir::Type type,
                                              mlir::Block& module_block,
                                              mlir::Location location) {
  // Global Const values will be created as global_memrefs
  // These can be constant ints or floats, type will tell us
  // They can be constructed from arithmetic operations on previous
  // global const values / global_memrefs. They need to
  // be stored so that future get_symbol calls will use the getGlobalMemref op.

  std::vector<std::string> variable_names;
  std::vector<double> variable_values;

  for (auto& [name, global_value] : global_constants) {
    variable_names.push_back(name);
    variable_values.push_back(global_value);
  }

  double ref = 0.0;

  symbol_table_t exprtk_symbol_table;
  exprtk_symbol_table.add_constants();
  for (int i = 0; i < variable_names.size(); i++) {
    exprtk_symbol_table.add_variable(variable_names[i], variable_values[i]);
  }

  expression_t expr;
  expr.register_symbol_table(exprtk_symbol_table);
  parser_t parser;
  if (parser.compile(expr_str, expr)) {
    ref = expr.value();
  } else {
    printErrorMessage("Failed to evaluate constant integer expression: " +
                      expr_str + ". Must be a constant integer type.");
  }

  // Now create the Global Memref Op
  // Note: need to use Tensor type to init the GlobalMemrefOp;
  llvm::ArrayRef<int64_t> shaperef{};
  mlir::DenseElementsAttr initial_attr;
  if (type.isa<mlir::IntegerType>()) {
    initial_attr = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get(shaperef, type), {(int64_t)ref});
  } else {
    initial_attr = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get(shaperef, type), {ref});
  }

  auto memref_type = mlir::MemRefType::get(shaperef, type);
  auto savept = builder->saveInsertionPoint();
  builder->setInsertionPointToStart(&module_block);
  auto x = builder->create<mlir::GlobalMemrefOp>(
      location, variable_name, builder->getStringAttr("private"),
      mlir::TypeAttr::get(memref_type), initial_attr, true);
  builder->restoreInsertionPoint(savept);

  global_constant_memref_types.insert({variable_name, type});
  global_constants.insert({variable_name, ref});

  // add_symbol(variable_name, x, {"const"});
  return;
}

void ScopedSymbolTable::add_measure_bit_assignment(
    const mlir::Value &bit_var, const mlir::Value &result_var) {
  assert(result_var.getType().isa<mlir::OpaqueType>());
  bit_var_ptr_to_meas_result_var[bit_var.getAsOpaquePointer()] = result_var;
}

std::optional<mlir::Value>
ScopedSymbolTable::try_lookup_meas_result(const mlir::Value &bit_var) {
  const auto iter =
      bit_var_ptr_to_meas_result_var.find(bit_var.getAsOpaquePointer());
  if (iter != bit_var_ptr_to_meas_result_var.end()) {
    return iter->second;
  }
  return std::nullopt;
}

std::optional<mlir::Value>
ScopedSymbolTable::try_lookup_meas_result(const std::string &bit_var_name) {
  if (!has_symbol(bit_var_name)) {
    return std::nullopt;
  }
  return try_lookup_meas_result(get_symbol(bit_var_name));
}

std::unordered_map<std::string, mlir::Value>
ScopedSymbolTable::get_all_visible_symbols() {
  std::unordered_map<std::string, mlir::Value> all_symbols;
  for (int i = current_scope; i >= 0; i--) {
    for (auto &[k, v] : scoped_symbol_tables[i]) {
      // Don't override if seeing duplicated variables.
      if (all_symbols.find(k) == all_symbols.end()) {
        all_symbols[k] = v;
      }
    }
  }
  return all_symbols;
}

std::optional<size_t> ScopedSymbolTable::get_qreg_size(const std::string &qreg_name) {
  if (!has_symbol(qreg_name)) {
    return std::nullopt;
  }
  auto qreg_value = get_symbol(qreg_name);
  if (qreg_value.getType().isa<mlir::OpaqueType>() &&
      qreg_value.getType().cast<mlir::OpaqueType>().getTypeData() == "Array") {
    // Should be Array type:
    if (auto op = qreg_value.getDefiningOp<mlir::quantum::QallocOp>()) {
      return op.size().getLimitedValue();
    } else {
      auto attributes = get_variable_attributes(qreg_name);
      if (!attributes.empty()) {
        try {
          return std::stoi(attributes[0]);
        } catch (...) {
          return std::nullopt;
        }
      }
    }
  }
  return std::nullopt;
}

void ScopedSymbolTable::invalidate_qubit_extracts(
    const std::string &qreg_name, const std::vector<int> &indices) {
  if (!get_qreg_size(qreg_name).has_value()) {
    return;
  }
  if (indices.empty()) {
    const auto reg_size = get_qreg_size(qreg_name).value();
    for (size_t i = 0; i < reg_size; ++i) {
      erase_symbol(array_qubit_symbol_name(qreg_name, i));
    }
  } else {
    const auto reg_size = get_qreg_size(qreg_name).value();
    for (const auto &idx : indices) {
      assert(idx < reg_size);
      erase_symbol(array_qubit_symbol_name(qreg_name, idx));
    }
  }
}
void ScopedSymbolTable::erase_symbol(const std::string &var_name) {
  for (auto &table : scoped_symbol_tables) {
    table.erase_symbol(var_name);
  }
}

bool ScopedSymbolTable::verify_qubit_ssa_dominance_property(
    mlir::Value qubit, mlir::Block* current_block) {
  assert(qubit.getType().isa<mlir::OpaqueType>() &&
         qubit.getType().dyn_cast<mlir::OpaqueType>().getTypeData() == "Qubit");

  // Checking that the QVS Op that **produces** this qubit is in the same
  // region as this op.
  // i.e., if the Op that produces this qubit SSA is in a for or if region,
  // this qubit SSA is **not** properly dominated. Hence, requires a re-extract. 
  if (auto *useOp = qubit.getDefiningOp()) {
    if (mlir::dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(useOp)) {
      mlir::Block *block2 = useOp->getBlock();
      mlir::Region *region1 = current_block->getParent();
      mlir::Region *region2 = block2->getParent();
      return region1 == region2;
    }
  }

  return true;
}
} // namespace qukkos