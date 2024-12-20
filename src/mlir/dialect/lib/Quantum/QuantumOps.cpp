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
#include "Quantum/QuantumOps.h"
#include "Quantum/QuantumDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
bool isOpaqueTypeWithName(mlir::Type type, std::string dialect,
                          std::string type_name) {
  if (type.isa<mlir::OpaqueType>() && dialect == "quantum") {
    if (type_name == "Qubit" || type_name == "Result" || type_name == "Array" ||
        type_name == "ArgvType" || type_name == "QregType" ||
        type_name == "StringType" || type_name == "Tuple" ||
        type_name == "Callable") {
      return true;
    }
  }

  return false;
}

#define GET_OP_CLASSES
#include "Quantum/QuantumOps.cpp.inc"


