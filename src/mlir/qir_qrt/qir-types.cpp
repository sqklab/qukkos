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
#include "qir-types.hpp"

namespace qukkos {
namespace internal {

AllocationTracker *AllocationTracker::m_globalTracker = nullptr;

AllocationTracker &AllocationTracker::get() {
  if (!m_globalTracker) {
    m_globalTracker = new AllocationTracker();
  }
  return *m_globalTracker;
}
} // namespace internal
} // namespace qukkos