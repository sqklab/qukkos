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

#include <memory>

namespace qukkos {
template <typename T>
class qukkos_pimpl {
 private:
  std::unique_ptr<T> m;

 public:
  qukkos_pimpl();
  qukkos_pimpl(const qukkos_pimpl<T>&);
  template <typename... Args>
  qukkos_pimpl(Args&&...);
  ~qukkos_pimpl();
  T* operator->();
  T* operator->() const;
  T& operator*();
};
}