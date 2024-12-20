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

#include <utility>

namespace qukkos {
template <typename T>
qukkos_pimpl<T>::qukkos_pimpl() : m{new T{}} {}

template <typename T>
qukkos_pimpl<T>::qukkos_pimpl(const qukkos_pimpl<T>& other)
    : m(std::make_unique<T>(other)) {}

template <typename T>
template <typename... Args>
qukkos_pimpl<T>::qukkos_pimpl(Args&&... args)
    : m{new T{std::forward<Args>(args)...}} {}

template <typename T>
qukkos_pimpl<T>::~qukkos_pimpl() {}

template <typename T>
T* qukkos_pimpl<T>::operator->() {
  return m.get();
}
template <typename T>
T* qukkos_pimpl<T>::operator->() const {
  return m.get();
}

template <typename T>
T& qukkos_pimpl<T>::operator*() {
  return *m.get();
}

}  // namespace qukkos