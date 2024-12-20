//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <XACC/Kokkos_XACC.hpp>
#include <XACC/Kokkos_XACC_Instance.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <impl/Kokkos_DeviceManagement.hpp>

#include <XACC.h>

#include <iostream>

// Arbitrary value to denote that we don't know yet what device to use.
int Kokkos::Experimental::Impl::XACCInternal::m_acc_device_num = -1;
int Kokkos::Experimental::Impl::XACCInternal::m_concurrency    = -1;

Kokkos::Experimental::Impl::XACCInternal&
Kokkos::Experimental::Impl::XACCInternal::singleton() {
  static XACCInternal self;
  return self;
}

bool Kokkos::Experimental::Impl::XACCInternal::verify_is_initialized(
    const char* const label) const {
  if (!m_is_initialized) {
    Kokkos::abort((std::string("Kokkos::Experimental::XACC::") + label +
                   " : ERROR device not initialized\n")
                      .c_str());
  }
  return m_is_initialized;
}

void Kokkos::Experimental::Impl::XACCInternal::initialize(int async_arg) {
  if ((async_arg < 0) && (async_arg != acc_async_sync) &&
      (async_arg != acc_async_noval)) {
    Kokkos::abort((std::string("Kokkos::Experimental::XACC::initialize()") +
                   " : ERROR async_arg should be a non-negative integer" +
                   " unless being a special value defined in XACC\n")
                      .c_str());
  }
  m_async_arg      = async_arg;
  m_is_initialized = true;
}

void Kokkos::Experimental::Impl::XACCInternal::finalize() {
  m_is_initialized = false;
}

bool Kokkos::Experimental::Impl::XACCInternal::is_initialized() const {
  return m_is_initialized;
}

void Kokkos::Experimental::Impl::XACCInternal::print_configuration(
    std::ostream& os, bool /*verbose*/) const {
  os << "Using XACC\n";  // FIXME_XACC
}

void Kokkos::Experimental::Impl::XACCInternal::fence(
    std::string const& name) const {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<
      Kokkos::Experimental::XACC>(
      name,
      Kokkos::Tools::Experimental::Impl::DirectFenceIDHandle{instance_id()},
      [&]() { acc_wait(m_async_arg); });
}

uint32_t Kokkos::Experimental::Impl::XACCInternal::instance_id()
    const noexcept {
  return Kokkos::Tools::Experimental::Impl::idForInstance<XACC>(
      reinterpret_cast<uintptr_t>(this));
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
int Kokkos::Experimental::XACC::concurrency() {
  return Impl::XACCInternal::m_concurrency;
}
#else
int Kokkos::Experimental::XACC::concurrency() const {
  return Impl::XACCInternal::m_concurrency;
}
#endif
