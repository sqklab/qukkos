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

#ifndef KOKKOS_XACC_MACROS_HPP
#define KOKKOS_XACC_MACROS_HPP

#define KOKKOS_IMPL_ACC_PRAGMA_HELPER(x) _Pragma(#x)
#define KOKKOS_IMPL_ACC_PRAGMA(x) KOKKOS_IMPL_ACC_PRAGMA_HELPER(acc x)

#endif
