add_subdirectory(print_llvm_qir)
add_subdirectory(xacc-ir-transformation)
#add_subdirectory(xacc-jit-pass)
install(FILES qukkos_base_llvm_pass.hpp DESTINATION include/qukkos)
