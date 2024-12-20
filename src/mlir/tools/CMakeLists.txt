add_llvm_executable(qukkos-mlir-tool qukkos-mlir-tool.cpp)

target_compile_options(qukkos-mlir-tool PUBLIC "-fexceptions")

target_compile_features(qukkos-mlir-tool
                        PUBLIC
                        cxx_std_17)
llvm_update_compile_flags(qukkos-mlir-tool)
target_link_libraries(qukkos-mlir-tool PUBLIC MLIRQuantum quantum-to-llvm-lowering openqasm-mlir-generator openqasmv3-mlir-generator)

if(APPLE)
  set_target_properties(qukkos-mlir-tool
                        PROPERTIES INSTALL_RPATH "${MLIR_INSTALL_DIR}/lib;${CMAKE_INSTALL_PREFIX}/lib")
else()
set_target_properties(qukkos-mlir-tool
                        PROPERTIES INSTALL_RPATH "${MLIR_INSTALL_DIR}/lib:${CMAKE_INSTALL_PREFIX}/lib")
endif()

#set_target_properties(qukkos-mlir-tool
#                        PROPERTIES INSTALL_RPATH "${MLIR_INSTALL_DIR}/lib:${CMAKE_BINARY_DIR}/mlir/parsers/openqasm:${CMAKE_BINARY_DIR}/lib")
install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/qukkos-mlir-tool DESTINATION bin)
