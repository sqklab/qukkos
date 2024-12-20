
# Store the location of the clang executable
set (CLANG_EXECUTABLE "${LLVM_INSTALL_PREFIX}/bin/clang++")
set(QUKKOS_APPEND_PLUGIN_PATH "")
if (NOT ${XACC_ROOT} MATCHES ${CMAKE_INSTALL_PREFIX}) 
  set (QUKKOS_APPEND_PLUGIN_PATH "${CMAKE_INSTALL_PREFIX}/plugins")
endif()

configure_file(qukkos.in
               ${CMAKE_BINARY_DIR}/qukkos)

install(PROGRAMS ${CMAKE_BINARY_DIR}/qukkos DESTINATION bin)
