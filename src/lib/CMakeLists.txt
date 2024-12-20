add_subdirectory(quasimo)
add_subdirectory(mirror_rb)

# Install QUKKOS standard library header
file(GLOB LIB_HEADERS qukkos_*)
install(FILES ${LIB_HEADERS} DESTINATION include/qukkos)
file(GLOB IMPL_HEADERS impl/*.hpp)
install(FILES ${IMPL_HEADERS} DESTINATION include/qukkos/impl)
