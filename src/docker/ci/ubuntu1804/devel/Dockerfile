from xacc/ubuntu:18.04
run git clone --recursive -b xacc-devel https://github.com/eclipse/xacc && cd xacc && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) install \
    && cd ../../ && git clone -b devel https://github.com/sqkcloud/qukkos && cd qukkos && mkdir build && cd build \
    && cmake .. -DXACC_DIR=~/.xacc -DQUKKOS_BUILD_TESTS=TRUE \
    && make -j$(nproc) install && ctest --output-on-failure
