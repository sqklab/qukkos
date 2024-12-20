FROM qukkos/llvm-alpine as llvm_install
FROM xacc/alpine
COPY --from=llvm_install /usr/local/aideqc/llvm /usr/local/aideqc/llvm
RUN apk add libc6-compat ninja bash sudo curl && git clone https://github.com/sqkcloud/qukkos && cd qukkos && mkdir build && cd build \
   && cmake .. -G Ninja -DXACC_DIR=/usr/local/aideqc/qukkos -DCMAKE_INSTALL_PREFIX=/usr/local/aideqc/qukkos -DLLVM_ROOT=/usr/local/aideqc/llvm -DQUKKOS_EXTRA_COMPILER_FLAGS="-B /usr/lib/gcc/x86_64-alpine-linux-musl/10.3.1 -L /usr/lib/gcc/x86_64-alpine-linux-musl/10.3.1" -DQUKKOS_EXTRA_HEADERS="/usr/include/c++/10.3.1;/usr/include/c++/10.3.1/x86_64-alpine-linux-musl" \
   && cmake --build . --target install && cd ../.. && rm -rf qukkos \
   && adduser --gecos '' --disabled-password coder \
   && echo "coder ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd \
   && curl -fsSL "https://github.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz" | tar -C /usr/local/bin -xzf - \
   && chown root:root /usr/local/bin/fixuid \
   && chmod 4755 /usr/local/bin/fixuid \
   && mkdir -p /etc/fixuid \
   && printf "user: coder\ngroup: coder\n" > /etc/fixuid/config.yml \
   && rm -rf fixuid-0.5-linux* \
   && ln -s /lib/libc.musl-x86_64.so.1 /lib/ld-linux-x86-64.so.2
   
USER 1000
ENV USER=coder
WORKDIR /home/coder
RUN git clone https://github.com/sqkcloud/qukkos && cp -r qukkos/examples cpp-examples \
   && cp -r qukkos/python/examples py-examples && rm -rf qukkos 
ENV PYTHONPATH "${PYTHONPATH}:/usr/local/aideqc/qukkos"
ENV PATH "${PATH}:/usr/local/aideqc/qukkos/bin"
ENTRYPOINT [ "/bin/bash" ]