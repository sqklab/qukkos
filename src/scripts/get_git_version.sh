#!/bin/bash
git ls-remote https://github.com/sqkcloud/qukkos HEAD | awk '{ print $1}' | head -c 7