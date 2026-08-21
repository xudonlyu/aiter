#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Rebuild dsv4_mla_prefill_opus.co.
#
#   ./build_co.sh                        # source taken from the branch below
#   SRC_DIR=/path/to/csrc/include ./build_co.sh
#
# The kernel source is NOT vendored next to the code object: it lives on the
# branch that owns it, and a second copy here would be one more thing to keep
# in step.  This object was built from exactly:
#
#   pa_sparse_prefill_opus.h  <-  wenwzhan/dsv4-opus-fp8-paged
#
# The flags below are aiter's own JIT flags for module_pa_sparse_prefill_opus,
# minus the torch include paths the device TU does not use.  Two of them are
# load-bearing and neither is obvious:
#
#   -mllvm -enable-post-misched=1        aiter's generated build.ninja carries
#                                        =0 and then =1; the later wins and the
#                                        kernel is 5-9% slower without it.
#   -mllvm -amdgpu-early-inline-all=false  the ninja *rule* appends this after
#                                        cuda_cflags' =true.  Reading only
#                                        cuda_cflags gets you =true and a
#                                        different register allocation.
#
# Verify a rebuild rather than trusting it -- compare the disassembly against a
# JIT-built object:
#   llvm-objdump -d --mcpu=gfx950 dsv4_mla_prefill_opus.co
# Two consecutive builds from the same checkout are byte-identical.  Across two
# checkouts at different paths the instruction stream is still identical but
# the .dynstr is not -- __hip_cuid_ follows the paths -- so compare the
# disassembly, not md5.
set -euo pipefail

ARCH=${ARCH:-gfx950}
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
BUNDLER=${BUNDLER:-/opt/rocm/llvm/bin/clang-offload-bundler}
NM=${NM:-/opt/rocm/llvm/bin/llvm-nm}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
INC=${SRC_DIR:-$ROOT/csrc/include}
# A fixed build directory, not mktemp: hipcc derives __hip_cuid_ from the
# translation unit's path, so a moving path breaks byte-reproducibility.
WORK="$HERE/.build"
mkdir -p "$WORK"

cat > "$WORK/dsv4_mla_prefill_opus_device.cu" <<'EOF'
// Device-only translation unit.  The kernels are templates, so only the one
// explicitly instantiated here reaches the code object -- the flat variants in
// the same header do not.
#include <hip/hip_runtime.h>
#define PA_SPARSE_PREFILL_OPUS_IMPL
#include "pa_sparse_prefill_opus.h"

template __global__ void pa_prefill_16mx8_32nx1_fp8_paged_kernel<
    pa_16mx8_32nx1_fp8_paged_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>>(
    pa_fp8_paged_kargs);
EOF

FLAGS=(-O3 -std=c++20 "--offload-arch=$ARCH"
       -DWITH_HIP -D_GLIBCXX_USE_CXX11_ABI=1 -DENABLE_CK=1 -DENABLE_ROPE_POSITIONS_INT32=0
       -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -DUSE_ROCM=1 -DHIPBLAS_V2
       -DCUDA_HAS_FP16=1 -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1
       -DLEGACY_HIPBLAS_DIRECT -DUSE_PROF_API=1
       -U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__
       -ffast-math -fgpu-flush-denormals-to-zero -fno-offload-uniform-block -fno-gpu-rdc
       -mcmodel=large -fno-unique-section-names -ffunction-sections -fdata-sections
       -fvisibility=hidden -fvisibility-inlines-hidden -fPIC
       -I"$INC"
       -mllvm --amdgpu-kernarg-preload-count=32 -mllvm --lsr-drop-solution=1
       -mllvm -amdgpu-early-inline-all=false -mllvm -amdgpu-function-calls=false
       -mllvm -enable-post-misched=1)

echo "building dsv4_mla_prefill_opus.co"
# Compile from inside $WORK with a relative source name: hipcc derives
# __hip_cuid_ from the translation unit's path, and a bare filename keeps one
# source of variance out of it.
( cd "$WORK" && "$HIPCC" "${FLAGS[@]}" -c -x hip dsv4_mla_prefill_opus_device.cu -o opus.o )

# The fat binary hipcc emits is not loadable: hipModuleLoad wants a bare ELF
# code object, the same form the assembly kernels ship in.  `file` on a bundled
# object says "data", it builds fine, and it fails at load time.
/opt/rocm/llvm/bin/llvm-objcopy --dump-section=.hip_fatbin="$WORK/opus.fat" "$WORK/opus.o" /dev/null
"$BUNDLER" --unbundle --type=o --targets="hipv4-amdgcn-amd-amdhsa--$ARCH" \
    --input="$WORK/opus.fat" --output="$HERE/dsv4_mla_prefill_opus.co"

file -b "$HERE/dsv4_mla_prefill_opus.co" | grep -q '^ELF' \
    || { echo "ERROR: not a bare ELF code object"; exit 1; }

n=$("$NM" "$HERE/dsv4_mla_prefill_opus.co" | awk '$2=="T"' | wc -l)
[ "$n" = "1" ] || { echo "ERROR: expected 1 kernel symbol, found $n"; exit 1; }
echo "kernel symbol:"
"$NM" "$HERE/dsv4_mla_prefill_opus.co" | awk '$2=="T"{print "    " $3}'
