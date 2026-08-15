// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

fn main() {
    // Link the CUDA driver API (libcuda). The driver's PTX JIT
    // (libnvidia-ptxjitcompiler) is used at runtime; no nvcc/NVRTC is needed.
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rerun-if-changed=src/gpu/kernels.ptx");
}
