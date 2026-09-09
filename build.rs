// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

fn main() {
    // Link the CUDA driver API (libcuda). The driver's PTX JIT
    // (libnvidia-ptxjitcompiler) is used at runtime; no nvcc/NVRTC is needed.
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rerun-if-changed=src/gpu/kernels.ptx");
    println!("cargo:rerun-if-changed=.git/HEAD");

    // Bake the git branch and commit into the binary for --version.
    let branch = {
        let b = git(&["branch", "--show-current"]);
        if b.is_empty() { "(detached)".to_string() } else { b }
    };
    let commit = git(&["rev-parse", "--short", "HEAD"]);
    println!("cargo:rustc-env=PLASMA_GIT_BRANCH={branch}");
    println!("cargo:rustc-env=PLASMA_GIT_COMMIT={commit}");
}

fn git(args: &[&str]) -> String {
    std::process::Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
