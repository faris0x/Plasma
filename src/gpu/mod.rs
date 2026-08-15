// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

//! Host-side GPU backend. Loads the hand-written PTX kernel library through
//! the CUDA driver API, drives it from the Plasma IR, and samples host-side
//! (determinism: every random decision uses the host RNG).

use std::cell::Cell;
use std::ffi::c_void;
use std::time::Instant;

use plasma::ir::{IrOp, Program};
use plasma::rng::Lcg64;

// ─── CUDA driver API bindings (raw FFI, no dependencies) ─────────

mod ffi {
    use std::ffi::c_char;
    use std::ffi::c_void;

    #[repr(C)]
    pub struct Ctx {
        _p: [u8; 0],
    }

    pub const CUDA_SUCCESS: i32 = 0;

    extern "C" {
        pub fn cuInit(flags: u32) -> i32;
        pub fn cuDeviceGet(dev: *mut i32, ordinal: i32) -> i32;
        pub fn cuCtxCreate(ctx: *mut *mut Ctx, flags: u32, dev: i32) -> i32;
        pub fn cuCtxDestroy(ctx: *mut Ctx) -> i32;
        pub fn cuModuleLoadData(module: *mut *mut (), image: *const u8) -> i32;
        pub fn cuModuleUnload(module: *mut ()) -> i32;
        pub fn cuModuleGetFunction(
            func: *mut *mut (),
            module: *mut (),
            name: *const c_char,
        ) -> i32;
        pub fn cuMemAlloc(ptr: *mut u64, size: u64) -> i32;
        pub fn cuMemFree(ptr: u64) -> i32;
        pub fn cuMemcpyHtoD(dst: u64, src: *const c_void, count: u64) -> i32;
        pub fn cuMemcpyDtoH(dst: *mut c_void, src: u64, count: u64) -> i32;
        pub fn cuEventCreate(event: *mut *mut (), flags: u32) -> i32;
        pub fn cuEventRecord(event: *mut (), stream: *mut ()) -> i32;
        pub fn cuEventSynchronize(event: *mut ()) -> i32;
        pub fn cuEventElapsedTime(ms: *mut f32, start: *mut (), end: *mut ()) -> i32;
        pub fn cuEventDestroy(event: *mut ()) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn cuLaunchKernel(
            func: *mut (),
            grid_x: u32,
            grid_y: u32,
            grid_z: u32,
            block_x: u32,
            block_y: u32,
            block_z: u32,
            shared: u32,
            stream: *mut (),
            kernel_params: *const *const c_void,
            extra: *const *const c_void,
        ) -> i32;
    }
}

fn cu_check(rc: i32, what: &str) -> Result<(), String> {
    if rc == ffi::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("CUDA error 0x{rc:x} at {what}"))
    }
}

// Kernel argument: a u32 scalar or a u64 device pointer, passed by address.
trait KernelArg {
    fn as_ptr(&self) -> *const c_void;
}
impl KernelArg for u32 {
    fn as_ptr(&self) -> *const c_void {
        self as *const u32 as *const c_void
    }
}
impl KernelArg for u64 {
    fn as_ptr(&self) -> *const c_void {
        self as *const u64 as *const c_void
    }
}
impl KernelArg for f32 {
    fn as_ptr(&self) -> *const c_void {
        self as *const f32 as *const c_void
    }
}

const BLOCK: u32 = 256;

// ─── GPU backend ──────────────────────────────────────────────────

pub struct Gpu {
    ctx: *mut ffi::Ctx,
    module: *mut (),
    f_reset: *mut (),
    f_h: *mut (),
    f_x: *mut (),
    f_cnot: *mut (),
    f_toff: *mut (),
    f_prob: *mut (),
    f_collapse: *mut (),
    state_buf: u64,
    prob_buf: u64,
    n: usize, // number of amplitudes (2^num_qubits)
    ev_start: *mut (),
    ev_stop: *mut (),
    /// Microseconds spent inside the IR executor (GPU kernel launches).
    pub exec_us: Cell<u128>,
    /// Microseconds of true GPU execution, measured with CUDA events.
    pub exec_gpu_us: Cell<u128>,
    /// Microseconds spent copying the state device->host.
    pub transfer_us: Cell<u128>,
    /// Microseconds spent sampling host-side.
    pub sample_us: Cell<u128>,
    /// Number of GPU kernel launches performed.
    pub launches: Cell<u64>,
    /// Init stage breakdown (microseconds): driver init, context, PTX JIT, alloc.
    pub init_cuinit_us: Cell<u128>,
    pub init_ctx_us: Cell<u128>,
    pub init_jit_us: Cell<u128>,
    pub init_alloc_us: Cell<u128>,
}

impl Gpu {
    /// Initialise the driver, load the PTX module, allocate buffers.
    pub fn init(num_qubits: u8) -> Result<Self, String> {
        let n = 1usize << num_qubits;
        let ptx: &[u8] = include_bytes!("kernels.ptx");

        unsafe {
            let t0 = Instant::now();
            cu_check(ffi::cuInit(0), "cuInit")?;
            let init_cuinit_us = t0.elapsed().as_micros();

            let t0 = Instant::now();
            let mut dev: i32 = 0;
            cu_check(ffi::cuDeviceGet(&mut dev, 0), "cuDeviceGet")?;
            let mut ctx: *mut ffi::Ctx = std::ptr::null_mut();
            cu_check(ffi::cuCtxCreate(&mut ctx, 0, dev), "cuCtxCreate")?;
            let init_ctx_us = t0.elapsed().as_micros();

            let t0 = Instant::now();
            let mut module: *mut () = std::ptr::null_mut();
            // cuModuleLoadData expects a null-terminated image for PTX.
            let mut image: Vec<u8> = Vec::with_capacity(ptx.len() + 1);
            image.extend_from_slice(ptx);
            image.push(0);
            cu_check(ffi::cuModuleLoadData(&mut module, image.as_ptr()), "cuModuleLoadData")?;
            let init_jit_us = t0.elapsed().as_micros();

            let t0 = Instant::now();
            let get = |name: &str| -> Result<*mut (), String> {
                let mut f: *mut () = std::ptr::null_mut();
                let cname = std::ffi::CString::new(name).unwrap();
                cu_check(ffi::cuModuleGetFunction(&mut f, module, cname.as_ptr()), name)?;
                Ok(f)
            };

            let f_reset = get("reset")?;
            let f_h = get("apply_h")?;
            let f_x = get("apply_x")?;
            let f_cnot = get("apply_cnot")?;
            let f_toff = get("apply_toff")?;
            let f_prob = get("prob")?;
            let f_collapse = get("collapse")?;

            let mut state_buf: u64 = 0;
            cu_check(
                ffi::cuMemAlloc(&mut state_buf, (8 * n) as u64),
                "cuMemAlloc state",
            )?;
            let mut prob_buf: u64 = 0;
            cu_check(ffi::cuMemAlloc(&mut prob_buf, 4), "cuMemAlloc prob")?;
            let init_alloc_us = t0.elapsed().as_micros();

            // CUDA events for true GPU execution timing.
            let mut ev_start: *mut () = std::ptr::null_mut();
            let mut ev_stop: *mut () = std::ptr::null_mut();
            cu_check(ffi::cuEventCreate(&mut ev_start, 0), "cuEventCreate start")?;
            cu_check(ffi::cuEventCreate(&mut ev_stop, 0), "cuEventCreate stop")?;

            Ok(Gpu {
                ctx,
                module,
                f_reset,
                f_h,
                f_x,
                f_cnot,
                f_toff,
                f_prob,
                f_collapse,
                state_buf,
                prob_buf,
                n,
                ev_start,
                ev_stop,
                exec_us: Cell::new(0),
                exec_gpu_us: Cell::new(0),
                transfer_us: Cell::new(0),
                sample_us: Cell::new(0),
                launches: Cell::new(0),
                init_cuinit_us: Cell::new(init_cuinit_us),
                init_ctx_us: Cell::new(init_ctx_us),
                init_jit_us: Cell::new(init_jit_us),
                init_alloc_us: Cell::new(init_alloc_us),
            })
        }
    }

    fn launch(&self, func: *mut (), args: &[&dyn KernelArg]) -> Result<(), String> {
        let grid = ((self.n as u32 + BLOCK - 1) / BLOCK).max(1);
        self.launches.set(self.launches.get() + 1);
        unsafe {
            let params: Vec<*const c_void> = args.iter().map(|a| a.as_ptr()).collect();
            cu_check(
                ffi::cuLaunchKernel(
                    func,
                    grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_ptr(),
                    std::ptr::null(),
                ),
                "cuLaunchKernel",
            )
        }
    }

    fn reset(&self) -> Result<(), String> {
        let n = self.n as u32;
        self.launch(self.f_reset, &[&self.state_buf, &n])
    }

    fn apply_h(&self, q: u8) -> Result<(), String> {
        let n = self.n as u32;
        let q = q as u32;
        self.launch(self.f_h, &[&self.state_buf, &n, &q])
    }

    fn apply_x(&self, q: u8) -> Result<(), String> {
        let n = self.n as u32;
        let q = q as u32;
        self.launch(self.f_x, &[&self.state_buf, &n, &q])
    }

    fn apply_cnot(&self, c: u8, t: u8) -> Result<(), String> {
        let n = self.n as u32;
        let c = c as u32;
        let t = t as u32;
        self.launch(self.f_cnot, &[&self.state_buf, &n, &c, &t])
    }

    fn apply_toff(&self, c1: u8, c2: u8, t: u8) -> Result<(), String> {
        let n = self.n as u32;
        let c1 = c1 as u32;
        let c2 = c2 as u32;
        let t = t as u32;
        self.launch(self.f_toff, &[&self.state_buf, &n, &c1, &c2, &t])
    }

    /// Return P(qubit q = 1) by launching the prob reduction kernel.
    fn measure_prob(&self, q: u8) -> Result<f32, String> {
        let n = self.n as u32;
        let q = q as u32;
        // Zero the prob buffer before accumulating.
        unsafe {
            let zero = 0.0f32;
            cu_check(
                ffi::cuMemcpyHtoD(self.prob_buf, (&zero as *const f32).cast(), 4),
                "cuMemcpyHtoD prob zero",
            )?;
        }
        self.launch(self.f_prob, &[&self.state_buf, &n, &q, &self.prob_buf])?;
        let mut out = 0.0f32;
        unsafe {
            cu_check(
                ffi::cuMemcpyDtoH((&mut out as *mut f32).cast(), self.prob_buf, 4),
                "cuMemcpyDtoH prob",
            )?;
        }
        Ok(out)
    }

    fn collapse(&self, q: u8, outcome: u32, inv_norm: f32) -> Result<(), String> {
        let n = self.n as u32;
        let q = q as u32;
        self.launch(
            self.f_collapse,
            &[&self.state_buf, &n, &q, &outcome, &inv_norm],
        )
    }

    /// Transfer the final state to the host and draw one sample.
    fn sample_host(&self, rng: &mut Lcg64) -> Result<usize, String> {
        let t0 = Instant::now();
        let mut buf = vec![0.0f32; 2 * self.n];
        unsafe {
            cu_check(
                ffi::cuMemcpyDtoH(buf.as_mut_ptr().cast(), self.state_buf, (8 * self.n) as u64),
                "cuMemcpyDtoH state",
            )?;
        }
        self.transfer_us.set(self.transfer_us.get() + t0.elapsed().as_micros());
        let t1 = Instant::now();
        let r = rng.next_f64();
        let mut cum = 0.0f64;
        for i in 0..self.n {
            let re = buf[2 * i] as f64;
            let im = buf[2 * i + 1] as f64;
            cum += re * re + im * im;
            if r < cum {
                self.sample_us.set(self.sample_us.get() + t1.elapsed().as_micros());
                return Ok(i);
            }
        }
        self.sample_us.set(self.sample_us.get() + t1.elapsed().as_micros());
        Ok(self.n - 1)
    }

    /// Transfer the state once and build the cumulative probability array.
    fn transfer_cumulative(&self) -> Result<Vec<f64>, String> {
        let t0 = Instant::now();
        let mut buf = vec![0.0f32; 2 * self.n];
        unsafe {
            cu_check(
                ffi::cuMemcpyDtoH(buf.as_mut_ptr().cast(), self.state_buf, (8 * self.n) as u64),
                "cuMemcpyDtoH state",
            )?;
        }
        self.transfer_us.set(self.transfer_us.get() + t0.elapsed().as_micros());
        let mut cum = Vec::with_capacity(self.n + 1);
        cum.push(0.0f64);
        let mut acc = 0.0f64;
        for i in 0..self.n {
            let re = buf[2 * i] as f64;
            let im = buf[2 * i + 1] as f64;
            acc += re * re + im * im;
            cum.push(acc);
        }
        Ok(cum)
    }

    /// Draw one sample from a cumulative probability array. Binary search
    /// (O(log n) per sample) so host-side sampling stays cheap for large
    /// states; the linear walk was O(n) and dominated at scale.
    fn sample_from_cum(cum: &[f64], r: f64) -> usize {
        let n = cum.len() - 1;
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if r < cum[mid + 1] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo.min(n - 1)
    }

    /// True if the program contains no measurement or classical branching
    /// (its final state is deterministic, so batch sampling is valid).
    fn is_pure_unitary(prog: &Program) -> bool {
        for i in 0..prog.len {
            match prog.ops[i] {
                IrOp::Measure(_, _) | IrOp::IfEq(_, _, _, _) => return false,
                _ => {}
            }
        }
        true
    }

    /// Fast path: for a pure-unitary program, apply the gates once, transfer
    /// the fixed final state once, and draw all samples host-side. Identical
    /// to the general path (same RNG stream) but one transfer instead of N.
    fn run_pure_batch(
        &self,
        prog: &Program,
        shots_override: Option<u32>,
        classical: &mut [u8],
        histogram: &mut [u32],
        rng: &mut Lcg64,
    ) -> Result<(), String> {
        let (end, samples): (usize, u64) = if !prog.has_explicit_shot {
            (prog.len, shots_override.unwrap_or(1) as u64)
        } else {
            // Assume trailing bare SHOT(s): unitary is everything before the
            // last SHOT; total samples = sum of all shot counts.
            let mut last_shot = 0usize;
            let mut sum = 0u64;
            for i in 0..prog.len {
                if let IrOp::Shot(c, _, _) = prog.ops[i] {
                    last_shot = i;
                    sum += c.get() as u64;
                }
            }
            (last_shot, sum.max(1))
        };

        self.reset()?;
        unsafe {
            cu_check(ffi::cuEventRecord(self.ev_start, std::ptr::null_mut()), "cuEventRecord")?;
        }
        self.exec(prog, 0, end, classical, histogram, rng)?;
        unsafe {
            cu_check(ffi::cuEventRecord(self.ev_stop, std::ptr::null_mut()), "cuEventRecord")?;
            cu_check(ffi::cuEventSynchronize(self.ev_stop), "cuEventSynchronize")?;
            let mut ms: f32 = 0.0;
            cu_check(
                ffi::cuEventElapsedTime(&mut ms, self.ev_start, self.ev_stop),
                "cuEventElapsedTime",
            )?;
            self.exec_gpu_us.set((ms * 1000.0) as u128);
        }
        let cum = self.transfer_cumulative()?;
        let t_samp = Instant::now();
        for _ in 0..samples {
            let r = rng.next_f64();
            let idx = Self::sample_from_cum(&cum, r);
            histogram[idx] += 1;
        }
        self.sample_us.set(self.sample_us.get() + t_samp.elapsed().as_micros());
        Ok(())
    }

    // ─── IR executor ──────────────────────────────────────────────

    fn exec(
        &self,
        prog: &Program,
        offset: usize,
        end: usize,
        classical: &mut [u8],
        histogram: &mut [u32],
        rng: &mut Lcg64,
    ) -> Result<(), String> {
        let t_exec = Instant::now();
        let mut i = offset;
        while i < end {
            match prog.ops[i] {
                IrOp::H(q) => self.apply_h(q)?,
                IrOp::X(q) => self.apply_x(q)?,
                IrOp::CNOT(c, t) => self.apply_cnot(c, t)?,
                IrOp::Toff(c1, c2, t) => self.apply_toff(c1, c2, t)?,
                IrOp::Measure(q, c) => {
                    let p = self.measure_prob(q)?;
                    let outcome = if rng.next_f64() < p as f64 { 1u32 } else { 0u32 };
                    // Renormalise the surviving branch: its probability is p
                    // when outcome==1, and 1-p when outcome==0.
                    let p_branch = if outcome == 1 { p as f64 } else { 1.0 - p as f64 };
                    let inv = if p_branch > 0.0 { 1.0 / p_branch.sqrt() } else { 0.0 };
                    self.collapse(q, outcome, inv as f32)?;
                    classical[c as usize] = outcome as u8;
                }
                IrOp::IfEq(c, v, boff, blen) => {
                    let (b0, bl) = (boff as usize, blen as usize);
                    if classical[c as usize] == v {
                        self.exec(prog, b0, b0 + bl, classical, histogram, rng)?;
                    }
                    i = b0 + bl;
                    continue;
                }
                IrOp::Shot(count, boff, blen) => {
                    let (b0, bl) = (boff as usize, blen as usize);
                    if bl == 0 {
                        let shot_pos = i;
                        for _ in 0..count.get() {
                            self.reset()?;
                            self.exec(prog, 0, shot_pos, classical, histogram, rng)?;
                            let s = self.sample_host(rng)?;
                            histogram[s] += 1;
                        }
                    } else {
                        for _ in 0..count.get() {
                            self.reset()?;
                            self.exec(prog, b0, b0 + bl, classical, histogram, rng)?;
                            let s = self.sample_host(rng)?;
                            histogram[s] += 1;
                        }
                        i = b0 + bl;
                        continue;
                    }
                }
                IrOp::Print => {}
            }
            i += 1;
        }
        self.exec_us.set(self.exec_us.get() + t_exec.elapsed().as_micros());
        Ok(())
    }

    /// Run a program on the GPU, filling the histogram.
    pub fn run_program(
        &self,
        prog: &Program,
        shots_override: Option<u32>,
        classical: &mut [u8],
        histogram: &mut [u32],
        rng: &mut Lcg64,
    ) -> Result<(), String> {
        for c in classical.iter_mut() {
            *c = 0;
        }
        for h in histogram.iter_mut() {
            *h = 0;
        }
        self.exec_us.set(0);
        self.exec_gpu_us.set(0);
        self.transfer_us.set(0);
        self.sample_us.set(0);
        self.launches.set(0);

        if Self::is_pure_unitary(prog) {
            return self.run_pure_batch(prog, shots_override, classical, histogram, rng);
        }

        match (shots_override, prog.has_explicit_shot) {
            // CLI override on a SHOT-less program: bare-shot loop.
            (Some(n), false) => {
                for _ in 0..n {
                    self.reset()?;
                    self.exec(prog, 0, prog.len, classical, histogram, rng)?;
                    let s = self.sample_host(rng)?;
                    histogram[s] += 1;
                }
            }
            _ => {
                self.exec(prog, 0, prog.len, classical, histogram, rng)?;
                if !prog.has_explicit_shot {
                    let s = self.sample_host(rng)?;
                    histogram[s] += 1;
                }
            }
        }
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cuMemFree(self.state_buf);
            let _ = ffi::cuMemFree(self.prob_buf);
            let _ = ffi::cuEventDestroy(self.ev_start);
            let _ = ffi::cuEventDestroy(self.ev_stop);
            let _ = ffi::cuModuleUnload(self.module);
            let _ = ffi::cuCtxDestroy(self.ctx);
        }
    }
}
