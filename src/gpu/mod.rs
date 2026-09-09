// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

//! Host-side GPU backend. Loads the PTX kernel library through
//! the CUDA driver API, drives it from the Plasma IR, and samples host-side
//! (determinism: every random decision uses the host RNG).

use std::cell::Cell;
use std::ffi::c_void;
use std::time::Instant;

use plasma::ir::{IrOp, Program};
use plasma::rng::Lcg64;

// CUDA driver API bindings (raw FFI, no dependencies)

// Hand-declared CUDA driver surface; entries are added ahead of use by
// later pipeline work (async/pinned copies, streams).
#[allow(dead_code)]
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
        pub fn cuMemsetD8(ptr: u64, value: u8, count: u64) -> i32;
        pub fn cuMemHostAlloc(ptr: *mut *mut c_void, size: u64, flags: u32) -> i32;
        pub fn cuMemFreeHost(ptr: *mut c_void) -> i32;
        pub fn cuMemcpyDtoHAsync(
            dst: *mut c_void,
            src: u64,
            count: u64,
            stream: *mut (),
        ) -> i32;
        pub fn cuStreamSynchronize(stream: *mut ()) -> i32;
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
        pub fn cuLaunchCooperativeKernel(
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
        ) -> i32;
        pub fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
            num_blocks: *mut i32,
            func: *mut (),
            block_size: i32,
            dynamic_shared: usize,
        ) -> i32;
        pub fn cuFuncSetAttribute(func: *mut (), attrib: i32, value: i32) -> i32;
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

// GPU backend

pub struct Gpu {
    ctx: *mut ffi::Ctx,
    module: *mut (),
    f_reset: *mut (),
    f_h: *mut (),
    f_x: *mut (),
    f_cnot: *mut (),
    f_toff: *mut (),
    f_prob: *mut (),
    f_finalize: *mut (),
    f_collapse: *mut (),
    // v0.2 kernels
    f_rz: *mut (),
    f_rx: *mut (),
    f_ry: *mut (),
    f_phase: *mut (),
    f_s: *mut (),
    f_sdg: *mut (),
    f_t: *mut (),
    f_sx: *mut (),
    f_swap: *mut (),
    f_iswap: *mut (),
    f_cz: *mut (),
    f_cphase: *mut (),
    f_cswap: *mut (),
    f_mcx: *mut (),
    // Shared-memory whole-circuit kernel (M4-P2).
    f_mega: *mut (),
    ops_buf: u64,
    // EXPECT (Pauli expectation) kernels + buffers.
    f_expect_partial: *mut (),
    f_expect_finalize: *mut (),
    expect_scratch: u64,
    expect_out: u64,
    /// Pinned host staging buffer for state transfers (async DtoH). Null if
    /// pinned memory could not be allocated (falls back to sync copies).
    pinned: *mut f32,
    state_buf: u64,
    prob_buf: u64,
    prob_scratch: u64,
    nblocks: u32,
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
            let f_prob = get("prob_partial")?;
            let f_finalize = get("prob_finalize")?;
            let f_collapse = get("collapse")?;
            let f_rz = get("apply_rz")?;
            let f_rx = get("apply_rx")?;
            let f_ry = get("apply_ry")?;
            let f_phase = get("apply_phase")?;
            let f_s = get("apply_s")?;
            let f_sdg = get("apply_sdg")?;
            let f_t = get("apply_t")?;
            let f_sx = get("apply_sx")?;
            let f_swap = get("apply_swap")?;
            let f_iswap = get("apply_iswap")?;
            let f_cz = get("apply_cz")?;
            let f_cphase = get("apply_cphase")?;
            let f_cswap = get("apply_cswap")?;
            let f_mcx = get("apply_mcx")?;
            let f_mega = get("mega")?;
            // Opt in to >48 KB dynamic shared memory so the mega kernel can
            // hold states up to 2^14 amps (128 KB) in one block. CUDA's
            // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8.
            let _ = ffi::cuFuncSetAttribute(f_mega, 8, 64 * 1024);
            let f_expect_partial = get("expect_partial")?;
            let f_expect_finalize = get("expect_finalize")?;

            let mut ops_buf: u64 = 0;
            cu_check(
                ffi::cuMemAlloc(&mut ops_buf, (8 * 4 * plasma::ir::MAX_OPS) as u64),
                "cuMemAlloc ops",
            )?;

            let mut state_buf: u64 = 0;
            cu_check(
                ffi::cuMemAlloc(&mut state_buf, (8 * n) as u64),
                "cuMemAlloc state",
            )?;
            // Zero the initial state so no execution path can depend on the
            // driver's (unreliable) fresh-allocation contents.
            cu_check(
                ffi::cuMemsetD8(state_buf, 0, (8 * n) as u64),
                "cuMemsetD8 state",
            )?;
            let mut prob_buf: u64 = 0;
            cu_check(ffi::cuMemAlloc(&mut prob_buf, 4), "cuMemAlloc prob")?;
            let nblocks = ((n as u32 + BLOCK - 1) / BLOCK).max(1);
            let mut prob_scratch: u64 = 0;
            cu_check(
                ffi::cuMemAlloc(&mut prob_scratch, (8 * nblocks) as u64),
                "cuMemAlloc prob scratch",
            )?;
            let mut expect_scratch: u64 = 0;
            cu_check(
                ffi::cuMemAlloc(&mut expect_scratch, (16 * nblocks) as u64),
                "cuMemAlloc expect scratch",
            )?;
            let mut expect_out: u64 = 0;
            cu_check(ffi::cuMemAlloc(&mut expect_out, 8), "cuMemAlloc expect out")?;
            // Pinned staging buffer for fast async device->host transfers.
            let mut pinned: *mut f32 = std::ptr::null_mut();
            let mut pin_rc = 0;
            if std::env::var("PLASMA_NO_PINNED").is_err() {
                pin_rc = ffi::cuMemHostAlloc(
                (&mut pinned as *mut *mut f32).cast(),
                (8 * n) as u64,
                0,
                );
            }
            if pin_rc != 0 {
                pinned = std::ptr::null_mut();
            }
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
                f_finalize,
                f_collapse,
                f_rz,
                f_rx,
                f_ry,
                f_phase,
                f_s,
                f_sdg,
                f_t,
                f_sx,
                f_swap,
                f_iswap,
                f_cz,
                f_cphase,
                f_cswap,
                f_mcx,
                f_mega,
                ops_buf,
                f_expect_partial,
                f_expect_finalize,
                expect_scratch,
                expect_out,
                pinned,
                state_buf,
                prob_buf,
                prob_scratch,
                nblocks,
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
        self.launch_grid(func, grid, BLOCK, args)
    }

    fn launch_grid(
        &self,
        func: *mut (),
        grid: u32,
        block: u32,
        args: &[&dyn KernelArg],
    ) -> Result<(), String> {
        self.launches.set(self.launches.get() + 1);
        unsafe {
            let params: Vec<*const c_void> = args.iter().map(|a| a.as_ptr()).collect();
            cu_check(
                ffi::cuLaunchKernel(
                    func,
                    grid,
                    1,
                    1,
                    block,
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

    // v0.2 kernels

    /// Half-angle params for RX/RY/RZ (host computes the deterministic trig).
    fn half_angle(theta: f64) -> (f32, f32) {
        let (s, c) = plasma::math::sin_cos(theta * 0.5);
        (c as f32, s as f32)
    }

    /// Full-angle params for Phase/CPHASE.
    fn full_angle(theta: f64) -> (f32, f32) {
        let (s, c) = plasma::math::sin_cos(theta);
        (c as f32, s as f32)
    }

    fn apply_rz(&self, q: u8, theta: f64) -> Result<(), String> {
        let (cs, sn) = Self::half_angle(theta);
        let (n, q) = (self.n as u32, q as u32);
        self.launch(self.f_rz, &[&self.state_buf, &n, &q, &cs, &sn])
    }

    fn apply_rx(&self, q: u8, theta: f64) -> Result<(), String> {
        let (cs, sn) = Self::half_angle(theta);
        let (n, q) = (self.n as u32, q as u32);
        self.launch(self.f_rx, &[&self.state_buf, &n, &q, &cs, &sn])
    }

    fn apply_ry(&self, q: u8, theta: f64) -> Result<(), String> {
        let (cs, sn) = Self::half_angle(theta);
        let (n, q) = (self.n as u32, q as u32);
        self.launch(self.f_ry, &[&self.state_buf, &n, &q, &cs, &sn])
    }

    fn apply_phase(&self, q: u8, theta: f64) -> Result<(), String> {
        let (c, sn) = Self::full_angle(theta);
        let (n, q) = (self.n as u32, q as u32);
        self.launch(self.f_phase, &[&self.state_buf, &n, &q, &c, &sn])
    }

    fn apply_s(&self, q: u8) -> Result<(), String> {
        let (n, q) = (self.n as u32, q as u32);
        self.launch(self.f_s, &[&self.state_buf, &n, &q])
    }

    fn apply_sdg(&self, q: u8) -> Result<(), String> {
        let (n, q) = (self.n as u32, q as u32);
        self.launch(self.f_sdg, &[&self.state_buf, &n, &q])
    }

    fn apply_t(&self, q: u8) -> Result<(), String> {
        let (n, q) = (self.n as u32, q as u32);
        self.launch(self.f_t, &[&self.state_buf, &n, &q])
    }

    fn apply_sx(&self, q: u8) -> Result<(), String> {
        let (n, q) = (self.n as u32, q as u32);
        self.launch(self.f_sx, &[&self.state_buf, &n, &q])
    }

    fn apply_swap(&self, a: u8, b: u8) -> Result<(), String> {
        let (n, a, b) = (self.n as u32, a as u32, b as u32);
        self.launch(self.f_swap, &[&self.state_buf, &n, &a, &b])
    }

    fn apply_iswap(&self, a: u8, b: u8) -> Result<(), String> {
        let (n, a, b) = (self.n as u32, a as u32, b as u32);
        self.launch(self.f_iswap, &[&self.state_buf, &n, &a, &b])
    }

    fn apply_cz(&self, a: u8, b: u8) -> Result<(), String> {
        let (n, a, b) = (self.n as u32, a as u32, b as u32);
        self.launch(self.f_cz, &[&self.state_buf, &n, &a, &b])
    }

    fn apply_cphase(&self, a: u8, b: u8, theta: f64) -> Result<(), String> {
        let (c, sn) = Self::full_angle(theta);
        let (n, a, b) = (self.n as u32, a as u32, b as u32);
        self.launch(self.f_cphase, &[&self.state_buf, &n, &a, &b, &c, &sn])
    }

    fn apply_cswap(&self, ctl: u8, b: u8, c: u8) -> Result<(), String> {
        let (n, ctl, b, c) = (self.n as u32, ctl as u32, b as u32, c as u32);
        self.launch(self.f_cswap, &[&self.state_buf, &n, &ctl, &b, &c])
    }

    fn apply_mcx(&self, mask: u32, t: u8) -> Result<(), String> {
        let (n, t) = (self.n as u32, t as u32);
        self.launch(self.f_mcx, &[&self.state_buf, &n, &mask, &t])
    }

    /// Measure |q>: deterministic P(q=1), host RNG draw, collapse. Returns
    /// the outcome and leaves the collapsed state on the device.
    fn measure_outcome(&self, q: u8, rng: &mut Lcg64) -> Result<u32, String> {
        let p = self.measure_prob(q)?;
        let outcome = if rng.next_f64() < p as f64 { 1u32 } else { 0u32 };
        let p_branch = if outcome == 1 { p as f64 } else { 1.0 - p as f64 };
        let inv = if p_branch > 0.0 { 1.0 / p_branch.sqrt() } else { 0.0 };
        self.collapse(q, outcome, inv as f32)?;
        Ok(outcome)
    }

    /// Split a 2-bits-per-qubit Pauli code into GPU kernel parameters:
    /// (flip_mask, z_mask, y_mask, ybase_re, ybase_im) with ybase = (-i)^|Y|.
    fn pauli_masks(pauli: u64) -> (u32, u32, u32, f32, f32) {
        let mut flip = 0u32;
        let mut zm = 0u32;
        let mut ym = 0u32;
        let mut ny = 0u32;
        for q in 0..32u64 {
            match (pauli >> (2 * q)) & 3 {
                1 => flip |= 1 << q,
                2 => {
                    flip |= 1 << q;
                    ym |= 1 << q;
                    ny += 1;
                }
                3 => zm |= 1 << q,
                _ => {}
            }
        }
        let (yb_r, yb_i) = match ny & 3 {
            0 => (1.0f32, 0.0f32),
            1 => (0.0f32, -1.0f32),
            2 => (-1.0f32, 0.0f32),
            _ => (0.0f32, 1.0f32),
        };
        (flip, zm, ym, yb_r, yb_i)
    }

    /// <ψ|P|ψ> via the deterministic two-stage reduction. Bit-reproducible
    /// run-to-run (same fixed-order summation as prob).
    fn expect_value(&self, pauli: u64) -> Result<f64, String> {
        let (flip, zm, ym, yb_r, yb_i) = Self::pauli_masks(pauli);
        let n = self.n as u32;
        self.launch(
            self.f_expect_partial,
            &[
                &self.state_buf,
                &n,
                &flip,
                &zm,
                &ym,
                &yb_r,
                &yb_i,
                &self.expect_scratch,
            ],
        )?;
        self.launch_grid(
            self.f_expect_finalize,
            1,
            BLOCK,
            &[&self.expect_scratch, &self.nblocks, &self.expect_out],
        )?;
        let mut out = [0.0f32; 2];
        unsafe {
            cu_check(
                ffi::cuMemcpyDtoH(out.as_mut_ptr().cast(), self.expect_out, 8),
                "cuMemcpyDtoH expect",
            )?;
        }
        Ok(out[0] as f64)
    }

    /// Return P(qubit q = 1) via a deterministic two-stage reduction: each block
    /// writes a fixed-order partial to scratch[block_id], then prob_finalize
    /// sums the partials in block order. No atomics, no HtoD zeroing, and the
    /// result is bit-reproducible run-to-run for a given program.
    fn measure_prob(&self, q: u8) -> Result<f32, String> {
        let n = self.n as u32;
        let q = q as u32;
        self.launch(self.f_prob, &[&self.state_buf, &n, &q, &self.prob_scratch])?;
        self.launch_grid(
            self.f_finalize,
            1,
            BLOCK,
            &[&self.prob_scratch, &self.nblocks, &self.prob_buf],
        )?;
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

    /// Transfer the device state into host memory. Returns a slice of the
    /// (re, im) interleaved f32 amplitudes. Uses the pinned staging buffer
    /// with an async copy when available (faster, and the host can start
    /// reading while the copy engine works); otherwise falls back to a
    /// synchronous copy into a caller-owned scratch Vec.
    fn transfer_to_host<'a>(
        &self,
        scratch: &'a mut Vec<f32>,
    ) -> Result<&'a [f32], String> {
        let t0 = Instant::now();
        if !self.pinned.is_null() {
            unsafe {
                cu_check(
                    ffi::cuMemcpyDtoHAsync(
                        self.pinned.cast(),
                        self.state_buf,
                        (8 * self.n) as u64,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync state",
                )?;
                cu_check(ffi::cuStreamSynchronize(std::ptr::null_mut()), "cuStreamSynchronize")?;
            }
            let sl = unsafe {
                std::slice::from_raw_parts(self.pinned as *const f32, 2 * self.n)
            };
            self.transfer_us.set(self.transfer_us.get() + t0.elapsed().as_micros());
            Ok(sl)
        } else {
            scratch.resize(2 * self.n, 0.0);
            unsafe {
                cu_check(
                    ffi::cuMemcpyDtoH(
                        scratch.as_mut_ptr().cast(),
                        self.state_buf,
                        (8 * self.n) as u64,
                    ),
                    "cuMemcpyDtoH state",
                )?;
            }
            self.transfer_us.set(self.transfer_us.get() + t0.elapsed().as_micros());
            Ok(scratch)
        }
    }

    /// Transfer the final state to the host and draw one sample.
    fn sample_host(&self, rng: &mut Lcg64) -> Result<usize, String> {
        let mut scratch = Vec::new();
        let buf = self.transfer_to_host(&mut scratch)?;
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
        let mut scratch = Vec::new();
        let buf = self.transfer_to_host(&mut scratch)?;
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
    /// MeasureX/Y and Reset are excluded too: they collapse the state and
    /// draw from the host RNG, so the final state is not deterministic.
    fn is_pure_unitary(prog: &Program) -> bool {
        for i in 0..prog.len {
            match prog.ops[i] {
                IrOp::Measure(_, _)
                | IrOp::IfEq(_, _, _, _)
                | IrOp::IfNe(_, _, _, _)
                | IrOp::MeasureX(_, _)
                | IrOp::MeasureY(_, _)
                | IrOp::Reset(_) => return false,
                _ => {}
            }
        }
        true
    }

    /// True if the single-transfer fast path is valid: pure-unitary AND at most
    /// one explicit SHOT. Multiple SHOTs interleave resets and replay different
    /// prefixes per iteration, which a single transfer cannot reproduce.
    fn is_batchable(prog: &Program) -> bool {
        if !Self::is_pure_unitary(prog) {
            return false;
        }
        if !prog.has_explicit_shot {
            return true;
        }
        let mut shots = 0;
        for i in 0..prog.len {
            if let IrOp::Shot(_, _, _) = prog.ops[i] {
                shots += 1;
            }
        }
        shots <= 1
    }

    // Shared-memory whole-circuit kernel (M4-P2)

    /// Amplitudes that fit one block's shared memory. The RTX 50-series
    /// (sm_120) opt-in dynamic shared limit is ~99 KB, so 2^13 amps (64 KB)
    /// fits; the default 48 KB covers 2^12 amps without the attribute.
    const MEGA_MAX_AMPS: usize = 1 << 13;

    /// Encode the gate ops in `prog.ops[start..end]` as mega-kernel records
    /// (8 u32 each). Returns None if any op is unsupported (caller falls back).
    fn build_mega_ops(prog: &Program, start: usize, end: usize) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        for i in start..end {
            let mut r = [0u32; 8];
            match prog.ops[i] {
                IrOp::H(q) => {
                    r[0] = 0;
                    r[1] = q as u32;
                }
                IrOp::X(q) => {
                    r[0] = 1;
                    r[1] = q as u32;
                }
                IrOp::CNOT(c, t) => {
                    r[0] = 2;
                    r[1] = c as u32;
                    r[2] = t as u32;
                }
                IrOp::Toff(a, b, t) => {
                    r[0] = 3;
                    r[1] = a as u32;
                    r[2] = b as u32;
                    r[3] = t as u32;
                }
                IrOp::RZ(q, k) => {
                    let (s, c) = plasma::math::sin_cos(prog.consts[k as usize] * 0.5);
                    r[0] = 4;
                    r[1] = q as u32;
                    r[4] = (c as f32).to_bits();
                    r[5] = (s as f32).to_bits();
                }
                IrOp::RX(q, k) => {
                    let (s, c) = plasma::math::sin_cos(prog.consts[k as usize] * 0.5);
                    r[0] = 5;
                    r[1] = q as u32;
                    r[4] = (c as f32).to_bits();
                    r[5] = (s as f32).to_bits();
                }
                IrOp::RY(q, k) => {
                    let (s, c) = plasma::math::sin_cos(prog.consts[k as usize] * 0.5);
                    r[0] = 6;
                    r[1] = q as u32;
                    r[4] = (c as f32).to_bits();
                    r[5] = (s as f32).to_bits();
                }
                IrOp::Phase(q, k) => {
                    let (s, c) = plasma::math::sin_cos(prog.consts[k as usize]);
                    r[0] = 7;
                    r[1] = q as u32;
                    r[4] = (c as f32).to_bits();
                    r[5] = (s as f32).to_bits();
                }
                IrOp::S(q) => {
                    r[0] = 8;
                    r[1] = q as u32;
                }
                IrOp::T(q) => {
                    r[0] = 9;
                    r[1] = q as u32;
                }
                IrOp::SX(q) => {
                    r[0] = 10;
                    r[1] = q as u32;
                }
                IrOp::SWAP(a, b) => {
                    r[0] = 11;
                    r[1] = a as u32;
                    r[2] = b as u32;
                }
                IrOp::ISWAP(a, b) => {
                    r[0] = 12;
                    r[1] = a as u32;
                    r[2] = b as u32;
                }
                IrOp::CZ(a, b) => {
                    r[0] = 13;
                    r[1] = a as u32;
                    r[2] = b as u32;
                }
                IrOp::CPHASE(a, b, k) => {
                    let (s, c) = plasma::math::sin_cos(prog.consts[k as usize]);
                    r[0] = 14;
                    r[1] = a as u32;
                    r[2] = b as u32;
                    r[4] = (c as f32).to_bits();
                    r[5] = (s as f32).to_bits();
                }
                IrOp::CSWAP(a, b, c) => {
                    r[0] = 15;
                    r[1] = a as u32;
                    r[2] = b as u32;
                    r[3] = c as u32;
                }
                IrOp::MCX(mask, t) => {
                    r[0] = 16;
                    r[1] = mask;
                    r[2] = t as u32;
                }
                // PRINT is a no-op for state evolution; skip it.
                IrOp::Print => continue,
                _ => return None,
            }
            out.extend_from_slice(&r);
        }
        Some(out)
    }

    /// Run the whole pure-unitary gate region in ONE shared-memory launch.
    /// Returns Ok(true) if the mega path ran; Ok(false) if it fell back (state
    /// too large for shared or an unsupported op); Err on CUDA failure.
    fn run_mega(
        &self,
        prog: &Program,
        start: usize,
        end: usize,
        samples: u64,
        histogram: &mut [u32],
        rng: &mut Lcg64,
    ) -> Result<bool, String> {
        if std::env::var("PLASMA_NO_MEGA").is_ok() {
            return Ok(false);
        }
        if self.n > Self::MEGA_MAX_AMPS {
            return Ok(false);
        }
        let Some(ops) = Self::build_mega_ops(prog, start, end) else {
            return Ok(false);
        };
        if ops.is_empty() {
            return Ok(false);
        }
        let nops = (ops.len() / 8) as u32;
        unsafe {
            cu_check(
                ffi::cuMemcpyHtoD(self.ops_buf, ops.as_ptr().cast(), (ops.len() * 4) as u64),
                "cuMemcpyHtoD ops",
            )?;
        }
        let n = self.n as u32;
        let block = n.min(1024);
        let shared = (8 * self.n) as u32;
        unsafe {
            cu_check(ffi::cuEventRecord(self.ev_start, std::ptr::null_mut()), "cuEventRecord")?;
            cu_check(
                ffi::cuLaunchKernel(
                    self.f_mega,
                    1,
                    1,
                    1,
                    block,
                    1,
                    1,
                    shared,
                    std::ptr::null_mut(),
                    [&self.state_buf as *const u64 as *const std::ffi::c_void,
                     &n as *const u32 as *const std::ffi::c_void,
                     &self.ops_buf as *const u64 as *const std::ffi::c_void,
                     &nops as *const u32 as *const std::ffi::c_void]
                    .as_ptr(),
                    std::ptr::null(),
                ),
                "cuLaunchKernel mega",
            )?;
        }
        self.launches.set(self.launches.get() + 1);
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
        // The state now holds the post-circuit state; sample host-side.
        let cum = self.transfer_cumulative()?;
        let t_samp = Instant::now();
        for _ in 0..samples {
            let r = rng.next_f64();
            let idx = Self::sample_from_cum(&cum, r);
            histogram[idx] += 1;
        }
        self.sample_us.set(self.sample_us.get() + t_samp.elapsed().as_micros());
        Ok(true)
    }

    /// Fast path: for a pure-unitary program with at most one SHOT, apply the
    /// relevant unitary once, transfer the fixed final state once, and draw
    /// all samples host-side. Identical to the general path (same RNG stream)
    /// but one transfer instead of N.
    fn run_pure_batch(
        &self,
        prog: &Program,
        shots_override: Option<u32>,
        results: &mut plasma::sim::Results,
        classical: &mut [u8],
        histogram: &mut [u32],
        rng: &mut Lcg64,
    ) -> Result<(), String> {
        // The state the samples are drawn from. Every SHOT resets to |0...0>
        // and replays only its own region, so the sampled state is exactly the
        // post-region state and surrounding ops never contribute to samples.
        //   - no explicit SHOT: the whole program, sampled shots_override times;
        //   - bare SHOT (blen == 0): everything before the SHOT op;
        //   - body SHOT (blen != 0): only the inline body region.
        let (start, end, samples): (usize, usize, u64) = if !prog.has_explicit_shot {
            (0, prog.len, shots_override.unwrap_or(1) as u64)
        } else {
            let mut single = None;
            for i in 0..prog.len {
                if let IrOp::Shot(c, boff, blen) = prog.ops[i] {
                    if single.is_some() {
                        // Multiple SHOTs: fall back to the general executor.
                        self.exec(prog, 0, prog.len, results, classical, histogram, rng, &[None; 3])?;
                        return Ok(());
                    }
                    single = Some((i, c.get(), boff, blen));
                }
            }
            match single {
                Some((shot_pos, count, _, 0)) => (0, shot_pos, count as u64),
                Some((_, count, boff, blen)) => {
                    (boff as usize, (boff + blen) as usize, count as u64)
                }
                None => (0, prog.len, shots_override.unwrap_or(1) as u64),
            }
        };

        // Fastest path: run the whole region in ONE shared-memory launch
        // (mega kernel) when the state fits. Falls back to per-gate launches.
        if self.run_mega(prog, start, end, samples, histogram, rng)? {
            return Ok(());
        }

        self.reset()?;
        unsafe {
            cu_check(ffi::cuEventRecord(self.ev_start, std::ptr::null_mut()), "cuEventRecord")?;
        }
        self.exec(prog, start, end, results, classical, histogram, rng, &[None; 3])?;
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

    // IR executor

    fn exec(
        &self,
        prog: &Program,
        offset: usize,
        end: usize,
        results: &mut plasma::sim::Results,
        classical: &mut [u8],
        histogram: &mut [u32],
        rng: &mut Lcg64,
        remap: &plasma::sim::Remap,
    ) -> Result<(), String> {
        let t_exec = Instant::now();
        let mut i = offset;
        while i < end {
            let raw = prog.ops[i];
            let op = if plasma::sim::remap_active(remap) {
                plasma::sim::translate_op(raw, remap)
            } else {
                raw
            };
            match op {
                IrOp::H(q) => self.apply_h(q)?,
                IrOp::X(q) => self.apply_x(q)?,
                IrOp::CNOT(c, t) => self.apply_cnot(c, t)?,
                IrOp::Toff(c1, c2, t) => self.apply_toff(c1, c2, t)?,
                IrOp::RZ(q, k) => self.apply_rz(q, prog.consts[k as usize])?,
                IrOp::RX(q, k) => self.apply_rx(q, prog.consts[k as usize])?,
                IrOp::RY(q, k) => self.apply_ry(q, prog.consts[k as usize])?,
                IrOp::Phase(q, k) => self.apply_phase(q, prog.consts[k as usize])?,
                IrOp::S(q) => self.apply_s(q)?,
                IrOp::T(q) => self.apply_t(q)?,
                IrOp::SX(q) => self.apply_sx(q)?,
                IrOp::SWAP(a, b) => self.apply_swap(a, b)?,
                IrOp::ISWAP(a, b) => self.apply_iswap(a, b)?,
                IrOp::CZ(a, b) => self.apply_cz(a, b)?,
                IrOp::CPHASE(a, b, k) => self.apply_cphase(a, b, prog.consts[k as usize])?,
                IrOp::CSWAP(c, b, t) => self.apply_cswap(c, b, t)?,
                IrOp::MCX(mask, t) => self.apply_mcx(mask, t)?,
                IrOp::Reset(q) => {
                    if self.measure_outcome(q, rng)? == 1 {
                        self.apply_x(q)?;
                    }
                }
                IrOp::MeasureX(q, c) => {
                    self.apply_h(q)?;
                    let outcome = self.measure_outcome(q, rng)?;
                    self.apply_h(q)?;
                    classical[c as usize] = outcome as u8;
                }
                IrOp::MeasureY(q, c) => {
                    self.apply_sdg(q)?;
                    self.apply_h(q)?;
                    let outcome = self.measure_outcome(q, rng)?;
                    self.apply_h(q)?;
                    self.apply_s(q)?;
                    classical[c as usize] = outcome as u8;
                }
                // Classical computation is host-side on both backends.
                IrOp::Set(c, v) => classical[c as usize] = v,
                IrOp::Not(c) => classical[c as usize] ^= 1,
                IrOp::And(a, b) => classical[a as usize] &= classical[b as usize],
                IrOp::Or(a, b) => classical[a as usize] |= classical[b as usize],
                IrOp::Xor(a, b) => classical[a as usize] ^= classical[b as usize],
                IrOp::Add(a, b) => {
                    classical[a as usize] = classical[a as usize].wrapping_add(classical[b as usize])
                }
                IrOp::Sub(a, b) => {
                    classical[a as usize] = classical[a as usize].wrapping_sub(classical[b as usize])
                }
                // Observables: EXPECT uses the deterministic two-stage
                // reduction; SAVE_* transfer the state and capture host-side.
                IrOp::Expect(pauli) => {
                    let v = self.expect_value(pauli)?;
                    results.expectations.push(v);
                }
                IrOp::Estimate(off, len) => {
                    let mut h = 0.0;
                    for k in off as usize..(off as usize + len as usize) {
                        let (c, p) = prog.estimate_terms[k];
                        h += c * self.expect_value(p as u64)?;
                    }
                    results.estimates.push(h);
                }
                IrOp::SaveState => {
                    let buf = self.dump_state()?;
                    results.saved_states.push(buf);
                }
                IrOp::SaveAmps => {
                    let buf = self.dump_state()?;
                    let mut v = Vec::with_capacity(self.n);
                    for k in 0..self.n {
                        v.push((k, buf[2 * k], buf[2 * k + 1]));
                    }
                    results.saved_amplitudes.push(v);
                }
                IrOp::SaveProbs => {
                    let buf = self.dump_state()?;
                    let mut v = Vec::with_capacity(self.n);
                    for k in 0..self.n {
                        v.push(buf[2 * k] * buf[2 * k] + buf[2 * k + 1] * buf[2 * k + 1]);
                    }
                    results.saved_probs.push(v);
                }
                IrOp::Call(sub_id, a0, a1, a2) => {
                    let sub = prog.subs[sub_id as usize];
                    let np = sub.nparams;
                    let mut nremap = [None; 3];
                    let cargs = [a0, a1, a2];
                    for k in 0..np {
                        nremap[k as usize] = Some(cargs[k as usize]);
                    }
                    let (off, len) = (sub.off as usize, sub.len as usize);
                    self.exec(
                        prog,
                        off,
                        off + len,
                        results,
                        classical,
                        histogram,
                        rng,
                        &nremap,
                    )?;
                }
                IrOp::Measure(q, c) => {
                    let outcome = self.measure_outcome(q, rng)?;
                    classical[c as usize] = outcome as u8;
                }
                IrOp::IfEq(c, v, boff, blen) => {
                    let (b0, bl) = (boff as usize, blen as usize);
                    if classical[c as usize] == v {
                        self.exec(prog, b0, b0 + bl, results, classical, histogram, rng, remap)?;
                    }
                    i = b0 + bl;
                    continue;
                }
                IrOp::IfNe(c, v, boff, blen) => {
                    let (b0, bl) = (boff as usize, blen as usize);
                    if classical[c as usize] != v {
                        self.exec(prog, b0, b0 + bl, results, classical, histogram, rng, remap)?;
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
                            self.exec(prog, 0, shot_pos, results, classical, histogram, rng, remap)?;
                            let s = self.sample_host(rng)?;
                            histogram[s] += 1;
                        }
                    } else {
                        for _ in 0..count.get() {
                            self.reset()?;
                            self.exec(prog, b0, b0 + bl, results, classical, histogram, rng, remap)?;
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

    /// Transfer the current device state to the host as f64 (re,im interleaved).
    pub fn dump_state(&self) -> Result<Vec<f64>, String> {
        let mut scratch = Vec::new();
        let buf = self.transfer_to_host(&mut scratch)?;
        Ok(buf.iter().map(|&x| x as f64).collect())
    }

    /// Run a program on the GPU, filling the histogram and observable results.
    pub fn run_program(
        &self,
        prog: &Program,
        shots_override: Option<u32>,
        results: &mut plasma::sim::Results,
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

        if Self::is_batchable(prog) {
            return self.run_pure_batch(prog, shots_override, results, classical, histogram, rng);
        }

        match (shots_override, prog.has_explicit_shot) {
            // CLI override on a SHOT-less program: bare-shot loop.
            (Some(n), false) => {
                for _ in 0..n {
                    self.reset()?;
                    self.exec(prog, 0, prog.len, results, classical, histogram, rng, &[None; 3])?;
                    let s = self.sample_host(rng)?;
                    histogram[s] += 1;
                }
            }
            _ => {
                // Single execution from |0...0>: unlike the batch path and the
                // per-shot loop, this branch previously ran on whatever cuMemAlloc
                // left in the buffer (often zeroed, but never guaranteed). Reset
                // explicitly so measurement programs are deterministic regardless
                // of driver memory state.
                self.reset()?;
                self.exec(prog, 0, prog.len, results, classical, histogram, rng, &[None; 3])?;
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
            let _ = ffi::cuMemFree(self.ops_buf);
            if !self.pinned.is_null() {
                let _ = ffi::cuMemFreeHost(self.pinned.cast());
            }
            let _ = ffi::cuMemFree(self.prob_buf);
            let _ = ffi::cuMemFree(self.prob_scratch);
            let _ = ffi::cuEventDestroy(self.ev_start);
            let _ = ffi::cuEventDestroy(self.ev_stop);
            let _ = ffi::cuModuleUnload(self.module);
            let _ = ffi::cuCtxDestroy(self.ctx);
        }
    }
}
