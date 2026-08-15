// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

use std::env;
use std::fs;
use std::process::exit;
use std::time::Instant;

use plasma::diag::Diag;
use plasma::ir::{MAX_CLASSICAL, Program};
use plasma::lexer::{LexResult, tokenise};
use plasma::parser;
use plasma::rng::Lcg64;
use plasma::sim;

mod gpu;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Bare invocation: print the banner.
    if args.len() == 1 {
        println!("Plasma Quantum Language, Copyright (c) 2026 Faris Alfarhan");
        return;
    }

    if args.len() < 3 || args[1] != "run" {
        eprintln!("Plasma Quantum Language, Copyright (c) 2026 Faris Alfarhan");
        eprintln!("usage: plasma run <file.qs0> [--seed N] [--shots N] [--gpu] [--bench]");
        exit(1);
    }
    let path = &args[2];

    let mut seed: u64 = 42;
    let mut shots: Option<u32> = None;
    let mut gpu = false;
    let mut bench = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(42);
            }
            "--shots" => {
                i += 1;
                shots = args.get(i).and_then(|s| s.parse().ok());
            }
            "--gpu" => gpu = true,
            "--bench" => bench = true,
            other => {
                eprintln!("unknown option: {other}");
                exit(1);
            }
        }
        i += 1;
    }

    // ── Pipeline timing ───────────────────────────────────────────
    let mut phases: Vec<(&'static str, u128)> = Vec::new();
    let t_pipeline = Instant::now();

    let t0 = Instant::now();
    let src = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("plasma: cannot read {path}: {e}");
            exit(1);
        }
    };
    phases.push(("read", t0.elapsed().as_micros()));

    let t0 = Instant::now();
    let mut lex = LexResult::new();
    tokenise(&src, &mut lex);
    if lex.error {
        eprintln!(
            "plasma: {}:{}: lexer error: {}",
            lex.diag.line,
            lex.diag.col,
            lex.diag.message()
        );
        exit(1);
    }
    phases.push(("lex", t0.elapsed().as_micros()));

    let t0 = Instant::now();
    let mut prog = Program::new();
    let mut diag = Diag::new();
    if !parser::parse(&lex, &mut prog, &mut diag) {
        eprintln!("plasma: {}:{}: {}", diag.line, diag.col, diag.message());
        exit(1);
    }
    phases.push(("parse", t0.elapsed().as_micros()));

    let t0 = Instant::now();
    let mut classical = [0u8; MAX_CLASSICAL as usize];
    let mut histogram = vec![0u32; 1 << prog.num_qubits];
    let mut rng = Lcg64::new(seed);

    if gpu {
        let t_init = Instant::now();
        let g = match gpu::Gpu::init(prog.num_qubits) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("plasma: GPU init failed: {e}");
                exit(1);
            }
        };
        phases.push(("gpu_init_total", t_init.elapsed().as_micros()));
        if let Err(e) = g.run_program(&prog, shots, &mut classical, &mut histogram, &mut rng) {
            eprintln!("plasma: GPU run failed: {e}");
            exit(1);
        }
        phases.push(("gpu_cuinit", g.init_cuinit_us.get()));
        phases.push(("gpu_ctx", g.init_ctx_us.get()));
        phases.push(("gpu_jit", g.init_jit_us.get()));
        phases.push(("gpu_alloc", g.init_alloc_us.get()));
        phases.push(("gpu_exec", g.exec_us.get()));
        phases.push(("gpu_exec_real", g.exec_gpu_us.get()));
        phases.push(("gpu_transfer", g.transfer_us.get()));
        phases.push(("gpu_sample", g.sample_us.get()));
        phases.push(("gpu_launches", g.launches.get() as u128));
        // Skip CUDA context teardown: for a one-shot CLI the OS reclaims the
        // context on exit, and cuCtxDestroy costs ~40ms of pure overhead.
        std::mem::forget(g);
    } else {
        match (shots, prog.has_explicit_shot) {
            (Some(n), false) => {
                sim::run_program_shots(&prog, n, &mut classical, &mut histogram, &mut rng);
            }
            _ => {
                sim::run_program(&prog, &mut classical, &mut histogram, &mut rng);
            }
        }
        phases.push(("cpu_run", t0.elapsed().as_micros()));
    }

    let total_us = t_pipeline.elapsed().as_micros();

    print_histogram(&prog, &histogram, seed);

    if bench {
        print_bench(&phases, total_us);
    }
}

fn print_histogram(prog: &Program, histogram: &[u32], seed: u64) {
    let total: u64 = histogram.iter().map(|&c| c as u64).sum();
    println!("Plasma Quantum Language, Copyright (c) 2026 Faris Alfarhan");
    println!("{} qubits, {} shots (seed {seed})", prog.num_qubits, total);
    if total == 0 {
        println!("  (no samples)");
        return;
    }
    for (i, &c) in histogram.iter().enumerate() {
        if c > 0 {
            let mut s = String::new();
            for q in (0..prog.num_qubits).rev() {
                s.push(if (i >> q) & 1 == 1 { '1' } else { '0' });
            }
            println!("  |{s}> : {c}  ({:.2}%)", c as f64 * 100.0 / total as f64);
        }
    }
}

/// Print the raw microsecond benchmark table.
fn print_bench(phases: &[(&'static str, u128)], total_us: u128) {
    println!();
    println!("Plasma benchmark (microseconds)");
    for (name, us) in phases {
        println!("  {name:<14} {us:>12} us");
    }
    println!("  {:<14} {:>12} us", "total", total_us);
}
