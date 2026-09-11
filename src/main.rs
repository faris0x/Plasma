// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

use std::env;
use std::fs;
use std::process::exit;
use std::time::Instant;

use plasma::diag::Diag;
use plasma::fusion;
use plasma::ir::{MAX_CLASSICAL, Program};
use plasma::lexer::{LexResult, tokenise};
use plasma::parser;
use plasma::rng::Lcg64;
use plasma::sim;

mod gpu;

fn print_help() {
    println!("Plasma, a high-level quantum computing language and deterministic quantum circuit simulator");
    println!("Copyright (c) 2026 Faris Alfarhan");
    println!();
    println!("Usage: plasma run <file.qs0> [options]");
    println!();
    println!("Options:");
    println!("  --help                   print this help and exit");
    println!("  --version                print version and license information");
    println!("  --seed N                 deterministic RNG seed (default 42)");
    println!("  --shots N                override the shot count");
    println!("  --gpu                    GPU backend (PTX kernels)");
    println!("  --fuse                   fused gate-stream execution");
    println!("  --stabilizer             exact Clifford (stabilizer) engine");
    println!("  --mps D                  matrix-product-state engine, bond dimension D");
    println!("  --noise-depolarizing P   sampled-Kraus depolarizing noise per gate");
    println!("  --noise-readout P        readout error probability");
    println!("  --noise-amp-damping P    amplitude-damping noise per gate");
    println!("  --noise-phase-damping P  phase-damping noise per gate");
    println!("  --mitigate-readout       invert the readout confusion matrix on the histogram");
    println!("  --precision MODE         auto | f32 | f64 (default f32)");
    println!("  --tolerance T            f32 accuracy bound for --precision auto (default 1e-5)");
    println!("  --bench                  print per-phase timing");
    println!("  --dump-statevector       print the final statevector");
    println!("  --format FORMAT          text | json (default text)");
}

fn print_version() {
    let version = env!("CARGO_PKG_VERSION");
    let branch = option_env!("PLASMA_GIT_BRANCH").unwrap_or("unknown");
    let commit = option_env!("PLASMA_GIT_COMMIT").unwrap_or("unknown");
    println!("Plasma Quantum Language, Copyright (c) 2026 Faris Alfarhan");
    println!("Plasma {version} ({branch}, {commit})");
    println!("This software is licensed under the GNU GPL version 3.0 License only. A copy of the license is available in the project source.");
    println!("For the full license text, please visit:");
    println!("https://github.com/faris0x/Plasma/blob/main/LICENSE");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // --help / -h anywhere prints the full help and exits cleanly.
    if args.iter().skip(1).any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    // --version anywhere prints version and license information, ignoring
    // all other flags.
    if args.iter().skip(1).any(|a| a == "--version") {
        print_version();
        return;
    }

    // Bare invocation: print the banner.
    if args.len() == 1 {
        println!("Plasma Quantum Language, Copyright (c) 2026 Faris Alfarhan");
        println!("Run 'plasma --help' for usage");
        return;
    }

    if args.len() < 3 || args[1] != "run" {
        eprintln!("Plasma Quantum Language, Copyright (c) 2026 Faris Alfarhan");
        eprintln!("usage: plasma run <file.qs0> [--seed N] [--shots N] [--gpu] [--fuse] [--stabilizer]");
        eprintln!("       [--mps D] [--noise-depolarizing P] [--noise-readout P] [--noise-amp-damping P]");
        eprintln!("       [--noise-phase-damping P] [--mitigate-readout] [--precision auto|f32|f64]");
        eprintln!("       [--tolerance T] [--bench] [--dump-statevector] [--format text|json]");
        exit(1);
    }
    let path = &args[2];

    let mut seed: u64 = 42;
    let mut shots: Option<u32> = None;
    let mut gpu = false;
    let mut bench = false;
    let mut dump_statevector = false;
    let mut fuse = false;
    let mut json = false;
    // Precision mode: "f32" (default) keeps the fast GPU path unchanged.
    // "auto" uses the GPU f32 path only when the error model says it is
    // accurate enough, else falls back to the CPU f64 reference. "f64"
    // forces the CPU reference.
    let mut precision: &str = "f32";
    let mut tolerance: f64 = 1e-5;
    let mut noise = sim::NoiseModel::default();
    let mut mitigate_readout = false;
    let mut stabilizer = false;
    let mut mps: Option<usize> = None;
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
            "--dump-statevector" => dump_statevector = true,
            "--fuse" => fuse = true,
            "--noise-depolarizing" => {
                i += 1;
                noise.depolarizing = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            }
            "--noise-readout" => {
                i += 1;
                noise.readout = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            }
            "--noise-amp-damping" => {
                i += 1;
                noise.amp_damping = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            }
            "--noise-phase-damping" => {
                i += 1;
                noise.phase_damping = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            }
            "--mitigate-readout" => mitigate_readout = true,
            "--stabilizer" => stabilizer = true,
            "--mps" => {
                i += 1;
                mps = args.get(i).and_then(|s| s.parse().ok());
            }
            "--precision" => {
                i += 1;
                match args.get(i).map(|s| s.as_str()) {
                    Some("auto") | Some("f32") | Some("f64") => precision = args[i].as_str(),
                    _ => {
                        eprintln!("usage: --precision auto|f32|f64");
                        exit(1);
                    }
                }
            }
            "--tolerance" => {
                i += 1;
                tolerance = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1e-5);
            }
            "--format" => {
                i += 1;
                json = match args.get(i).map(|s| s.as_str()) {
                    Some("json") => true,
                    Some("text") => false,
                    _ => {
                        eprintln!("usage: --format text|json");
                        exit(1);
                    }
                };
            }
            other => {
                eprintln!("unknown option: {other}");
                exit(1);
            }
        }
        i += 1;
    }

    // Pipeline timing
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
    let mut results = sim::Results::default();

    // Adaptive precision: the f32 GPU path is a fast approximation whose
    // amplitude error grows as ~4*eps_f32*sqrt(gates). When the model says
    // it exceeds the tolerance, fall back to the f64 CPU reference.
    let est_err = sim::estimate_f32_amp_error(prog.len);
    match precision {
        "f64" => {
            if gpu {
                eprintln!("plasma: --precision f64 forces the CPU (f64) reference");
                gpu = false;
            }
        }
        "auto" => {
            if gpu && est_err > tolerance {
                eprintln!(
                    "plasma: f32 GPU error estimate {est_err:.1e} exceeds tolerance \
                     {tolerance:.1e}; using the f64 CPU reference (--precision f32 to force)"
                );
                gpu = false;
            }
        }
        _ => {} // "f32" keeps the GPU path as-is
    }
    if noise.is_active() {
        // Sampled-Kraus noise is applied on the (non-fused) CPU reference
        // path only: it is deterministic (seeded RNG) but not yet GPU-aware.
        if gpu {
            eprintln!("plasma: --noise-* forces the CPU (f64) reference");
            gpu = false;
        }
        if fuse {
            eprintln!("plasma: --noise-* forces the non-fused CPU reference");
            fuse = false;
        }
    }
    if let Some(dmax) = mps {
        if gpu {
            eprintln!("plasma: --mps forces the CPU reference");
            gpu = false;
        }
        if fuse {
            eprintln!("plasma: --fuse is ignored with --mps");
            fuse = false;
        }
        if stabilizer {
            eprintln!("plasma: --mps and --stabilizer are mutually exclusive; using --mps");
            stabilizer = false;
        }
        if noise.is_active() {
            eprintln!("plasma: --mps is incompatible with --noise-*");
            exit(1);
        }
        if dmax < 1 {
            eprintln!("plasma: --mps bond dimension must be >= 1");
            exit(1);
        }
    }
    if stabilizer {
        // The adaptive stabilizer engine runs exact Clifford programs on the
        // CPU; it is incompatible with noise, the GPU, and fusion.
        if gpu {
            eprintln!("plasma: --stabilizer forces the CPU reference");
            gpu = false;
        }
        if fuse {
            eprintln!("plasma: --stabilizer forces the non-fused CPU reference");
            fuse = false;
        }
        if noise.is_active() {
            eprintln!("plasma: --stabilizer is incompatible with --noise-*");
            exit(1);
        }
        if !sim::is_clifford_program(&prog) {
            eprintln!("plasma: program is not Clifford-only; falling back to the statevector");
            stabilizer = false;
        }
    }
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
        if let Err(e) = g.run_program(&prog, shots, &mut results, &mut classical, &mut histogram, &mut rng)
        {
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
        if dump_statevector {
            match g.dump_state() {
                Ok(v) => {
                    for (idx, pair) in v.chunks(2).enumerate() {
                        println!("{} {} {}", idx, pair[0], pair[1]);
                    }
                }
                Err(e) => {
                    eprintln!("plasma: statevector dump failed: {e}");
                    exit(1);
                }
            }
        }
        // Skip CUDA context teardown: for a one-shot CLI the OS reclaims the
        // context on exit, and cuCtxDestroy costs ~40ms of pure overhead.
        std::mem::forget(g);
} else if let Some(dmax) = mps {
        if sim::is_mps_suitable(&prog) {
            let trunc = match (shots, prog.has_explicit_shot) {
                (Some(n), false) => sim::run_mps_program_shots(&prog, dmax, n, &mut results, &mut classical, &mut histogram, &mut rng),
                _ => sim::run_mps_program(&prog, dmax, &mut results, &mut classical, &mut histogram, &mut rng),
            };
            if trunc > 1e-6 {
                eprintln!("plasma: MPS truncation error {trunc:.2e} (increase the bond dimension for exact results)");
            }
        } else {
            eprintln!("plasma: program has 3-qubit gates or SAVE_*; falling back to the statevector");
            match (shots, prog.has_explicit_shot) {
                (Some(n), false) => {
                    sim::run_program_shots(&prog, n, &mut results, &mut classical, &mut histogram, &mut rng, noise);
                }
                _ => {
                    sim::run_program(&prog, &mut results, &mut classical, &mut histogram, &mut rng, noise);
                }
            }
        }
    } else if stabilizer {
        match (shots, prog.has_explicit_shot) {
            (Some(n), false) => {
                sim::run_stabilizer_program_shots(&prog, n, &mut results, &mut classical, &mut histogram, &mut rng);
            }
            _ => {
                sim::run_stabilizer_program(&prog, &mut results, &mut classical, &mut histogram, &mut rng);
            }
        }
    } else if fuse {
        let t_fuse = Instant::now();
        let fused = fusion::fuse_program(&prog);
        phases.push(("fuse", t_fuse.elapsed().as_micros()));
        phases.push(("fused_ops", fused.len() as u128));
        phases.push(("orig_ops", prog.len as u128));
        let t0 = Instant::now();
        match (shots, prog.has_explicit_shot) {
            (Some(n), false) => {
                sim::run_fused_program_shots(&prog, &fused, n, &mut results, &mut classical, &mut histogram, &mut rng, noise);
            }
            _ => {
                sim::run_fused_program(&prog, &fused, &mut results, &mut classical, &mut histogram, &mut rng, noise);
            }
        }
        phases.push(("cpu_run", t0.elapsed().as_micros()));
    } else {
        match (shots, prog.has_explicit_shot) {
            (Some(n), false) => {
                sim::run_program_shots(&prog, n, &mut results, &mut classical, &mut histogram, &mut rng, noise);
            }
            _ => {
                sim::run_program(&prog, &mut results, &mut classical, &mut histogram, &mut rng, noise);
            }
        }
        phases.push(("cpu_run", t0.elapsed().as_micros()));
    }

    if mitigate_readout {
        if noise.readout <= 0.0 || noise.readout >= 0.5 {
            eprintln!("plasma: --mitigate-readout requires 0 < --noise-readout < 0.5");
            exit(1);
        }
        sim::mitigate_readout(&mut histogram, prog.num_qubits, noise.readout);
    }

    let total_us = t_pipeline.elapsed().as_micros();

    if json {
        print_json(&prog, &histogram, &results, seed);
    } else {
        print_histogram(&prog, &histogram, seed);
        print_results(&results, prog.num_qubits);
    }

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

/// Print observable results captured by EXPECT / SAVE_* ops.
fn print_results(results: &sim::Results, num_qubits: u8) {
    for (i, v) in results.expectations.iter().enumerate() {
        println!("  expect[{i}] = {v:.9}");
    }
    for (i, v) in results.estimates.iter().enumerate() {
        println!("  estimate[{i}] = {v:.9}");
    }
    for (i, sv) in results.saved_states.iter().enumerate() {
        println!("  state[{i}]:");
        let n = 1usize << num_qubits;
        for k in 0..n {
            let re = sv[2 * k];
            let im = sv[2 * k + 1];
            if re.abs() > 1e-12 || im.abs() > 1e-12 {
                println!("    {k} {re:.9} {im:.9}");
            }
        }
    }
    for (i, amps) in results.saved_amplitudes.iter().enumerate() {
        println!("  amplitudes[{i}]:");
        for (idx, re, im) in amps.iter() {
            if re.abs() > 1e-12 || im.abs() > 1e-12 {
                println!("    {idx} {re:.9} {im:.9}");
            }
        }
    }
    for (i, probs) in results.saved_probs.iter().enumerate() {
        println!("  probabilities[{i}]:");
        for (k, &p) in probs.iter().enumerate() {
            if p > 1e-12 {
                println!("    {k} {p:.9}");
            }
        }
    }
}

/// Print the full result set as JSON (machine-readable).
fn print_json(prog: &Program, histogram: &[u32], results: &sim::Results, seed: u64) {
    let total: u64 = histogram.iter().map(|&c| c as u64).sum();
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!(
        "  \"plasma\": {{\"version\": \"{}\", \"qubits\": {}, \"shots\": {}, \"seed\": {}}},\n",
        env!("CARGO_PKG_VERSION"), prog.num_qubits, total, seed
    ));
    // Histogram: map of basis-string -> count.
    out.push_str("  \"histogram\": {");
    let mut first = true;
    for (i, &c) in histogram.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let mut s = String::new();
        for q in (0..prog.num_qubits).rev() {
            s.push(if (i >> q) & 1 == 1 { '1' } else { '0' });
        }
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&format!(" \"{s}\": {c}"));
    }
    out.push_str(" },\n");
    out.push_str("  \"results\": {\n");
    out.push_str("    \"expectations\": [");
    out.push_str(
        &results
            .expectations
            .iter()
            .map(|v| format!("{v}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str("],\n");
    out.push_str("    \"estimates\": [");
    out.push_str(
        &results
            .estimates
            .iter()
            .map(|v| format!("{v}"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str("],\n");
    out.push_str("    \"statevectors\": [");
    for (i, sv) in results.saved_states.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('[');
        out.push_str(
            &sv.chunks(2)
                .map(|p| format!("[{}, {}]", p[0], p[1]))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push(']');
    }
    out.push_str("],\n");
    out.push_str("    \"amplitudes\": [");
    for (i, amps) in results.saved_amplitudes.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('[');
        out.push_str(
            &amps
                .iter()
                .map(|(idx, re, im)| format!("[{idx}, {re}, {im}]"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push(']');
    }
    out.push_str("],\n");
    out.push_str("    \"probabilities\": [");
    for (i, probs) in results.saved_probs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('[');
        out.push_str(
            &probs
                .iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push(']');
    }
    out.push_str("]\n");
    out.push_str("  }\n");
    out.push('}');
    println!("{out}");
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
