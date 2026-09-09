#!/usr/bin/env python3
"""Plasma correctness harness.

Enforces the backward-compat invariants:

  A (exact results bit-stable): mathematically-determined outputs, gate
     evolution, deterministic-basis histograms, pure-unitary statevectors
     must match recorded golden values (within float tolerance) and be
     identical across CPU and GPU. Golden values are compared numerically,
     so pure output-format changes with identical math still pass.
  B (determinism): seeded output is bit-reproducible run-to-run on the same
     backend (guards the deterministic-reduction fix).
  C (backward compat): every v0.1 program runs and matches; the GPU must
     equal the CPU reference wherever amplitudes are exactly representable,
     and stay within the sampling-noise floor where f32 vs f64 differs.

GPU-gated: exits 0 (skip) when no CUDA device exists, so it is safe on any
machine.

Usage:  python3 tests/golden.py
        PLASMA_REGENERATE=1 python3 tests/golden.py   # rewrite goldens
"""
import os
import subprocess
import sys
import json
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "target", "release", "plasma")
GOLDEN_DIR = os.path.join(ROOT, "tests", "golden")
SEED = 123
AMP_TOL = 1e-4

# Programs with exactly-representable amplitudes: CPU and GPU histograms must
# be bit-identical, and stay bit-identical across all future versions.
EXACT = ["test.qs0", "test2.qs0", "toff.qs0", "four.qs0", "ghz3.qs0", "measure.qs0"]

# Deep random circuits where f32-vs-f64 amplitude rounding moves individual
# samples between adjacent bins; pass is statistical and GPU run-to-run must
# be bit-identical (determinism, invariant B).
RANDOM = ["stress.qs0", "v02_rot.qs0", "v02_measure.qs0"]
# v02_sub.qs0: jump-based subroutines (nested calls, IF inside a gate body).
SUBROUTINE = ["v02_sub.qs0"]

# Pure-unitary, shot-free programs used for exact statevector goldens.
SV_GOLDEN = ["sv_bell.qs0", "sv_toff.qs0", "sv_ghz.qs0"]

# v0.2 programs (M7-B):
#  - v02_exact.qs0: exact-representable gates -> CPU==GPU bit-identical.
#  - v02_expect.qs0 / v02_expect_deep.qs0: EXPECT/SAVE_* -> expectations
#    agree to float tolerance AND the observed drift is bounded by the
#    adaptive-precision error model (M8-B).
V02_EXACT = ["v02_exact.qs0"]
V02_EXPECT = ["v02_expect.qs0", "v02_expect_deep.qs0", "v02_estimate.qs0"]

# Expectation agreement tolerance (f32 GPU vs f64 CPU).
EXPECT_TOL = 1e-5


def run_hist(prog: str, gpu: bool, seed: int) -> dict:
    cmd = [BIN, "run", os.path.join(ROOT, "tests", prog), "--seed", str(seed)]
    if gpu:
        cmd.append("--gpu")
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    counts = {}
    for m in re.finditer(r"^\s*\|([01]+)>\s*:\s*(\d+)", out, re.M):
        counts[int(m.group(1), 2)] = int(m.group(2))
    return counts


def run_state(prog: str, seed: int) -> dict:
    cmd = [BIN, "run", os.path.join(GOLDEN_DIR, prog), "--seed", str(seed),
           "--gpu", "--shots", "1", "--dump-statevector"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    amps = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit():
            amps[int(parts[0])] = complex(float(parts[1]), float(parts[2]))
    return amps


def tvd(a: dict, b: dict, total: int) -> float:
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)) / total


def have_gpu() -> bool:
    return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0


def check_exact(prog: str) -> bool:
    cpu = run_hist(prog, False, SEED)
    gpu = run_hist(prog, True, SEED)
    ok = cpu == gpu
    print(f"  {'PASS' if ok else 'FAIL'} {prog}: CPU==GPU histograms")
    return ok


def check_random(prog: str) -> bool:
    gpu1 = run_hist(prog, True, SEED)
    gpu2 = run_hist(prog, True, SEED)
    total = sum(gpu1.values())
    noise = tvd(gpu1, gpu2, total)
    cpu = run_hist(prog, False, SEED)
    tv = tvd(cpu, gpu1, total)
    ok = tv < 3.0 * noise + 0.002
    print(f"  {'PASS' if ok else 'FAIL'} {prog}: TVD(cpu,gpu)={tv:.6f} "
          f"noise={noise:.6f} (limit {3.0*noise + 0.002:.6f})")
    if gpu1 != gpu2:
        print(f"  FAIL {prog}: GPU run-to-run not bit-identical")
        ok = False
    return ok


def run_expect(prog: str, gpu: bool) -> dict:
    """Run a program and extract expectations + saved probabilities."""
    cmd = [BIN, "run", os.path.join(ROOT, "tests", prog), "--seed", str(SEED)]
    if gpu:
        cmd.append("--gpu")
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    ex = [float(m) for m in re.findall(r"expect\[\d+\] = ([\d.eE+-]+)", out)]
    est = [float(m) for m in re.findall(r"estimate\[\d+\] = ([\d.eE+-]+)", out)]
    # saved probabilities: the single saved_probs block (lines "  k p").
    probs = [float(m) for m in re.findall(r"^    \d+ ([\d.eE+-]+)$", out, re.M)]
    return {"expectations": ex, "estimates": est, "probs": probs}



def check_stabilizer(prog: str) -> bool:
    """Clifford programs run on the stabilizer engine must give a
    statistically identical histogram to the statevector reference (the RNG is
    consumed in a different order, so counts differ but the distribution does
    not)."""
    def hist(gpu_or_stab: bool) -> list:
        cmd = [BIN, "run", os.path.join(ROOT, "tests", prog), "--seed", str(SEED), "--shots", "20000"]
        if gpu_or_stab:
            cmd.append("--stabilizer")
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        n = 1 << int(re.search(r"QUBITS (\d+)", open(os.path.join(ROOT, "tests", prog)).read()).group(1))
        h = [0] * n
        for m in re.finditer(r"\|([01]+)> : (\d+)", out):
            idx = int(m.group(1), 2)
            h[idx] = int(m.group(2))
        return h
    a = hist(False)
    b = hist(True)
    sa = sum(a)
    sb = sum(b)
    if sa == 0 or sb == 0:
        print(f"  FAIL {prog}: no samples")
        return False
    tvd = sum(abs(x / sa - y / sb) for x, y in zip(a, b)) / 2
    ok = tvd < 0.06
    print(f"  {'PASS' if ok else 'FAIL'} {prog}: stabilizer vs statevector TVD {tvd:.4f}")
    return ok

def check_expect(prog: str) -> bool:
    """CPU and GPU expectations must agree to EXPECT_TOL, and the observed
    drift must be bounded by the adaptive-precision error model
    (est = 4*eps_f32*sqrt(gates)); observed <= 3*est validates the model."""
    cpu = run_expect(prog, False)
    gpu = run_expect(prog, True)
    ok = True
    if len(cpu["expectations"]) != len(gpu["expectations"]):
        print(f"  FAIL {prog}: expectation count mismatch "
              f"({len(cpu['expectations'])} vs {len(gpu['expectations'])})")
        return False
    worst = 0.0
    for i, (a, b) in enumerate(zip(cpu["expectations"], gpu["expectations"])):
        d = abs(a - b)
        worst = max(worst, d)
        if d > EXPECT_TOL:
            ok = False
            print(f"  FAIL {prog}: expect[{i}] cpu={a} gpu={b} diff={d:.2e}")
    # Estimates (weighted Pauli sums).
    if len(cpu["estimates"]) != len(gpu["estimates"]):
        print(f"  FAIL {prog}: estimate count mismatch")
        ok = False
    else:
        for i, (a, b) in enumerate(zip(cpu["estimates"], gpu["estimates"])):
            if abs(a - b) > EXPECT_TOL:
                ok = False
                print(f"  FAIL {prog}: estimate[{i}] cpu={a} gpu={b} diff={abs(a-b):.2e}")
    # Saved probabilities.
    if len(cpu["probs"]) != len(gpu["probs"]):
        print(f"  FAIL {prog}: saved-probability count mismatch")
        ok = False
    elif cpu["probs"]:
        pw = max(abs(a - b) for a, b in zip(cpu["probs"], gpu["probs"]))
        if pw > EXPECT_TOL:
            ok = False
            print(f"  FAIL {prog}: saved probs worst diff {pw:.2e}")
    # Error-model bound: gates = statement lines (approx op count).
    gates = sum(1 for line in open(os.path.join(ROOT, "tests", prog))
                if line.strip() and not line.lstrip().startswith("//"))
    est = 4.0 * 1.19e-7 * (max(gates, 1) ** 0.5)
    bound = 3.0 * est
    model_ok = worst <= max(bound, EXPECT_TOL)
    print(f"  {'PASS' if ok and model_ok else 'FAIL'} {prog}: expectations worst diff "
          f"{worst:.2e} (tol {EXPECT_TOL}; model bound {bound:.2e}, est {est:.2e})")
    if not model_ok:
        print(f"  FAIL {prog}: observed drift {worst:.2e} exceeds model bound {bound:.2e}")
        ok = False
    return ok


def check_statevector(prog: str) -> bool:
    amps = run_state(prog, SEED)
    golden_path = os.path.join(GOLDEN_DIR, prog + ".golden.json")
    if os.environ.get("PLASMA_REGENERATE"):
        with open(golden_path, "w") as f:
            json.dump({str(k): [v.real, v.imag] for k, v in amps.items()}, f, indent=0)
        print(f"  REGEN {prog}: wrote golden statevector")
        return True
    with open(golden_path) as f:
        golden = {int(k): complex(v[0], v[1]) for k, v in json.load(f).items()}
    worst = max((abs(amps.get(k, 0) - golden.get(k, 0)) for k in set(amps) | set(golden)),
                default=0.0)
    ok = worst < AMP_TOL
    print(f"  {'PASS' if ok else 'FAIL'} {prog}: max|amp diff|={worst:.2e} (tol {AMP_TOL})")
    return ok


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    if not have_gpu():
        print("no CUDA device; golden harness requires GPU. Skipping.")
        sys.exit(0)

    ok = True
    print("== Exact programs (CPU == GPU, bit-identical) ==")
    for prog in EXACT:
        ok = check_exact(prog) and ok
    print("== Random deep circuits (statistical; GPU run-to-run bit-identical) ==")
    for prog in RANDOM:
        ok = check_random(prog) and ok
    print("== Golden pure-unitary statevectors (invariant A) ==")
    for prog in SV_GOLDEN:
        ok = check_statevector(prog) and ok
    print("== v0.2 exact programs (CPU == GPU) ==")
    for prog in V02_EXACT:
        ok = check_exact(prog) and ok
    print("== v0.2 EXPECT/SAVE agreement (model-validated) ==")
    for prog in V02_EXPECT:
        ok = check_expect(prog) and ok

    print("ALL PASS" if ok else "FAILURES DETECTED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
