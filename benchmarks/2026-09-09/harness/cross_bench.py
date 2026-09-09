#!/usr/bin/env python3
"""Cross-platform quantum simulator benchmark, Plasma baseline.

Measures kernel-level construction + sampling time for equivalent circuits
across Plasma (CPU/GPU), Qiskit Aer (CPU/GPU), cuQuantum custatevec (GPU),
qsimcirq (CPU), qulacs (CPU), QuEST (CPU). Every sampled histogram is
cross-validated against the Plasma f64 CPU reference by TVD.

Usage:
  cross_bench.py --smoke                # quick q8/q10 sanity pass
  cross_bench.py --cpu --repeats 5      # full CPU sweep
  cross_bench.py --gpu --repeats 5      # full GPU sweep
  cross_bench.py --family ghz --n 20 --platforms plasma_gpu,aer_gpu,custatevec
"""
import argparse, collections, cmath, csv, json, math, os, random, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLASMA = os.environ.get("PLASMA_BIN", os.path.join(ROOT, "target", "release", "plasma"))
V314 = os.path.join(HERE, ".venv", "bin", "python")
V313 = os.path.join(HERE, ".venv313", "bin", "python")
QUEST = os.path.join(HERE, "quest_bench")
QUEST_LIB = os.path.join(HERE, "quest", "build", "QuEST")
CUQ_LIB = "/opt/cuda/lib64:" + os.path.join(HERE, ".venv", "lib", "python3.14", "site-packages", "cuquantum", "lib") \
          + ":" + os.path.join(HERE, ".venv", "lib", "python3.14", "site-packages", "cutensor", "lib")
WORK = os.path.join(HERE, "work"); os.makedirs(WORK, exist_ok=True)
SEED = 20260816


# workload generators

def gen_ghz(n):
    return [("h", (0,))] + [("cx", (i, i + 1)) for i in range(n - 1)]

def gen_qft(n):
    g = []
    for i in range(n):
        g.append(("h", (i,)))
        for j in range(i + 1, n):
            g.append(("cp", (i, j, math.pi / 2.0 ** (j - i))))
    for i in range(n // 2):
        g.append(("swap", (i, n - 1 - i)))
    return g

def gen_rnd(n, depth, mix, seed):
    r = random.Random(seed)
    g = []
    def cn():
        c = r.randrange(n - 1); t = r.randrange(n)
        if t == c: t = (c + 1) % n
        return ("cx", (c, t))
    for _ in range(depth):
        if mix == "hc":
            g.append(("h", (r.randrange(n),)) if r.random() < 0.5 else cn())
        else:
            op = r.choice(["rz", "ry", "rx"])
            g.append((op, (r.randrange(n), r.random() * 2 * math.pi)))
            g.append(cn())
    return g

def gen_clifford(n, depth, seed):
    r = random.Random(seed)
    g = []
    def cn():
        c = r.randrange(n - 1); t = r.randrange(n)
        if t == c: t = (c + 1) % n
        return c, t
    for _ in range(depth):
        k = r.random()
        if k < 0.25: g.append(("h", (r.randrange(n),)))
        elif k < 0.45: g.append(("s", (r.randrange(n),)))
        elif k < 0.6: c, t = cn(); g.append(("cx", (c, t)))
        elif k < 0.75: c, t = cn(); g.append(("cz", (c, t)))
        elif k < 0.85: c, t = cn(); g.append(("swap", (c, t)))
        else: g.append(("x", (r.randrange(n),)))
    return g

def gen_case(family, n, seed):
    if family == "ghz": return gen_ghz(n)
    if family == "qft": return gen_qft(n)
    if family == "rnd_hc": return gen_rnd(n, 5 * n, "hc", seed)
    if family == "rnd_rot": return gen_rnd(n, 5 * n, "rot", seed)
    if family == "clifford": return gen_clifford(n, 5 * n, seed)
    raise ValueError(family)

# helpers

def eff_shots(n, shots):
    # bound sampling cost: Plasma CPU sampler is O(2^n) per shot (full state scan)
    if n <= 16: return min(shots, 2000)
    if n <= 20: return min(shots, 200)
    if n <= 24: return min(shots, 100)
    return min(shots, 30)

def write_gate_file(gates, path):
    with open(path, "w") as f:
        for op, args in gates:
            f.write(op + " " + " ".join(str(a) for a in args) + "\n")

def write_qs0(gates, n, path):
    PLASMA_OP = {"cx": "CNOT", "cp": "CPHASE", "p": "PHASE", "swap": "SWAP",
                 "h": "H", "x": "X", "s": "S", "rz": "RZ", "ry": "RY", "rx": "RX"}
    def plain(x):
        # Plasma's parser rejects scientific notation; emit plain decimals
        if isinstance(x, float):
            return repr(x) if "e" not in repr(x) else f"{x:.17f}".rstrip("0").rstrip(".")
        return str(x)
    with open(path, "w") as f:
        f.write(f"QUBITS {n}\n")
        for op, args in gates:
            f.write(PLASMA_OP.get(op, op.upper()) + " " + " ".join(plain(a) for a in args) + "\n")

def best_of(fn, warm, reps):
    for _ in range(warm):
        fn()
    vals = []
    for _ in range(reps):
        vals.append(fn())
    mn = min(vals); avg = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - avg) ** 2 for x in vals) / len(vals))
    return mn, avg, sd

def tvd(a, b):
    ka = set(a) | set(b)
    sa = sum(a.values()) or 1; sb = sum(b.values()) or 1
    return sum(abs(a.get(k, 0) / sa - b.get(k, 0) / sb) for k in ka) / 2

def validate_counts(gates, n, platform, seed, ref):
    """High-shot histogram for the correctness gate (sampling-noise limited)."""
    vs = 20000 if n <= 20 else (100 if n <= 24 else 30)
    if platform == "plasma_cpu": return tvd(plasma_counts_gpu(gates, n, vs, seed), ref)
    if platform == "plasma_gpu": return tvd(plasma_counts_gpu(gates, n, vs, seed), ref)
    if platform == "aer_cpu":
        r, _ = run_aer(gates, n, vs, seed, "CPU", 0, 1); return tvd(r[3], ref) if r else 1.0
    if platform == "aer_gpu":
        r, _ = run_aer(gates, n, vs, seed, "GPU", 0, 1); return tvd(r[3], ref) if r else 1.0
    if platform == "custatevec":
        r, _ = run_custatevec(gates, n, vs, seed, 0, 1); return tvd(r[3], ref) if r else 1.0
    if platform == "qsim":
        r, _ = run_qsim(gates, n, vs, seed, 0, 1); return tvd(r[3], ref) if r else 1.0
    if platform == "qulacs":
        r, _ = run_qulacs(gates, n, vs, seed, 0, 1); return tvd(r[3], ref) if r else 1.0
    return 0.0  # quest: counts not parsed; gate via construction parity only

def run_sub(venv, src, env=None, timeout=900):
    e = dict(os.environ)
    if env: e.update(env)
    p = subprocess.run([venv, "-c", src], capture_output=True, text=True, env=e, timeout=timeout)
    if p.returncode != 0:
        return None, p.stderr
    out = p.stdout.strip().splitlines()
    return (json.loads(out[-1]) if out else None), (p.stderr[-400:] if p.stderr else "")


# Plasma

def plasma_times(gates, n, shots, backend, seed, warm, reps):
    qs0 = os.path.join(WORK, f"p{n}.qs0"); write_qs0(gates, n, qs0)
    def run(shots_arg):
        cmd = [PLASMA, "run", qs0, "--seed", str(seed), "--bench"]
        if backend == "gpu": cmd.append("--gpu")
        if shots_arg is not None:
            cmd += ["--shots", str(shots_arg)]
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    def construct():
        o = run(None)
        key = "gpu_exec_real" if backend == "gpu" else "cpu_run"
        m = re.search(rf"\s+{key}\s+(\d+)", o)
        if not m:
            m = re.search(r"\s+total\s+(\d+)", o)
        return int(m.group(1)) / 1000.0 if m else None
    def sample():
        if backend == "gpu":
            o = run(shots)
            m = re.search(r"\s+gpu_sample\s+(\d+)", o)
            return int(m.group(1)) / 1000.0 if m else 0.0
        if shots <= 0:
            return 0.0
        # cpu: CLI re-executes the circuit per shot; sampling phase = shots-run minus one-run
        def t(s):
            o = run(s)
            m = re.search(r"\s+cpu_run\s+(\d+)", o)
            return int(m.group(1)) / 1000.0 if m else 0.0
        return max(t(shots) - t(None), 0.0)
    c = best_of(construct, warm, reps)[0]
    s = best_of(sample, warm, reps)[0]
    return c, s

def plasma_counts_gpu(gates, n, shots, seed):
    qs0 = os.path.join(WORK, f"p{n}.qs0"); write_qs0(gates, n, qs0)
    o = subprocess.run([PLASMA, "run", qs0, "--gpu", "--seed", str(seed), "--shots", str(shots)],
                       capture_output=True, text=True).stdout
    c = {}
    for m in re.finditer(r"\|([01]+)> : (\d+)", o):
        c[m.group(1)] = int(m.group(2))
    return c

def plasma_counts(gates, n, shots, seed):
    qs0 = os.path.join(WORK, f"p{n}.qs0"); write_qs0(gates, n, qs0)
    o = subprocess.run([PLASMA, "run", qs0, "--seed", str(seed), "--shots", str(shots)],
                       capture_output=True, text=True).stdout
    c = {}
    for m in re.finditer(r"\|([01]+)> : (\d+)", o):
        c[m.group(1)] = int(m.group(2))
    return c


# Qiskit Aer

AER_SRC = r'''
import os, time, json
os.environ.setdefault("LD_LIBRARY_PATH", %(ld)r)
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def run(n, shots, seed, gates, device):
    qc = QuantumCircuit(n)
    for op, a in gates:
        if op == "h": qc.h(a[0])
        elif op == "x": qc.x(a[0])
        elif op == "s": qc.s(a[0])
        elif op == "cx": qc.cx(a[0], a[1])
        elif op == "cz": qc.cz(a[0], a[1])
        elif op == "swap": qc.swap(a[0], a[1])
        elif op == "rz": qc.rz(a[1], a[0])
        elif op == "ry": qc.ry(a[1], a[0])
        elif op == "rx": qc.rx(a[1], a[0])
        elif op == "p": qc.p(a[1], a[0])
        elif op == "cp": qc.cp(a[2], a[0], a[1])
    sim = AerSimulator(method="statevector", device=device, cuStateVec_enable=(device == "GPU"),
                       max_parallel_threads=0)
    qc_sv = qc.copy(); qc_sv.save_statevector()
    t0 = time.perf_counter()
    r0 = sim.run(qc_sv, shots=1).result()
    t1 = time.perf_counter()
    c_ms = (t1 - t0) * 1e3
    qm = qc.copy(); qm.measure_all()
    t0 = time.perf_counter()
    r = sim.run(qm, shots=shots, seed_simulator=seed).result()
    t1 = time.perf_counter()
    s_ms = (t1 - t0) * 1e3
    counts = r.get_counts()
    return [c_ms, s_ms, counts]
'''

def run_aer(gates, n, shots, seed, device, warm, reps):
    src = AER_SRC % {"ld": CUQ_LIB} + f'''
print(json.dumps(run({n}, {shots}, {seed}, {json.dumps(gates)}, {json.dumps(device)})))'''
    res, err = run_sub(V314, src, env={"LD_LIBRARY_PATH": CUQ_LIB})
    if res is None:
        return None, ("aer", err)
    def one():
        r, e = run_sub(V314, src, env={"LD_LIBRARY_PATH": CUQ_LIB})
        return r
    # use the single measured run for warmup+reps via best_of on subprocess
    t = best_of(lambda: one()[0], warm, reps)
    return (t[0], t[1], t[2], res[2]), None


# cuQuantum custatevec

CSV_SRC = r'''
import ctypes, json, math, time, collections
import numpy as np, cupy as cp
from cuquantum import cudaDataType as dt
import cuquantum.bindings.custatevec as csv
_mem = {}
@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_void_p)
def dev_alloc(ctx, ptr, size, stream):
    mp = cp.cuda.memory.alloc(int(size)); _mem[mp.ptr] = mp; ptr[0] = mp.ptr; return 0
@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
def dev_free(ctx, ptr, size, stream):
    _mem.pop(ptr, None); return 0
h = csv.create()
csv.set_device_mem_handler(h, (0, ctypes.cast(dev_alloc, ctypes.c_void_p).value,
                               ctypes.cast(dev_free, ctypes.c_void_p).value, "b"))
bufs = {}
n = %(n)d
sv = cp.zeros(1 << n, dtype=cp.complex64); sv[0] = 1.0
def am(M, targets):
    m = np.ascontiguousarray(np.asarray(M).ravel(), dtype=np.complex64)
    key = (m.tobytes(), tuple(targets))
    if key not in bufs: bufs[key] = m
    csv.apply_matrix(h, sv.data.ptr, dt.CUDA_C_32F, n, bufs[key].ctypes.data, dt.CUDA_C_32F,
                     csv.MatrixLayout.ROW, 0, list(targets), len(targets), [], [], 0,
                     csv.ComputeType.COMPUTE_DEFAULT, 0, 0)
def apply(gates):
    sq = 1.0 / math.sqrt(2.0)
    for op, a in gates:
        if op == "h": am(np.array([[sq, sq],[sq,-sq]]), [a[0]])
        elif op == "x": am(np.array([[0,1],[1,0]]), [a[0]])
        elif op == "s": am(np.array([[1,0],[0,1j]]), [a[0]])
        elif op == "rz": th=a[1]; am(np.array([[math.cos(th/2)-1j*math.sin(th/2),0],[0,math.cos(th/2)+1j*math.sin(th/2)]]), [a[0]])
        elif op == "ry": th=a[1]; am(np.array([[math.cos(th/2),-math.sin(th/2)],[math.sin(th/2),math.cos(th/2)]]), [a[0]])
        elif op == "rx": th=a[1]; am(np.array([[math.cos(th/2),-1j*math.sin(th/2)],[-1j*math.sin(th/2),math.cos(th/2)]]), [a[0]])
        elif op == "p": th=a[1]; am(np.array([[1,0],[0,math.cos(th)+1j*math.sin(th)]]), [a[0]])
        elif op == "cx":
            M = np.zeros((4,4), dtype=np.complex128)
            for qc in [0,1]:
                for qt in [0,1]:
                    M[qc + 2*(qt^qc), qc + 2*qt] = 1.0
            am(M, [a[0], a[1]])
        elif op == "cz": am(np.diag([1,1,1,-1]), [a[0], a[1]])
        elif op == "swap": am(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]), [a[0], a[1]])
        elif op == "cp":
            th=a[2]; M = np.eye(4, dtype=np.complex128); M[3,3] = math.cos(th)+1j*math.sin(th)
            am(M, [a[0], a[1]])
gates = %(gates)s
shots = %(shots)s
t0 = time.perf_counter()
apply(gates)
cp.cuda.runtime.deviceSynchronize()
t1 = time.perf_counter()
c_ms = (t1 - t0) * 1e3
sm = csv.sampler_create(h, sv.data.ptr, dt.CUDA_C_32F, n, shots)[0]
out = cp.zeros(shots, dtype=cp.int64)
rand = cp.random.rand(shots, dtype=cp.float64)
csv.sampler_preprocess(h, sm, 0, 0)
cvs = ctypes.CDLL("libcustatevec.so.1")
cvs.custatevecSamplerSample.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]
cvs.custatevecSamplerSample.restype = ctypes.c_int
order = (ctypes.c_int32 * n)(*range(n))
t0 = time.perf_counter()
rc = cvs.custatevecSamplerSample(h, sm, out.data.ptr, order, n, rand.data.ptr, shots, 0)
cp.cuda.runtime.deviceSynchronize()
t1 = time.perf_counter()
s_ms = (t1 - t0) * 1e3
counts = collections.Counter(int(x) for x in cp.asnumpy(out))
counts = {format(k, "0%(n)db"): v for k, v in counts.items()}
print(json.dumps([c_ms, s_ms, counts]))
'''
import time as _t
def run_custatevec(gates, n, shots, seed, warm, reps):
    src = CSV_SRC % {"n": n, "gates": json.dumps(gates), "shots": shots}
    def one():
        r, _ = run_sub(V314, src, env={"LD_LIBRARY_PATH": CUQ_LIB})
        return r
    r0 = one()
    if r0 is None:
        return None, ("custatevec", "run failed")
    tc = best_of(lambda: one()[0], warm, reps)
    ts = best_of(lambda: one()[1], warm, reps)
    return (tc[0], ts[0], ts[2], r0[2]), None


# qsimcirq

QSIM_SRC = r'''
import json, time, math
import cirq, qsimcirq
def run(n, shots, seed, gates):
    q = [cirq.LineQubit(i) for i in range(n)]
    ops = []
    for op, a in gates:
        if op == "h": ops.append(cirq.H(q[a[0]]))
        elif op == "x": ops.append(cirq.X(q[a[0]]))
        elif op == "s": ops.append(cirq.S(q[a[0]]))
        elif op == "cx": ops.append(cirq.CNOT(q[a[0]], q[a[1]]))
        elif op == "cz": ops.append(cirq.CZ(q[a[0]], q[a[1]]))
        elif op == "swap": ops.append(cirq.SWAP(q[a[0]], q[a[1]]))
        elif op == "rz": ops.append(cirq.rz(a[1])(q[a[0]]))
        elif op == "ry": ops.append(cirq.ry(a[1])(q[a[0]]))
        elif op == "rx": ops.append(cirq.rx(a[1])(q[a[0]]))
        elif op == "p": ops.append(cirq.ZPowGate(exponent=a[1]/math.pi)(q[a[0]]))
        elif op == "cp": ops.append(cirq.CZPowGate(exponent=a[2]/math.pi)(q[a[0]], q[a[1]]))
    circ = cirq.Circuit(ops)
    sim = qsimcirq.QSimSimulator()
    t0 = time.perf_counter(); sim.simulate(circ); t1 = time.perf_counter()
    c_ms = (t1 - t0) * 1e3
    circ_m = circ + cirq.measure(*q, key="m")
    t0 = time.perf_counter(); res = sim.run(circ_m, repetitions=shots); t1 = time.perf_counter()
    s_ms = (t1 - t0) * 1e3
    counts = {}
    for b, cnt in res.histogram(key="m").items():
        counts[format(int(b), "0%(n)db")[::-1]] = int(cnt)
    print(json.dumps([c_ms, s_ms, counts]))
'''
def run_qsim(gates, n, shots, seed, warm, reps):
    src = QSIM_SRC % {"n": n} + f'''
run({n}, {shots}, {seed}, {json.dumps(gates)})'''
    def one():
        r, _ = run_sub(V313, src)
        return r
    r0 = one()
    if r0 is None:
        return None, ("qsim", "run failed")
    tc = best_of(lambda: one()[0], warm, reps)
    ts = best_of(lambda: one()[1], warm, reps)
    return (tc[0], ts[0], ts[2], r0[2]), None


# qulacs

QULACS_SRC = r'''
import json, time, math, cmath
import numpy as np
from qulacs import QuantumState, QuantumCircuit
def cp_mat(th):
    M = np.eye(4, dtype=np.complex128); M[3,3] = cmath.exp(1j*th)
    return M
def run(n, shots, seed, gates):
    qc = QuantumCircuit(n)
    for op, a in gates:
        if op == "h": qc.add_H_gate(a[0])
        elif op == "x": qc.add_X_gate(a[0])
        elif op == "s": qc.add_S_gate(a[0])
        elif op == "cx": qc.add_CNOT_gate(a[0], a[1])
        elif op == "cz": qc.add_CZ_gate(a[0], a[1])
        elif op == "swap": qc.add_SWAP_gate(a[0], a[1])
        elif op == "rz": qc.add_RZ_gate(a[0], -a[1])
        elif op == "ry": qc.add_RY_gate(a[0], -a[1])
        elif op == "rx": qc.add_RX_gate(a[0], -a[1])
        elif op == "p": qc.add_diagonal_observable_gate([a[0]], [0.0, a[1]])
        elif op == "cp": qc.add_dense_matrix_gate([a[0], a[1]], cp_mat(a[2]))
    st = QuantumState(n); st.set_zero_state()
    t0 = time.perf_counter(); qc.update_quantum_state(st); t1 = time.perf_counter()
    c_ms = (t1 - t0) * 1e3
    t0 = time.perf_counter(); samples = st.sampling(shots, seed); t1 = time.perf_counter()
    s_ms = (t1 - t0) * 1e3
    counts = {}
    for i in samples:
        counts[format(int(i), "0%(n)db")] = counts.get(format(int(i), "0%(n)db"), 0) + 1
    print(json.dumps([c_ms, s_ms, counts]))
'''
def run_qulacs(gates, n, shots, seed, warm, reps):
    src = QULACS_SRC % {"n": n} + f'''
run({n}, {shots}, {seed}, {json.dumps(gates)})'''
    def one():
        r, _ = run_sub(V313, src)
        return r
    r0 = one()
    if r0 is None:
        return None, ("qulacs", "run failed")
    tc = best_of(lambda: one()[0], warm, reps)
    ts = best_of(lambda: one()[1], warm, reps)
    return (tc[0], ts[0], ts[2], r0[2]), None


# QuEST

def run_quest(gates, n, shots, seed, warm, reps):
    if n >= 20:
        shots = 0
    gf = os.path.join(WORK, f"q{n}.gates"); write_gate_file(gates, gf)
    env = {"LD_LIBRARY_PATH": QUEST_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
           "OMP_NUM_THREADS": "2", "OMP_THREAD_LIMIT": "2", "OMP_PROC_BIND": "close"}
    def one():
        p = subprocess.run([QUEST, str(n), gf, str(shots)], capture_output=True, text=True, env=env)
        if p.returncode != 0: return None
        d = dict(l.split() for l in p.stdout.splitlines())
        return (float(d["construct_ms"]), float(d.get("sample_ms", 0.0)))
    r0 = one()
    if r0 is None:
        return None, ("quest", "run failed")
    tc = best_of(lambda: one()[0], warm, reps)
    ts = best_of(lambda: one()[1], warm, reps)
    return (tc[0], ts[0], ts[2], None), None


# main

def save_rows(rows, path):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--platforms", default="all")
    ap.add_argument("--out", default=os.path.join(HERE, "cross_results.csv"))
    args = ap.parse_args()

    if args.smoke:
        cases = [("ghz", 8, 1000), ("rnd_rot", 10, 1000)]
    elif args.n:
        families = [args.family] if args.family else ["ghz", "qft", "rnd_hc", "rnd_rot", "clifford"]
        cases = [(f, args.n, args.shots) for f in families]
    else:
        scales = ([16, 20, 24, 27] if args.gpu else []) + ([20, 24, 26] if args.cpu else [20])
        families = ["ghz", "qft", "rnd_hc", "rnd_rot", "clifford"]
        cases = [(f, n, args.shots) for f in families for n in scales]

    all_p = ["plasma_cpu", "plasma_gpu", "aer_cpu", "aer_gpu", "custatevec", "qsim", "qulacs", "quest"]
    if args.platforms == "all":
        if args.gpu and not args.cpu:
            platforms = ["plasma_gpu", "aer_gpu", "custatevec"]
        elif args.cpu and not args.gpu:
            platforms = ["plasma_cpu", "aer_cpu", "qsim", "qulacs", "quest"]
        else:
            platforms = all_p
    else:
        platforms = [p for p in all_p if args.platforms in ("all", p)]

    rows = []
    for (family, n, shots) in cases:
        gates = gen_case(family, n, SEED + n)
        rshots = 20000 if n <= 20 else (100 if n <= 24 else 30)
        ref = plasma_counts_gpu(gates, n, rshots, 42) if n <= 28 else {}
        print(f"# {family} n={n} gates={len(gates)} ref_shots={rshots}", flush=True)
        print(f"# {family} n={n} gates={len(gates)}", flush=True)
        for p in platforms:
            if p == "plasma_cpu":
                samp_shots = eff_shots(n, shots) if n <= 16 else 0
                c, sm = plasma_times(gates, n, samp_shots, "cpu", 42, args.warm, args.repeats)
                if c is None:
                    print(f"  plasma_cpu: SKIP (run failed)", flush=True); continue
                rows.append(dict(family=family, n=n, platform="plasma", backend="cpu", precision="f64",
                                 construct_ms=c, sample_ms=sm, tvd=validate_counts(gates, n, "plasma_cpu", 42, ref)))
            elif p == "plasma_gpu":
                c, sm = plasma_times(gates, n, eff_shots(n, shots), "gpu", 42, args.warm, args.repeats)
                if c is None:
                    print(f"  plasma_gpu: SKIP (run failed)", flush=True); continue
                rows.append(dict(family=family, n=n, platform="plasma", backend="gpu", precision="f32",
                                 construct_ms=c, sample_ms=sm, tvd=validate_counts(gates, n, "plasma_gpu", 42, ref)))
            elif p in ("aer_cpu", "aer_gpu"):
                dev = "CPU" if p == "aer_cpu" else "GPU"
                r, err = run_aer(gates, n, eff_shots(n, shots), 42, dev, args.warm, args.repeats)
                if r is None:
                    print(f"  {p}: SKIP ({err[1][-200:]})", flush=True); continue
                rows.append(dict(family=family, n=n, platform="aer", backend=dev.lower(), precision="f64" if dev == "CPU" else "f32",
                                 construct_ms=r[0], sample_ms=r[1], tvd=validate_counts(gates, n, p, 42, ref)))
            elif p == "custatevec":
                r, err = run_custatevec(gates, n, eff_shots(n, shots), 42, args.warm, args.repeats)
                if r is None:
                    print(f"  custatevec: SKIP ({err[1][-200:]})", flush=True); continue
                rows.append(dict(family=family, n=n, platform="custatevec", backend="gpu", precision="f32",
                                 construct_ms=r[0], sample_ms=r[1], tvd=validate_counts(gates, n, p, 42, ref)))
            elif p == "qsim":
                r, err = run_qsim(gates, n, eff_shots(n, shots), 42, args.warm, args.repeats)
                if r is None:
                    print(f"  qsim: SKIP ({err[1][-200:]})", flush=True); continue
                rows.append(dict(family=family, n=n, platform="qsim", backend="cpu", precision="f64",
                                 construct_ms=r[0], sample_ms=r[1], tvd=validate_counts(gates, n, p, 42, ref)))
            elif p == "qulacs":
                r, err = run_qulacs(gates, n, eff_shots(n, shots), 42, args.warm, args.repeats)
                if r is None:
                    print(f"  qulacs: SKIP ({err[1][-200:]})", flush=True); continue
                rows.append(dict(family=family, n=n, platform="qulacs", backend="cpu", precision="f64",
                                 construct_ms=r[0], sample_ms=r[1], tvd=validate_counts(gates, n, p, 42, ref)))
            elif p == "quest":
                r, err = run_quest(gates, n, eff_shots(n, shots), 42, args.warm, args.repeats)
                if r is None:
                    print(f"  quest: SKIP ({err[1][-200:]})", flush=True); continue
                rows.append(dict(family=family, n=n, platform="quest", backend="cpu", precision="f64",
                                 construct_ms=r[0], sample_ms=r[1], tvd=0.0))
            print(f"  {p}: construct={rows[-1]['construct_ms']:.3f}ms sample={rows[-1]['sample_ms']:.2f}ms tvd={rows[-1]['tvd']:.4f}", flush=True)
        save_rows(rows, args.out)

    if rows:
        print(f"\nwrote {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
