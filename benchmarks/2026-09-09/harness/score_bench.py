#!/usr/bin/env python3
"""Compute raw + weighted benchmark scores (Plasma baseline = 1.00)."""
import argparse, csv, math, os

HERE = os.path.dirname(os.path.abspath(__file__))

def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        out[(r["family"], int(r["n"]), r["platform"])] = r
    return out

def geo_mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs: return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

def ratio(plasma_ms, other_ms):
    if other_ms is None or other_ms <= 0 or plasma_ms is None:
        return None
    return plasma_ms / other_ms

def score(rows, axis_platforms, plasma_key, scales):
    """Weighted geometric-mean construction score. Weight = 1 per (family,scale);
    families equal, scales weighted equally. Plasma baseline=1.0."""
    fams = ["ghz", "qft", "rnd_hc", "rnd_rot", "clifford"]
    table = {}
    for p in axis_platforms:
        rs = []
        for f in fams:
            for n in scales:
                pp = rows.get((f, n, plasma_key))
                oo = rows.get((f, n, p))
                if pp is None or oo is None:
                    continue
                c = ratio(float(pp["construct_ms"]), float(oo["construct_ms"]))
                rs.append(c)
        table[p] = geo_mean(rs)
    return table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=os.path.join(HERE, "cross_results_gpu.csv"))
    ap.add_argument("--cpu", default=os.path.join(HERE, "cross_results_cpu.csv"))
    args = ap.parse_args()

    gpu = load(args.gpu) if os.path.exists(args.gpu) else {}
    cpu = load(args.cpu) if os.path.exists(args.cpu) else {}

    print("=" * 70)
    print("WEIGHTED CONSTRUCTION SCORES  (geometric mean, Plasma = 1.00)")
    print("=" * 70)

    gpu_scales = [16, 20, 24, 27]
    gpu_platforms = ["plasma", "aer", "custatevec"]
    g = score(gpu, gpu_platforms, "plasma", gpu_scales)
    print("\nGPU (statevector, f32, q16/20/24/27, 5 families):")
    for p in gpu_platforms:
        print(f"  {p:12s} {g.get(p):7.3f}" if g.get(p) else f"  {p:12s}  N/A")

    cpu_scales = [20, 24, 26]
    cpu_platforms = ["plasma", "aer", "qsim", "qulacs", "quest"]
    c = score(cpu, cpu_platforms, "plasma", cpu_scales)
    print("\nCPU (statevector, f64, q20/24/26, 5 families):")
    for p in cpu_platforms:
        print(f"  {p:12s} {c.get(p):7.3f}" if c.get(p) else f"  {p:12s}  N/A")

    print("\n" + "=" * 70)
    print("RAW CONSTRUCTION MATRIX (ms, min of runs; ratio = Plasma/candidate)")
    print("=" * 70)
    for axis, data, scales in [("GPU", gpu, gpu_scales), ("CPU", cpu, cpu_scales)]:
        print(f"\n--- {axis} ---")
        fams = ["ghz", "qft", "rnd_hc", "rnd_rot", "clifford"]
        platforms = gpu_platforms if axis == "GPU" else cpu_platforms
        for f in fams:
            for n in scales:
                row = []
                for p in platforms:
                    r = data.get((f, n, p))
                    if r:
                        row.append(f"{p[0]}:{float(r['construct_ms']):.3f}")
                print(f"  {f:8s} q{n:<3d}  " + "  ".join(row))

if __name__ == "__main__":
    main()