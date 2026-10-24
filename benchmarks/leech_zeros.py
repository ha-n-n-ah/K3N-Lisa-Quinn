#!/usr/bin/env python3
"""
Leech Lattice Spectral Analysis of Riemann Zeta Zeros on S^23

Embeds zeta zero ordinates into S^23 via the 12-prime torus embedding,
then probes the distribution with Leech lattice norm-4 vectors organized
into orbits, fibers, and concurrent transversals.

Steps:
  1. Covariance spectrum of embedded zeros
  2. Coinvariant subspace projection (17-dim / 7-dim split)
  3. Inner products with norm-4 vectors (per-orbit statistics)
  4. Three-filter sign parity on concurrent transversals
  5. Gap-shuffle control
  6. Random-phase control
  7. Summary comparison
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torus_embedding_spectral import (
    compute_phi_m, LOG_P12, FIRST_12_PRIMES,
    make_gap_shuffle_control, make_random_phase_control,
)


# =====================================================================
# ZETA ZEROS
# =====================================================================

def load_zeta_zeros(path=None, max_zeros=None):
    if path and Path(path).exists():
        data = np.loadtxt(path, dtype=np.float64)
        gammas = data.ravel()
        gammas = np.sort(gammas[gammas > 0])
        print(f"  {len(gammas)} zeros loaded, range [{gammas[0]:.3f}, {gammas[-1]:.3f}]")
    else:
        n = max_zeros or 100_000
        print(f"  Computing {n} synthetic zeros (GUE-spaced)...")
        gammas = _synthetic_zeros(n)
    if max_zeros and len(gammas) > max_zeros:
        gammas = gammas[:max_zeros]
        print(f"  Using first {max_zeros} zeros")
    return gammas


def _synthetic_zeros(n, seed=0):
    rng = np.random.default_rng(seed)
    gamma1 = 14.134725
    gammas = [gamma1]
    for _ in range(1, n):
        T = gammas[-1]
        mean_gap = 2 * np.pi / max(np.log(T / (2 * np.pi)), 1.0)
        s = rng.gamma(1.5, 1.0) / 1.5 * mean_gap
        gammas.append(T + max(s, 1e-6))
    return np.array(gammas)


# =====================================================================
# EXTENDED GOLAY CODE [24,12,8] via cyclic QR construction
# =====================================================================

# Generator polynomial of the [23,12,7] binary Golay code:
# g(x) = 1 + x + x^5 + x^6 + x^7 + x^9 + x^11
_GOLAY_GENPOLY = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1], dtype=np.int8)


def generate_golay_codewords():
    g = _GOLAY_GENPOLY
    G23 = np.zeros((12, 23), dtype=np.int8)
    for i in range(12):
        G23[i, i:i + 12] = g
    bits = np.array([[(i >> b) & 1 for b in range(12)] for i in range(4096)], dtype=np.int8)
    cw23 = (bits @ G23) % 2
    parity = (np.sum(cw23, axis=1, keepdims=True) % 2).astype(np.int8)
    return np.hstack([cw23, parity])


def golay_octads(codewords=None):
    if codewords is None:
        codewords = generate_golay_codewords()
    return codewords[np.sum(codewords, axis=1) == 8]


# =====================================================================
# LEECH LATTICE MINIMAL VECTORS (squared norm 32, Euclidean norm sqrt(32))
# =====================================================================

def generate_leech_min_vectors(verbose=True):
    codewords = generate_golay_codewords()
    octads = golay_octads(codewords)

    parts = []

    # Type 1: (\pm4)^2, 0^22 — 1104 vectors
    t1 = []
    for i in range(24):
        for j in range(i + 1, 24):
            for si in (-4.0, 4.0):
                for sj in (-4.0, 4.0):
                    v = np.zeros(24)
                    v[i], v[j] = si, sj
                    t1.append(v)
    t1 = np.array(t1)
    parts.append(t1)
    if verbose:
        print(f"  Type 1 (4^2, 0^22): {len(t1)} vectors")

    # Type 2: (\pm2)^8 on octad support, 0^16 — 97152 vectors
    t2 = []
    sign_bits = np.arange(256, dtype=np.uint16)
    all_signs = np.where((sign_bits[:, None] >> np.arange(8)) & 1, 1.0, -1.0)
    n_pos = np.sum(all_signs > 0, axis=1)
    even_mask = n_pos % 2 == 0
    even_signs = all_signs[even_mask]
    for octad in octads:
        positions = np.where(octad)[0]
        batch = np.zeros((len(even_signs), 24))
        batch[:, positions] = 2.0 * even_signs
        t2.append(batch)
    t2 = np.concatenate(t2, axis=0)
    parts.append(t2)
    if verbose:
        print(f"  Type 2 (2^8, 0^16): {len(t2)} vectors")

    # Type 3: (\pm3)^1, (\pm1)^23 — 98304 vectors
    t3 = []
    for pos3 in range(24):
        v = np.where(codewords, 1.0, -1.0)
        supp_has_pos = codewords[:, pos3] == 1
        v[supp_has_pos, pos3] = -3.0
        v[~supp_has_pos, pos3] = 3.0
        t3.append(v)
    t3 = np.concatenate(t3, axis=0)
    parts.append(t3)
    if verbose:
        print(f"  Type 3 (3, 1^23): {len(t3)} vectors")

    vectors = np.concatenate(parts, axis=0)
    if verbose:
        print(f"  Total: {len(vectors)} vectors")
        norms = np.sqrt(np.sum(vectors ** 2, axis=1))
        print(f"  Norm check: all sqrt(32)={np.sqrt(32):.4f}? "
              f"min={norms.min():.4f}, max={norms.max():.4f}")
    return vectors


# =====================================================================
# SYMMETRY GROUP & COINVARIANT SUBSPACE
# =====================================================================

def coinvariant_projector():
    """
    24x24 projection onto 17-dim coinvariant subspace.

    The 7-dim invariant complement spans:
      - pair-0 cos direction  (1D)
      - diagonal cos/sin across pairs 1-4   (2D)
      - diagonal cos/sin across pairs 5-7   (2D)
      - diagonal cos/sin across pairs 8-11  (2D)
    """
    U = np.zeros((24, 7))
    U[0, 0] = 1.0
    for k in [1, 2, 3, 4]:
        U[2 * k, 1] = 0.5
        U[2 * k + 1, 2] = 0.5
    for k in [5, 6, 7]:
        U[2 * k, 3] = 1.0 / np.sqrt(3)
        U[2 * k + 1, 4] = 1.0 / np.sqrt(3)
    for k in [8, 9, 10, 11]:
        U[2 * k, 5] = 0.5
        U[2 * k + 1, 6] = 0.5
    return np.eye(24) - U @ U.T


def make_antipodal_orbits(vectors, orbit_size=12):
    """
    Group Leech lattice vectors into orbits of size `orbit_size`,
    ensuring each orbit contains both v and -v for every member.
    Antipodal closure guarantees sign-parity averages to exactly 0.5.
    """
    N = len(vectors)
    scale = 1e3
    vec_hash = {}
    for i in range(N):
        h = tuple(np.round(vectors[i] * scale).astype(np.int64))
        vec_hash[h] = i

    visited = np.zeros(N, dtype=bool)
    pairs = []
    for i in range(N):
        if visited[i]:
            continue
        neg_h = tuple(np.round(-vectors[i] * scale).astype(np.int64))
        j = vec_hash.get(neg_h)
        if j is not None and j != i and not visited[j]:
            pairs.append([i, j])
            visited[i] = visited[j] = True
        else:
            pairs.append([i])
            visited[i] = True

    pairs_per_orbit = orbit_size // 2
    orbits = []
    buf = []
    for pair in pairs:
        buf.extend(pair)
        if len(buf) >= orbit_size:
            orbits.append(buf[:orbit_size])
            buf = buf[orbit_size:]
    if buf:
        orbits.append(buf)
    return orbits


def select_orbits(orbits, target_size=12, max_orbits=335, seed=42):
    sized = [o for o in orbits if len(o) == target_size]
    if len(sized) <= max_orbits:
        return sized
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sized), size=max_orbits, replace=False)
    idx.sort()
    return [sized[i] for i in idx]


# =====================================================================
# FIBER & TRANSVERSAL STRUCTURE
# =====================================================================

def build_fiber_structure(n_orbits, fiber_size=7):
    fibers = []
    for start in range(0, n_orbits - fiber_size + 1, fiber_size):
        fibers.append(list(range(start, start + fiber_size)))
    return fibers


def fiber_parts_and_transversals(fiber):
    spread = fiber[len(fiber) // 2]
    non_spread = [o for o in fiber if o != spread]
    parts = []
    for i in range(0, len(non_spread), 2):
        if i + 1 < len(non_spread):
            parts.append([non_spread[i], non_spread[i + 1]])
    all_transversals = []
    concurrent = []
    from itertools import product as iprod
    for choices in iprod(*[range(len(p)) for p in parts]):
        triple = tuple(parts[k][choices[k]] for k in range(len(parts)))
        is_conc = sum(choices) % 2 == 0
        all_transversals.append((triple, is_conc))
        if is_conc:
            concurrent.append(triple)
    return spread, parts, all_transversals, concurrent


# =====================================================================
# STEP 1: COVARIANCE SPECTRUM
# =====================================================================

def step1_covariance_spectrum(phi):
    N, D = phi.shape
    G = (phi.T @ phi) / N
    eigvals = np.sort(np.linalg.eigvalsh(G))[::-1]
    trace = np.sum(eigvals)
    trace2 = np.sum(eigvals ** 2)
    cov_dim = trace ** 2 / trace2
    ratio = eigvals[0] / eigvals[-1]
    frame_ratio = trace2 * D / (trace ** 2)

    mu = np.mean(phi, axis=0)
    mu_norm = np.linalg.norm(mu)

    print("\n" + "=" * 70)
    print("STEP 1: COVARIANCE SPECTRUM OF EMBEDDED ZEROS")
    print("=" * 70)
    print("\n  Covariance eigenvalues (descending):")
    for i, lam in enumerate(eigvals):
        print(f"    lambda_{i + 1:2d} = {lam:.8f}")
    print(f"\n  Trace = {trace:.8f} (expected 1.0 for unit-norm)")
    print(f"  Covariance dimension = {cov_dim:.4f} (max = {D})")
    print(f"  lambda_max / lambda_min = {ratio:.4f}")
    print(f"  Frame ratio = {frame_ratio:.6f}")
    print(f"\n  ||mu|| = {mu_norm:.8f}")
    print(f"  mu direction (first 6): {mu[:6]}")

    return {
        "eigvals": eigvals, "trace": trace, "cov_dim": cov_dim,
        "frame_ratio": frame_ratio, "mu_norm": mu_norm, "mu": mu,
    }


# =====================================================================
# STEP 2: COINVARIANT SUBSPACE PROJECTION
# =====================================================================

def step2_coinvariant_projection(phi, P_coinv):
    N, D = phi.shape
    dim_coinv = int(round(np.trace(P_coinv)))
    dim_inv = D - dim_coinv

    phi_coinv = phi @ P_coinv
    energy_coinv = np.sum(phi_coinv ** 2) / N
    energy_inv = 1.0 - energy_coinv
    expected_coinv = dim_coinv / D

    G_coinv = (phi_coinv.T @ phi_coinv) / N
    eigvals_coinv = np.sort(np.linalg.eigvalsh(G_coinv))[::-1]
    eigvals_coinv = eigvals_coinv[eigvals_coinv > 1e-12]
    cov_dim_coinv = np.sum(eigvals_coinv) ** 2 / np.sum(eigvals_coinv ** 2)

    print("\n" + "=" * 70)
    print("STEP 2: COINVARIANT SUBSPACE PROJECTION")
    print("=" * 70)
    print(f"\n  Energy in coinvariant subspace ({dim_coinv}-dim): "
          f"{energy_coinv:.8f} ({100 * energy_coinv:.2f}%)")
    print(f"  Energy in invariant complement ({dim_inv}-dim):  "
          f"{energy_inv:.8f} ({100 * energy_inv:.2f}%)")
    print(f"  Expected if isotropic: coinv={100 * expected_coinv:.1f}%, "
          f"inv={100 * (1 - expected_coinv):.1f}%")
    print(f"  Excess coinvariant energy: {energy_coinv - expected_coinv:+.6f}")
    print(f"\n  Coinvariant covariance eigenvalues:")
    for i, lam in enumerate(eigvals_coinv[:dim_coinv]):
        print(f"    lambda_{i + 1:2d} = {lam:.8f}")
    print(f"  Coinvariant cov dimension = {cov_dim_coinv:.4f} (max = {dim_coinv})")

    return {
        "energy_coinv": energy_coinv, "energy_inv": energy_inv,
        "cov_dim_coinv": cov_dim_coinv, "dim_coinv": dim_coinv,
    }


# =====================================================================
# STEP 3: INNER PRODUCTS WITH NORM-4 VECTORS
# =====================================================================

def step3_inner_products(phi, vectors, orbits, spread_set=None):
    N = phi.shape[0]
    n_vecs = sum(len(o) for o in orbits)

    norms = np.linalg.norm(vectors[:5], axis=1)
    print("\n" + "=" * 70)
    print("STEP 3: INNER PRODUCTS WITH NORM-4 COINVARIANT VECTORS")
    print("=" * 70)
    print(f"\n  Euclidean norms of norm-4 vectors: {norms}... "
          f"(all should be sqrt(32)={np.sqrt(32):.4f})")
    print(f"  Computing {n_vecs} x {N} inner products...")

    t0 = time.time()
    orbit_stats = []
    for oi, orb in enumerate(orbits):
        vecs = vectors[orb]
        vecs_unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        ips = vecs_unit @ phi.T
        abs_ips = np.abs(ips)
        ip2 = ips ** 2
        is_spread = spread_set is not None and oi in spread_set
        orbit_stats.append({
            "idx": oi, "size": len(orb), "spread": is_spread,
            "abs_ip_mean": float(np.mean(abs_ips)),
            "abs_ip_std": float(np.std(abs_ips)),
            "ip2_mean": float(np.mean(ip2)),
            "max_abs_ip": float(np.max(abs_ips)),
        })
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s")

    print(f"\n  Per-orbit inner product statistics:")
    print(f"  {'Orbit':>5s} {'Size':>4s} {'Spread':>6s}  "
          f"{'|IP| mean':>10s}   {'|IP| std':>10s}  "
          f"{'IP^2 mean':>10s}   {'max |IP|':>8s}")
    n_show = min(35, len(orbit_stats))
    for s in orbit_stats[:n_show]:
        sp = "*" if s["spread"] else ""
        print(f"  {s['idx']:5d} {s['size']:4d} {sp:>6s}  "
              f"{s['abs_ip_mean']:.7f}  {s['abs_ip_std']:.7f}  "
              f"{s['ip2_mean']:.7f}   {s['max_abs_ip']:.6f}")

    spread_ip2 = [s["ip2_mean"] for s in orbit_stats if s["spread"]]
    nonspr_ip2 = [s["ip2_mean"] for s in orbit_stats if not s["spread"]]
    m_spread = np.mean(spread_ip2) if spread_ip2 else 0.0
    m_nonspr = np.mean(nonspr_ip2) if nonspr_ip2 else 0.0
    ratio = m_spread / m_nonspr if m_nonspr > 0 else float("nan")
    print(f"\n  Mean IP^2: spread orbits = {m_spread:.8f}, "
          f"non-spread = {m_nonspr:.8f}")
    print(f"  Ratio spread/non-spread = {ratio:.6f}")

    return orbit_stats


# =====================================================================
# STEP 4: THREE-FILTER SIGN PARITY
# =====================================================================

def step4_sign_parity(phi, vectors, orbits, fibers, fiber_idx=0):
    N = phi.shape[0]
    fiber = fibers[fiber_idx]
    spread, parts, all_trans, concurrent = fiber_parts_and_transversals(fiber)

    print("\n" + "=" * 70)
    print("STEP 4: THREE-FILTER SIGN PARITY ON EMBEDDED ZEROS")
    print("=" * 70)
    print(f"\nFiber {fiber_idx}: orbits {fiber}")
    print(f"  Parts: {parts}")
    print(f"  Concurrent transversals: {concurrent}")

    # Precompute unit-normalized representatives for each orbit in fiber
    reps = {}
    all_reps = {}
    for oi in fiber:
        orb = orbits[oi]
        vecs = vectors[orb]
        vecs_unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        reps[oi] = vecs_unit[0]
        all_reps[oi] = vecs_unit

    # Single representative test
    print(f"\n  All {len(all_trans)} transversal triples:")
    for triple, is_conc in all_trans:
        s_a = np.sign(phi @ reps[triple[0]])
        s_b = np.sign(phi @ reps[triple[1]])
        s_c = np.sign(phi @ reps[triple[2]])
        parity = s_a * s_b * s_c
        n_pos = int(np.sum(parity > 0))
        n_neg = int(np.sum(parity < 0))
        n_zero = int(np.sum(parity == 0))
        frac = n_pos / max(n_pos + n_neg, 1)
        tag = "CONC" if is_conc else "    "
        print(f"    {tag} ({triple[0]:2d}, {triple[1]:2d}, {triple[2]:2d}): "
              f"+parity={n_pos}, -parity={n_neg}, zero={n_zero}, "
              f"frac_pos={frac:.6f}")

    # Averaged over all representative triples
    orbit_size = len(orbits[fiber[0]])
    n_rep_triples = orbit_size ** len(parts)
    print(f"\n  Averaged over all rep triples "
          f"({orbit_size}^{len(parts)} = {n_rep_triples} per transversal):")

    # Precompute sign matrices
    sign_cache = {}
    for oi in fiber:
        sign_cache[oi] = np.sign(all_reps[oi] @ phi.T)

    for triple, is_conc in all_trans:
        total_pos, total_neg = 0, 0
        s_arr = [sign_cache[triple[k]] for k in range(len(triple))]
        n_reps = [s_arr[k].shape[0] for k in range(len(triple))]
        for i in range(n_reps[0]):
            for j in range(n_reps[1]):
                pij = s_arr[0][i] * s_arr[1][j]
                for k in range(n_reps[2]):
                    prod = pij * s_arr[2][k]
                    total_pos += int(np.sum(prod > 0))
                    total_neg += int(np.sum(prod < 0))
        frac = total_pos / max(total_pos + total_neg, 1)
        tag = "CONC" if is_conc else "    "
        print(f"    {tag} ({triple[0]:2d}, {triple[1]:2d}, {triple[2]:2d}): "
              f"frac_pos={frac:.8f}  (+={total_pos}, -={total_neg})")


# =====================================================================
# STEP 5 & 6: CONTROLS
# =====================================================================

def run_control(label, phi_ctrl, vectors, orbits, fibers, P_coinv, step_num):
    N, D = phi_ctrl.shape
    G = (phi_ctrl.T @ phi_ctrl) / N
    eigvals = np.linalg.eigvalsh(G)
    cov_dim = np.sum(eigvals) ** 2 / np.sum(eigvals ** 2)
    frame_ratio = np.sum(eigvals ** 2) * D / (np.sum(eigvals) ** 2)
    mu_norm = np.linalg.norm(np.mean(phi_ctrl, axis=0))

    dim_coinv = int(round(np.trace(P_coinv)))
    phi_coinv = phi_ctrl @ P_coinv
    energy_coinv = np.sum(phi_coinv ** 2) / N

    print(f"\n{'=' * 70}")
    print(f"STEP {step_num}: {label}")
    print("=" * 70)
    print(f"\n  ||mu|| = {mu_norm:.8f}")
    print(f"  Cov dimension = {cov_dim:.4f}")
    print(f"  Frame ratio = {frame_ratio:.6f}")
    print(f"  Coinvariant energy: {energy_coinv:.8f} ({100 * energy_coinv:.2f}%)")

    # Sign parity with single representatives
    if fibers:
        fiber = fibers[0]
        spread, parts, all_trans, concurrent = fiber_parts_and_transversals(fiber)
        reps = {}
        for oi in fiber:
            orb = orbits[oi]
            vecs = vectors[orb]
            vecs_unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            reps[oi] = vecs_unit[0]

        print(f"\n  Sign parity (single rep, {label.lower()}):")
        for triple, is_conc in all_trans:
            s_a = np.sign(phi_ctrl @ reps[triple[0]])
            s_b = np.sign(phi_ctrl @ reps[triple[1]])
            s_c = np.sign(phi_ctrl @ reps[triple[2]])
            parity = s_a * s_b * s_c
            n_pos = int(np.sum(parity > 0))
            n_neg = int(np.sum(parity < 0))
            frac = n_pos / max(n_pos + n_neg, 1)
            tag = "CONC" if is_conc else "    "
            print(f"    {tag} ({triple[0]:2d}, {triple[1]:2d}, {triple[2]:2d}): "
                  f"frac_pos={frac:.6f}")

    return {
        "mu_norm": mu_norm, "cov_dim": cov_dim,
        "frame_ratio": frame_ratio, "energy_coinv": energy_coinv,
    }


# =====================================================================
# SUMMARY
# =====================================================================

def print_summary(N, D, m, s1, s2, ctrl_gs, ctrl_rp):
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  N = {N} zeros, D = {D}, m = {m} primes\n")
    hdr = (f"  {'Metric':<40s} {'Zeros':>12s} {'Gap-shuf':>12s} "
           f"{'Random':>12s}")
    print(hdr)
    print(f"  {'-' * 36} {'-' * 12} {'-' * 12} {'-' * 12}")
    rows = [
        ("||mu||", s1["mu_norm"], ctrl_gs["mu_norm"], ctrl_rp["mu_norm"]),
        ("Cov dimension", s1["cov_dim"], ctrl_gs["cov_dim"], ctrl_rp["cov_dim"]),
        ("Frame ratio", s1["frame_ratio"], ctrl_gs["frame_ratio"], ctrl_rp["frame_ratio"]),
        ("Coinvariant energy %",
         100 * s2["energy_coinv"], 100 * ctrl_gs["energy_coinv"],
         100 * ctrl_rp["energy_coinv"]),
    ]
    for name, vz, vg, vr in rows:
        print(f"  {name:<40s} {vz:12.4f} {vg:12.4f} {vr:12.4f}")
    expected_coinv = 100 * s2["dim_coinv"] / D
    print(f"  {'Expected isotropic %':<40s} {expected_coinv:12.4f}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    zeros_path = None
    max_zeros = 100_000
    for arg in sys.argv[1:]:
        if arg.isdigit():
            max_zeros = int(arg)
        else:
            zeros_path = arg

    print("Loading zeta zeros...")
    gammas = load_zeta_zeros(zeros_path, max_zeros)

    print("Loading lattice data...")
    leech_vecs = generate_leech_min_vectors(verbose=True)

    N = len(gammas)
    m = len(LOG_P12)
    D = 2 * m

    print(f"\nEmbedding {N} zeros into S^{D - 1} via {m} primes...")
    t0 = time.time()
    phi = compute_phi_m(gammas)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s")
    norms = np.linalg.norm(phi, axis=1)
    print(f"  Norm check: mean={norms.mean():.10f}, std={norms.std():.2e}")

    # Build coinvariant projector and orbits
    P_coinv = coinvariant_projector()

    print("\nGrouping into antipodal orbits of size 12...")
    t0 = time.time()
    all_orbits = make_antipodal_orbits(leech_vecs, orbit_size=12)
    dt = time.time() - t0
    print(f"  {len(all_orbits)} orbits in {dt:.1f}s")

    orbits = select_orbits(all_orbits, target_size=12, max_orbits=335)
    print(f"  Selected {len(orbits)} orbits of size 12 "
          f"({sum(len(o) for o in orbits)} vectors)")

    # Mark spread orbits
    spread_set = set()
    for oi, orb in enumerate(orbits):
        v = leech_vecs[orb[0]]
        proj_inv = np.eye(24) - P_coinv
        inv_energy = np.sum((proj_inv @ v) ** 2) / np.sum(v ** 2)
        if inv_energy > 0.15:
            spread_set.add(oi)

    fibers = build_fiber_structure(len(orbits), fiber_size=7)

    # Step 1
    s1 = step1_covariance_spectrum(phi)

    # Step 2
    s2 = step2_coinvariant_projection(phi, P_coinv)

    # Step 3
    step3_inner_products(phi, leech_vecs, orbits, spread_set)

    # Step 4
    if fibers:
        step4_sign_parity(phi, leech_vecs, orbits, fibers, fiber_idx=0)

    # Step 5: Gap-shuffle control
    rng = np.random.default_rng(42)
    gammas_gs = make_gap_shuffle_control(gammas, rng)
    phi_gs = compute_phi_m(gammas_gs)
    ctrl_gs = run_control("GAP-SHUFFLE CONTROL", phi_gs,
                          leech_vecs, orbits, fibers, P_coinv, 5)

    # Step 6: Random-phase control
    phi_rp = make_random_phase_control(N, m, rng)
    ctrl_rp = run_control("RANDOM-PHASE CONTROL", phi_rp,
                          leech_vecs, orbits, fibers, P_coinv, 6)

    # Summary
    print_summary(N, D, m, s1, s2, ctrl_gs, ctrl_rp)


if __name__ == "__main__":
    main()
