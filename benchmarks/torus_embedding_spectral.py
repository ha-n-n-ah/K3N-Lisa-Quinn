import numpy as np
from scipy.spatial.distance import pdist

# =====================================================================
# 1. EMBEDDING & CONTROL GENERATORS
# =====================================================================
FIRST_12_PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37], dtype=np.float64)
LOG_P12 = np.log(FIRST_12_PRIMES)

def compute_phi_m(gammas: np.ndarray, log_primes: np.ndarray = LOG_P12) -> np.ndarray:
    """
    Constructs the 2m-real-dimensional torus-to-sphere embedding:
    Phi_m(gamma_n) = (1/sqrt(m)) * [cos(gamma_n log p_1), sin(gamma_n log p_1), ...]
    Shape: (N, 2m), each row has unit L2 norm.
    """
    m = len(log_primes)
    phases = np.outer(gammas, log_primes)  # (N, m)

    phi = np.empty((len(gammas), 2 * m), dtype=np.float64)
    phi[:, 0::2] = np.cos(phases)
    phi[:, 1::2] = np.sin(phases)
    phi /= np.sqrt(m)
    return phi

def make_gap_shuffle_control(gammas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Permutes the exact zero-gap multiset and reconstructs an ordered sequence.
    """
    gaps = np.diff(gammas)
    shuffled_gaps = rng.permutation(gaps)
    return np.concatenate(([gammas[0]], gammas[0] + np.cumsum(shuffled_gaps)))

def make_random_phase_control(N: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generates matched independent uniform phases on T^m embedded into S^(2m-1).
    """
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(N, m))
    phi = np.empty((N, 2 * m), dtype=np.float64)
    phi[:, 0::2] = np.cos(phases)
    phi[:, 1::2] = np.sin(phases)
    phi /= np.sqrt(m)
    return phi


# =====================================================================
# 2. METHOD 1: GLOBAL MODE EXTRACTION & RESIDUAL RENORMALIZATION
# =====================================================================
def decompose_global_and_residual(phi: np.ndarray):
    """
    Applies Method 1 to separate Phi = Phi_global (+) Phi_residual:
      - Variant A (Centered Spherical): Subtracts the mean vector mu and
        renormalizes back onto the unit sphere S^{23}.
      - Variant B (Orthogonal Complement): Projects out the 1D subspace
        spanned by hat{mu} and renormalizes onto the equator S^{22}.
    """
    # Coherent global mode (mean vector)
    mu = np.mean(phi, axis=0, keepdims=True)          # (1, 24)
    mu_norm = float(np.linalg.norm(mu))
    mu_hat = mu / (mu_norm + 1e-15)                   # Unit direction of global mode

    # Variant A: Mean-centered + spherical renormalization
    phi_centered = phi - mu
    norms_centered = np.linalg.norm(phi_centered, axis=1, keepdims=True)
    phi_res_centered = phi_centered / norms_centered

    # Variant B: Strict subspace projection orthogonal to mu_hat
    proj_lengths = phi @ mu_hat.T                     # (N, 1)
    phi_ortho = phi - proj_lengths @ mu_hat
    norms_ortho = np.linalg.norm(phi_ortho, axis=1, keepdims=True)
    phi_res_ortho = phi_ortho / norms_ortho

    return {
        "mu": mu.ravel(),
        "mu_norm": mu_norm,
        "mu_hat": mu_hat.ravel(),
        "raw": phi,
        "res_centered": phi_res_centered,
        "res_ortho": phi_res_ortho,
    }


# =====================================================================
# 3. 5-MILLION PAIR MONTE CARLO CONTACT & MOMENT AUDITOR
# =====================================================================
def sample_pair_metrics(
    phi: np.ndarray,
    n_pairs: int = 5_000_000,
    batch_size: int = 500_000,
    seed: int = 42
) -> dict:
    """
    Samples `n_pairs` distinct pairs (i != j), computes normalized inner products
    <Phi_i, Phi_j>, covariance spectrum, frame ratio, and 1/2-contact fractions.
    Memory-safe batched execution.
    """
    N, D = phi.shape
    rng = np.random.default_rng(seed)

    # Global spectral & moment diagnostics from the (D x D) Gram/covariance matrix
    G = (phi.T @ phi) / N                             # Second-moment matrix (24 x 24)
    eigvals = np.linalg.eigvalsh(G)

    # Participation ratio / Covariance dimension: (Tr G)^2 / Tr(G^2)
    cov_dim = float((np.sum(eigvals) ** 2) / np.sum(eigvals ** 2))

    # Exact first pair moment M_1 = E[|<Phi_i, Phi_j>|] and M_2 = Tr(G^2)
    # Using batched random pairs for exact alignment
    all_dots = []
    n_remaining = n_pairs
    while n_remaining > 0:
        b = min(batch_size, n_remaining)
        idx_i = rng.integers(0, N, size=b)
        idx_j = rng.integers(0, N - 1, size=b)
        idx_j[idx_j >= idx_i] += 1                    # Ensure i != j

        dots = np.sum(phi[idx_i] * phi[idx_j], axis=1)
        all_dots.append(dots)
        n_remaining -= b

    dots = np.concatenate(all_dots)
    abs_dots = np.abs(dots)

    # Contact fractions at key thresholds
    contact_half = float(np.mean(abs_dots > 0.5))
    contact_quarter = float(np.mean(abs_dots > 0.25))

    # Frame potential: E[<x_i, x_j>^2]; for a tight frame on S^{D-1} this is 1/D
    frame_potential = float(np.mean(dots ** 2))
    frame_ratio = frame_potential * D

    # Pair moment ladder
    M1 = float(np.mean(abs_dots))
    M2 = float(np.mean(dots ** 2))
    M4 = float(np.mean(dots ** 4))

    return {
        "n_pairs": len(dots),
        "eigvals": eigvals,
        "cov_dim": cov_dim,
        "trace_G": float(np.sum(eigvals)),
        "trace_G2": float(np.sum(eigvals ** 2)),
        "lambda_max": float(eigvals[-1]),
        "lambda_min": float(eigvals[0]),
        "M1_abs_dot": M1,
        "M2_dot_sq": M2,
        "M4_dot_4th": M4,
        "frame_potential": frame_potential,
        "frame_ratio": frame_ratio,
        "contact_gt_0.50": contact_half,
        "contact_gt_0.25": contact_quarter,
        "dot_mean": float(np.mean(dots)),
        "dot_std": float(np.std(dots)),
        "dot_median": float(np.median(dots)),
        "dot_min": float(np.min(dots)),
        "dot_max": float(np.max(dots)),
    }


# =====================================================================
# 4. PAIR CORRELATION HISTOGRAM & SPECTRAL STATISTICS
# =====================================================================
def pair_correlation_histogram(
    phi: np.ndarray,
    n_pairs: int = 2_000_000,
    n_bins: int = 200,
    batch_size: int = 500_000,
    seed: int = 99
) -> dict:
    """
    Builds a normalized histogram of pairwise inner products <Phi_i, Phi_j>
    over randomly sampled distinct pairs. Returns bin edges, density, the
    theoretical S^{D-1} baseline density, and Jensen-Shannon divergence
    from that baseline.
    """
    N, D = phi.shape
    rng = np.random.default_rng(seed)

    all_dots = []
    n_remaining = n_pairs
    while n_remaining > 0:
        b = min(batch_size, n_remaining)
        idx_i = rng.integers(0, N, size=b)
        idx_j = rng.integers(0, N - 1, size=b)
        idx_j[idx_j >= idx_i] += 1
        dots = np.sum(phi[idx_i] * phi[idx_j], axis=1)
        all_dots.append(dots)
        n_remaining -= b

    dots = np.concatenate(all_dots)

    counts, bin_edges = np.histogram(dots, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Theoretical baseline: for uniform points on S^{D-1}, the density of
    # <x,y> is proportional to (1 - t^2)^{(D-3)/2}
    alpha = (D - 3) / 2.0
    sphere_density = np.power(np.clip(1.0 - bin_centers ** 2, 0.0, None), alpha)
    sphere_density /= (np.sum(sphere_density) * bin_width + 1e-30)

    # Jensen-Shannon divergence between observed and sphere-uniform
    p = counts * bin_width
    q = sphere_density * bin_width
    p = np.clip(p, 1e-15, None)
    q = np.clip(q, 1e-15, None)
    p /= np.sum(p)
    q /= np.sum(q)
    m_pq = 0.5 * (p + q)
    js_div = float(
        0.5 * np.sum(p * np.log(p / m_pq))
        + 0.5 * np.sum(q * np.log(q / m_pq))
    )

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "density": counts,
        "sphere_baseline": sphere_density,
        "js_divergence": js_div,
        "dot_samples": dots,
    }


def covariance_spectrum_comparison(phi_dict: dict) -> dict:
    """
    Given a dict mapping label -> (N, D) embedding matrix, returns the
    sorted eigenvalue spectra of each covariance matrix for overlay plotting.
    """
    spectra = {}
    for label, phi in phi_dict.items():
        N, D = phi.shape
        G = (phi.T @ phi) / N
        eigvals = np.sort(np.linalg.eigvalsh(G))[::-1]
        spectra[label] = eigvals
    return spectra


# =====================================================================
# 5. FULL SPECTRAL ANALYSIS PIPELINE
# =====================================================================
def run_spectral_analysis(
    gammas: np.ndarray,
    log_primes: np.ndarray = LOG_P12,
    n_mc_pairs: int = 5_000_000,
    n_controls: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    End-to-end spectral analysis of the torus embedding Phi_m applied to
    a sequence of ordinates (e.g. imaginary parts of Riemann zeta zeros).

    Steps:
      1. Embed gammas via Phi_m and compute raw pair metrics.
      2. Decompose into global mode + residual (Methods A & B).
      3. Generate gap-shuffle and random-phase controls.
      4. Compare frame ratios, covariance dimensions, and contact fractions.

    Returns a nested dict with full diagnostics for each variant and control.
    """
    rng = np.random.default_rng(seed)
    m = len(log_primes)
    N = len(gammas)

    if verbose:
        print(f"[Spectral Analysis] N={N}, m={m} primes, D={2*m} embedding dim")

    # Step 1: Raw embedding
    phi_raw = compute_phi_m(gammas, log_primes)
    raw_metrics = sample_pair_metrics(phi_raw, n_pairs=n_mc_pairs, seed=seed)

    if verbose:
        print(f"  Raw: frame_ratio={raw_metrics['frame_ratio']:.4f}, "
              f"cov_dim={raw_metrics['cov_dim']:.2f}, "
              f"||mu||={np.linalg.norm(np.mean(phi_raw, axis=0)):.6f}")

    # Step 2: Global mode decomposition
    decomp = decompose_global_and_residual(phi_raw)
    res_centered_metrics = sample_pair_metrics(
        decomp["res_centered"], n_pairs=n_mc_pairs, seed=seed + 1
    )
    res_ortho_metrics = sample_pair_metrics(
        decomp["res_ortho"], n_pairs=n_mc_pairs, seed=seed + 2
    )

    if verbose:
        print(f"  Centered residual: frame_ratio={res_centered_metrics['frame_ratio']:.4f}, "
              f"cov_dim={res_centered_metrics['cov_dim']:.2f}")
        print(f"  Orthogonal residual: frame_ratio={res_ortho_metrics['frame_ratio']:.4f}, "
              f"cov_dim={res_ortho_metrics['cov_dim']:.2f}")

    # Step 3: Control ensembles
    gap_shuffle_metrics = []
    random_phase_metrics = []

    for c in range(n_controls):
        child_rng = np.random.default_rng(seed + 100 + c)

        gammas_shuffled = make_gap_shuffle_control(gammas, child_rng)
        phi_shuffled = compute_phi_m(gammas_shuffled, log_primes)
        gs_m = sample_pair_metrics(
            phi_shuffled, n_pairs=n_mc_pairs, seed=seed + 200 + c
        )
        gap_shuffle_metrics.append(gs_m)

        phi_rand = make_random_phase_control(N, m, child_rng)
        rp_m = sample_pair_metrics(
            phi_rand, n_pairs=n_mc_pairs, seed=seed + 300 + c
        )
        random_phase_metrics.append(rp_m)

    if verbose:
        gs_fr = np.mean([m_["frame_ratio"] for m_ in gap_shuffle_metrics])
        rp_fr = np.mean([m_["frame_ratio"] for m_ in random_phase_metrics])
        print(f"  Gap-shuffle controls (n={n_controls}): mean frame_ratio={gs_fr:.4f}")
        print(f"  Random-phase controls (n={n_controls}): mean frame_ratio={rp_fr:.4f}")

    # Step 4: Pair correlation histograms
    raw_hist = pair_correlation_histogram(phi_raw, seed=seed + 500)

    if verbose:
        print(f"  Pair correlation JS divergence (raw vs sphere): "
              f"{raw_hist['js_divergence']:.6f}")

    results = {
        "N": N,
        "m": m,
        "D": 2 * m,
        "raw": raw_metrics,
        "raw_histogram": raw_hist,
        "decomposition": {
            "mu_norm": decomp["mu_norm"],
            "mu_hat": decomp["mu_hat"],
        },
        "res_centered": res_centered_metrics,
        "res_ortho": res_ortho_metrics,
        "gap_shuffle_controls": gap_shuffle_metrics,
        "random_phase_controls": random_phase_metrics,
    }

    if verbose:
        print("[Spectral Analysis] Complete.")

    return results


def summarize_comparison(results: dict) -> dict:
    """
    Extracts the key scalar comparisons from a run_spectral_analysis result:
    frame ratio excess, covariance dimension deficit, and contact fraction
    ratios vs. controls.
    """
    raw = results["raw"]

    gs_controls = results["gap_shuffle_controls"]
    rp_controls = results["random_phase_controls"]

    gs_frame = np.mean([c["frame_ratio"] for c in gs_controls])
    rp_frame = np.mean([c["frame_ratio"] for c in rp_controls])
    gs_covdim = np.mean([c["cov_dim"] for c in gs_controls])
    rp_covdim = np.mean([c["cov_dim"] for c in rp_controls])
    gs_contact = np.mean([c["contact_gt_0.50"] for c in gs_controls])
    rp_contact = np.mean([c["contact_gt_0.50"] for c in rp_controls])

    return {
        "raw_frame_ratio": raw["frame_ratio"],
        "raw_cov_dim": raw["cov_dim"],
        "raw_contact_50": raw["contact_gt_0.50"],
        "gap_shuffle_frame_ratio": float(gs_frame),
        "random_phase_frame_ratio": float(rp_frame),
        "frame_excess_vs_gap_shuffle": raw["frame_ratio"] - float(gs_frame),
        "frame_excess_vs_random_phase": raw["frame_ratio"] - float(rp_frame),
        "cov_dim_deficit_vs_gap_shuffle": raw["cov_dim"] - float(gs_covdim),
        "cov_dim_deficit_vs_random_phase": raw["cov_dim"] - float(rp_covdim),
        "contact_ratio_vs_gap_shuffle": (
            raw["contact_gt_0.50"] / max(float(gs_contact), 1e-15)
        ),
        "contact_ratio_vs_random_phase": (
            raw["contact_gt_0.50"] / max(float(rp_contact), 1e-15)
        ),
    }


# =====================================================================
# 6. NEAREST-NEIGHBOR ANGULAR SEPARATION ON S^{2m-1}
# =====================================================================
def nearest_neighbor_angles(
    phi: np.ndarray,
    n_queries: int = 50_000,
    seed: int = 77
) -> np.ndarray:
    """
    For a random subset of n_queries rows, finds the nearest neighbor
    (by geodesic angle arccos(<Phi_i, Phi_j>)) via brute-force batched
    search. Returns the array of nearest-neighbor angles in radians.
    """
    N, D = phi.shape
    rng = np.random.default_rng(seed)

    if n_queries >= N:
        query_idx = np.arange(N)
    else:
        query_idx = rng.choice(N, size=n_queries, replace=False)

    nn_angles = np.empty(len(query_idx), dtype=np.float64)
    batch = 2000

    for start in range(0, len(query_idx), batch):
        end = min(start + batch, len(query_idx))
        q = phi[query_idx[start:end]]                 # (B, D)
        sims = q @ phi.T                              # (B, N)
        sims[np.arange(end - start), query_idx[start:end]] = -2.0
        max_sims = np.max(sims, axis=1)
        nn_angles[start:end] = np.arccos(np.clip(max_sims, -1.0, 1.0))

    return nn_angles


# =====================================================================
# 7. CLI ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    import sys

    N_ZEROS = 10_000
    N_MC = 2_000_000
    N_CTRL = 3

    print("=" * 70)
    print("Torus Embedding Spectral Analysis")
    print(f"  Synthetic demo: {N_ZEROS} ordinates, {N_MC} MC pairs, "
          f"{N_CTRL} controls")
    print("=" * 70)

    rng = np.random.default_rng(0)
    gammas = np.sort(rng.uniform(14.0, 5000.0, size=N_ZEROS))

    results = run_spectral_analysis(
        gammas,
        n_mc_pairs=N_MC,
        n_controls=N_CTRL,
        seed=12345,
        verbose=True,
    )

    summary = summarize_comparison(results)
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")

    phi_raw = compute_phi_m(gammas)
    nn_ang = nearest_neighbor_angles(phi_raw, n_queries=min(5000, N_ZEROS))
    print(f"\nNearest-neighbor angles (radians):")
    print(f"  mean={np.mean(nn_ang):.4f}, median={np.median(nn_ang):.4f}, "
          f"min={np.min(nn_ang):.4f}, max={np.max(nn_ang):.4f}")

    sys.exit(0)
