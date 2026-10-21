# K3N-Lisa-Quinn
Deterministic Conformational Fingerprinting, Zero-Holdout Algebraic Coding, and High-Throughput Structural Triage

# QUINN Quantum Hashing & The STAA Quotient Framework

**Deterministic Conformational Fingerprinting, Zero-Holdout Algebraic Coding, and High-Throughput Structural Triage**

---

## 1. For the General Reader: Why This Exists

Proteins are long chains of amino acids that fold into complex 3D shapes to do their jobs in the body. In synucleinopathies—such as **Parkinson’s disease**, **Lewy body dementia (DLB)**, and **Multiple System Atrophy (MSA)**—a 140-amino-acid protein called **alpha-synuclein** misfolds and clumps together into toxic fibrils. Even a single point mutation (such as `E46K`, `A53T`, or `G51D`) or a subtle shift in the backbone's hinge angles can change how these fibrils twist, how fast they spread, and which brain regions they affect.

When researchers screen thousands of protein variants or compare large structural libraries, standard 3D alignment algorithms (like TM-align or DALI) compare every atom against every other atom. That takes $\mathcal{O}(N^2)$ time, creating a massive computational bottleneck when screening millions of candidate conformations.

**What this repository does:**
Instead of aligning 3D clouds of atoms in floating-point space, this framework translates a protein's backbone angles $(\phi, \psi)$ into a **fixed 128-bit digital fingerprint** and a **discrete structural shell index** at register speed (~8.5 nanoseconds per residue encode in C++). It allows anyone on a standard laptop to:
1. Fingerprint a protein chain's backbone trajectory in $\mathcal{O}(N)$ linear time.
2. Detect single-residue angular perturbations without training a neural network.
3. Rapidly triage large structural libraries to flag ordered fibril cores versus disordered segments before committing heavy GPU compute.

---

## 2. Our Scientific Posture: What This Is (and What It Is Not)

We believe open-source science should **under-claim and over-deliver**. Every claim in this repository is backed by an executable test that runs in seconds on a standard CPU.

### What This Framework IS:
* **A Mathematically Certified Quotient Sifter (`affine-d6`):** An exact $(24, 4096, 6)$ affine coset code (`OFFSET = 0x01F`) that maps all $4,096$ 12-bit Gray-coded dihedral cells into odd Hamming weights $k \in \{5, 7, 9, 11, 13, 15, 17, 19\}$ with **zero forbidden holdouts ($\vert{}U\vert{} = 0$)** and **zero retry branches**.
* **A Wire-Speed Structural Fingerprint:** A deterministic 128-bit cross-coupled accumulator (`quinn::hash_trajectory`) paired with a 4D Clifford Torus Mercer Gram purity operator $\operatorname{Tr}(\mathbf{K}^2)$ that separates ordered structural classes from disordered chains without parameter fitting.
* **A First-Pass Compute Triage Filter:** A fast pre-filter to help structural biologists and computational researchers spot conformational shifts, domain boundaries, and mutant hinges across PDB/mmCIF datasets.

### What This Framework IS NOT (Where Not to Overreach):
* **Not a Standalone Clinical Diagnostic:** While we benchmark against experimental Cryo-EM alpha-synuclein structures (such as `6A6B`, `6L4S`, `6LRQ`, and `8A9L`), this software is a research and structural-triage tool—not a medical device or clinical diagnostic for patients.
* **Not a Replacement for Atomistic Physics or Wet-Lab Assays:** A 24-bit codeword shell $s(k) = \operatorname{round}(k(25-k)/2)$ is a coarse, register-speed invariant based on Hamming weight (popcount). By itself, a single residue's shell index does not capture long-range electrostatic solvation or side-chain packing. That is why we pair the fast $\mathcal{O}(N)$ discrete hash with the continuous $\mathcal{O}(N^2)$ Clifford–Mercer trace $\operatorname{Tr}(\mathbf{K}^2)$ and sliding-window analyses.
* **Not a Black-Box AI:** There are zero learned weights, zero hidden neural layers, and zero stochastic outputs in this pipeline. The same input coordinates will produce the exact same 128-bit hex string on every machine.

---

## 3. For Technical Readers: Mathematical & Engineering Architecture

The pipeline bridges continuous backbone dihedral geometry on the flat torus $S^1 \times S^1$ with discrete coding theory in five deterministic stages:

```text
  Backbone Dihedrals (φ_i, ψ_i) ∈ [-π, π)²
            │
            ├──► [Path A: Continuous Semiclassical Operator]
            │     1. 4D Clifford Torus Embedding:
            │        x_i = (cos φ_i, sin φ_i, cos ψ_i, sin ψ_i) ∈ S¹ × S¹ ⊂ ℝ⁴
            │     2. Mercer Gram Kernel & Trace Purity:
            │        K_ij = exp(-||x_i - x_j||² / 2σ²),   γ = Tr(K²) = ∑ K_ij²
            │
            └──► [Path B: Discrete STAA Quotient & 128-Bit Hash]
                  1. 12-Bit Asymmetric Torus Quantization & Gray Fold:
                     m_i = gray12(w_φ ⊕ rotl12(w_ψ, 6)) ∈ F₂¹²   (4,096 cells)
                  2. 24-Bit Systematic Affine Encoding (`affine-d6`, OFFSET = 0x01F):
                     c_i = ((m_i << 12) | (m_i · B)) ⊕ 0x01F     (k ∈ {5..19} odd, |U| = 0)
                  3. Quadratic Shell Projection:
                     s(k) = round(k(25 - k) / 2) ∈ {50, 57, 63, 68, 72, 75, 77, 78}
                  4. 128-Bit Station-Rotated Cross-Coupled Accumulator:
                     rot = (7 · station) mod 64
                     low  ^= rotl64((c_high << 32) | c_low, rot)
                     high ^= rotl64((c_low << 32) | c_high, rot)

Why Standard Golay G 
24
​	
  Fails Protein Conformational Mapping (And How affine-d6 Fixes It)
A quadratic shell classifier s(k)=round(k(25−k)/2) is symmetric around k=12.5. Because k+(25−k)=25 is odd, every valid interior shell is populated by a complementary pair of weights {k 
odd
​	
 ,k 
even
​	
 } of opposite parity:
Shell s	Weight Pair {kodd​,keven​}	Structural Motif Regime	Populated by golay-g24 [24,12,8]?	Populated by affine-d6 (24,4096,6)?
78	{13,12}	alpha_R (Right-handed α-helix)	Yes (k=12)	Yes (k=13)
77	{11,14}	beta (β-sheet / extended)	NO (Missed!)	Yes (k=11)
75	{15,10}	PPII_or_coil (Polyproline II / coil)	NO (Missed!)	Yes (k=15)
72	{9,16}	alpha_L (Left-handed α-helix)	Yes (k=16)	Yes (k=9)
68	{17,8}	gamma_turn (γ-turn)	Yes (k=8)	Yes (k=17)
63	{7,18}	border_loop (Marginal turn)	NO (Missed!)	Yes (k=7)
57	{19,6}	border_loop (Marginal loop)	NO (Missed!)	Yes (k=19)
50	{5,20}	border_loop (Marginal loop)	NO (Missed!)	Yes (k=5)
The Golay Blind Spot: The classic Extended Binary Golay code G 
24
​	
  [24,12,8] is doubly-even (weights restricted to {0,8,12,16,24}). Because it lacks weights {10,11,14,15}, it is completely blind to shells 77 (beta) and 75 (PPII_or_coil), which are the exact conformational regimes critical to amyloid fibrils and intrinsically disordered proteins.
The Covering Radius Ceiling (d 
min
​	
 =6): Any [24,12,≥7] code punctured at one coordinate is equivalent to the perfect [23,12,7] Golay code G 
23
​	
 , which has covering radius ρ(G 
23
​	
 )=3. Extending by 1 bit yields covering radius ρ(C)≤4. Thus, every 24-bit affine offset t∈F 
2
24
​	
  lies within distance ≤4 of some codeword, forcing at least one holdout of weight k≤4. Therefore, d 
min
​	
 =6 (affine-d6) is the exact mathematical ceiling for a zero-holdout (∣U∣=0) Pass-0 code.
The affine-d6 Solution (OFFSET = 0x01F): By pairing a rank-12 parity matrix B whose 12 rows all have odd weight (5 or 7) with the weight-5 translation offset 0x01F, every linear even weight flips to an odd weight strictly inside k∈{5,7,9,11,13,15,17,19}. All 4,096 codewords lock on Pass 0 (∣U∣=0) across all 8 interior shells.
4. Summary of Verified Facts (Reproducible in This Repo)
When you run make check on this repository, the test harness independently verifies the following empirical and algebraic facts:
Zero Holdouts Across All 4,096 States (staa_certify & test_audit_pipeline.py):
affine-d6 produces 0 holdouts (∣U∣=0) with exact odd weight distribution {5: 24, 7: 171, 9: 630, 11: 1205, 13: 1260, 15: 613, 17: 166, 19: 27} and shell distribution {50: 24, 57: 27, 63: 171, 68: 166, 72: 630, 75: 613, 77: 1205, 78: 1260}.
b12-legacy (d 
min
​	
 =4, 3 holdouts) and linear-d6 (d 
min
​	
 =6, 1 holdout at 0x000) pass exhaustive quotient-fiber closure checks across widths w=12…30 (0 leaks) and 300,000 forced 1024-bit carry-wrap tests.
Asymmetric Dihedral Quantization Eliminates ϕ=ψ Cancellation (test_quinn.cpp & blind_test_pdb.py):
Circularly shifting w 
ψ
​	
  by 6 bits (w_phi ^ rotl12(w_psi, 6)) before Gray folding eliminates the ϕ≈ψ⟹0 diagonal XOR cancellation while preserving exact 4,096-preimage uniformity.
On 100-residue helical chains (blind_test_pdb.py Test 3), single-residue point-perturbation sensitivity reaches 100/100 (100%) for both ∼30 
∘
  shifts and subtle ∼6 
∘
  shifts.
Collision Resistance & Zero-Training Fold Separation (blind_test_pdb.py & quinn_bench.cpp):
Across 10,000 random 24-station trajectories and 1,000 pure secondary-structure chains (5 classes × 200 samples), the 128-bit accumulator achieves 100.0% unique hashes (0 intra-class and 0 inter-class collisions).
Across 1,000 synthetic multi-segment protein chains spanning 10 fold templates (globin_like, immunoglobulin, tim_barrel, amyloid_fibril, disordered, etc.), simple 2-feature nearest-centroid lookup (Tr(K 
2
 ), 
s
ˉ
 ) with zero training achieves 88.4% classification accuracy.
Hardware Throughput (quinn_bench.cpp):
Raw STAA affine-d6 encoding runs at ~8.5 ns per encode (~117.5 million encodes/sec) on a single CPU core.
Full 100-residue trajectory hashing + 4D Clifford–Mercer Tr(K 
2
 ) evaluation completes in ~113 microseconds per chain.
5. Repository Structure
Plaintext
.
├── Makefile                                      # Builds C++ binaries & runs full C++/Python verification
├── README.md                                     # Overview, scientific scope, and quickstart
├── docs/
│   └── audit_pipeline.md                         # Mathematical proofs, 4-profile tradeoffs & stage contracts
├── include/
│   ├── staa_profiles.hpp                         # Swappable STAA profiles (`b12-legacy`, `linear-d6`, `affine-d6`, `golay-g24`)
│   └── quinn_hash.hpp                            # 4D Clifford Torus Mercer Tr(K^2) & 128-bit M24 orbit hash
├── src/
│   └── staa_certify.cpp                          # Exhaustive width 12..30 & 1024-bit forced-carry profile certifier
├── tests/
│   ├── test_quinn.cpp                            # 13-gate C++ unit test suite
│   ├── test_audit_pipeline.py                    # Standalone 5-stage Python audit verification
│   └── test_codeword_tradeoffs_and_pairs.py      # Paired 4-profile spectrum, Odd-Sum pairing & JS divergence suite
└── benchmarks/
    ├── quinn_bench.cpp                           # C++ wire-speed latency & throughput benchmark
    ├── blind_test_pdb.py                         # Zero-training secondary structure, fold & mutant sensitivity test
    └── live_pdb_alpha_syn_benchmark.py           # Live RCSB Cryo-EM Alpha-Synuclein (PD / DLB) PDB parser & audit
6. Quickstart: How to Build and Verify Everything Yourself
Requires only a C++17 compiler (g++ or clang++) and Python 3 with numpy.
Run the Complete C++ and Python Verification Suite (One Command)
Bash
make check
Or Run the Python Verification Suites Directly (No Compiler Needed)
Bash
python3 tests/test_audit_pipeline.py
python3 tests/test_codeword_tradeoffs_and_pairs.py
python3 benchmarks/blind_test_pdb.py
python3 benchmarks/live_pdb_alpha_syn_benchmark.py
Build and Run the C++ Certifier & Wire-Speed Benchmarks
Bash
make all
./staa_certify
./test_quinn
./quinn_bench
7. License & Citation
Released open-source for independent verification, critique, and research use in structural bioinformatics and high-dimensional coding theory
