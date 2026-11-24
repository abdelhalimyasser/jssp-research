# Initial Population Generation (`A01`–`A24`, Phase A)

**Rust Implementation:** [`crates/jssp-core/src/initialization/`](file:///home/abdelhalimyasser/Downloads/jssp-research/crates/jssp-core/src/initialization/mod.rs)
**Protocol & Mapping Baseline:** [`qmc_mapping.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/qmc_mapping.md), [`phase_a.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/experiments/phase_a.md)

---

## 1. Unified Sequential OBR Mapping (`D02-A`)

Every unit-cube coordinate vector $u \in [0, 1)^L$ ($L = J \cdot M$) maps deterministically to a valid Operation-Based Representation (OBR) chromosome $\pi \in \{0, \dots, J-1\}^L$ via sequential inverse-CDF selection (`discrete_mapping.rs`):

1. Track dynamic remaining operations `rem_ops[j]`, next operation index `next_op[j]`, and remaining processing work $W_j(t) = \sum_{s = \text{next\_op}[j]}^{M-1} p_{j,s}$.
2. Build the ascending-`JobId` eligible set $R_t = \{j : \text{rem\_ops}[j] > 0\}$.
3. Assign probability mass $P_t(j) = W_j(t)^\alpha / \sum_{k \in R_t} W_k(t)^\alpha$ (falling back to uniform $1/|R_t|$ whenever $\alpha = 0$ or total remaining work is $0$).
4. Partition $[0, 1)$ in ascending `JobId` order and map $u_t \in [0, 1)$ cleanly to the matching interval, decrementing $W_{j^*}(t)$ dynamically after each step.

Every generated chromosome trivially satisfies exact job length $L$ and per-job multiplicity $M$. Note that this dynamic state-dependent mapping is discontinuous, so continuous Koksma–Hlawka low-discrepancy bounds do **not** automatically transfer to OBR schedule diversity; empirical Layer A diversity and schedule uniqueness are measured directly.

---

## 2. QMC Generators, Scrambling & Prefix Indexing (`D02-B`)

- **Deterministic MWR (`A02`):** Greedy $\arg\max_{j \in R_t} W_j(t)$ with lowest-`JobId` tie-breaking, replicated identically across $N = 64$ slots without artificial diversity noise (`unique_chromosomes = 1`, `obr_hamming_diversity = 0.0`).
- **Monte Carlo + MWR (`A01`, `A03`, `A15`, `A16`):** PRNG coordinates drawn sequentially from the isolated `"init"` `ChaCha20Rng` substream ($\alpha \in \{0.0, 1.0, 0.5, 2.0\}$).
- **Halton (`A04`, `A08`, `A17`, `A18`) & Scrambled Halton (`A05`, `A09`, `A19`, `A20`):** First $L$ prime bases ($2, 3, 5, \dots$), contiguous 1-based prefix $i \in \{1, \dots, N\}$. Scrambled variants draw exact per-(dimension, digit-level) random permutations via an $O(N)$ partial Fisher–Yates prefix on `"qmc_scramble"`, mathematically equivalent in distribution to full $S_{b_s}$ digit permutations.
- **Joe & Kuo (2008) Sobol (`d_max = 21,201`):** Direction polynomials embedded from UNSW `new-joe-kuo-6.21201` (`joe_kuo_21201.bin`).
  - **Unscrambled (`A06`, `A10`, `A21`, `A22`):** Shifted 1-based prefix $i \in \{1, \dots, 64\}$ omitting the degenerate all-zero point $i = 0$ (no 0-based dyadic net closure at $i = 64$).
  - **Scrambled (`A07`, `A11`, `A23`, `A24`):** Contiguous 0-based dyadic prefix $i \in \{0, \dots, 63\}$ ($N = 64 = 2^6$) randomized by a Matoušek (1998) invertible lower-triangular $32 \times 32$ binary matrix ordered from MSB (`bit 31`) to LSB (`bit 0`) plus a 32-bit XOR digital shift from `"qmc_scramble"`, preserving exact 1D/dyadic $(t, 6, s)$-net stratification.
  - **Dimension Guard:** Standard Sobol strictly rejects $L > 21,201$ with `JsspError::UnsupportedQmcDimension`.
- **Mixed Scrambled Halton / Sobol (`A12`–`A14`):** Independent contiguous prefixes ($N_H$ Halton $i=1..N_H$, $N_S$ Sobol $i=0..N_S-1$):
  - `A12` (`32:32`) and `A14` (`48:16`) yield dyadic power-of-two Sobol subsets ($2^5, 2^4$) preserving net balance on the Sobol half.
  - `A13` (`16:48`) has $N_S = 48 = 3 \times 2^4 \neq 2^m$ and is explicitly flagged as an empirical mixed-QMC configuration without standard dyadic net-balance guarantees.
