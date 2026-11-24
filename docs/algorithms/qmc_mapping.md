# QMC Discrete Mapping, MWR Bias & Initializers (`D02`)

**Decision IDs:** `D02-A` (`ACCEPTED`), `D02-B` (`ACCEPTED`)
**Scope:** Phase A (`A01`–`A24`) population initialization algorithms, MWR dynamic remaining-work probability model, Halton/Sobol sequence generation and scrambling, dimension limits, mixed populations, and discrepancy caveats.

---

## 1. Common Sequential Discrete OBR Mapping (`D02-A`: `ACCEPTED`)

To map any vector of coordinates $u = (u_0, u_1, \dots, u_{L-1}) \in [0, 1)^L$ (whether pseudo-random or QMC) onto a guaranteed valid OBR chromosome $\pi \in \{0, \dots, n-1\}^L$:

1. Maintain remaining operation counts `rem_ops[j]` (initialized to $m_j$) and next-operation index `next_op[j] = 0` for each job $j \in \{0, \dots, n-1\}$.
2. At construction step $t \in \{0, \dots, L-1\}$:
   - Form the **eligible job list** $R_t = [j \in \{0,\dots,n-1\} \mid \text{rem\_ops}[j] > 0]$ sorted in strictly **ascending `JobId` order** (`0 < 1 < ... < n-1`).
   - Compute dynamic remaining processing work for each $j \in R_t$:
     $$W_j(t) = \sum_{s = \text{next\_op}[j]}^{m_j - 1} p_{j,s}$$
   - Compute selection probabilities using MWR exponent $\alpha \ge 0$:
     $$P(j \mid t) = \begin{cases}
     \dfrac{1}{|R_t|}, & \text{if } \alpha = 0 \text{ or } \sum_{r \in R_t} W_r(t)^\alpha = 0, \\[1em]
     \dfrac{W_j(t)^\alpha}{\sum_{r \in R_t} W_r(t)^\alpha}, & \text{otherwise.}
     \end{cases}$$
   - Partition $[0, 1)$ into half-open cumulative intervals $[F_r, F_{r+1})$ where $F_0 = 0$, $F_{r+1} = F_r + P(R_t[r] \mid t)$, and the final boundary is **explicitly clamped** $F_{|R_t|} = 1.0$ to prevent IEEE-754 floating-point drift from failing to select an item when $u_t$ is near $1.0$.
   - Select the unique job $j^* = R_t[r]$ satisfying $F_r \le u_t < F_{r+1}$ (if $u_t \ge 1.0$ due to numerical artifact, clamp $u_t$ or fall back to $R_t[|R_t|-1]$).
   - Set $\pi_t = j^*$, increment `next_op[j^*] += 1`, and decrement `rem_ops[j^*] -= 1`.

**Validity Guarantee:** Because each job $j$ is eligible iff `rem_ops[j] > 0`, every job is selected exactly $m_j$ times, guaranteeing 100% OBR feasibility on every coordinate vector $u \in [0, 1)^L$.

---

## 2. MWR Exponent Sensitivity & Deterministic MWR (`A02`)

- **Dynamic vs. Static Work:** $W_j(t)$ is recomputed dynamically from unscheduled operations only, never fixed to initial job total work $W_j(0)$.
- **Exponent Settings (`A01`–`A24`):**
  - $\alpha = 0$: Uniform choice over $R_t$ (`A01`, `A04`–`A07`).
  - $\alpha = 0.5$: Mild MWR bias (`A15`, `A17`, `A19`, `A21`, `A23`).
  - $\alpha = 1.0$: Primary proportional MWR bias (`A03`, `A08`–`A11`).
  - $\alpha = 2.0$: Strong MWR bias (`A16`, `A18`, `A20`, `A22`, `A24`).
- **Deterministic MWR (`A02`):** At step $t$, select:
  $$j^* = \arg\max_{j \in R_t} W_j(t)$$
  breaking all exact ties deterministically by **lowest `JobId`**. No hidden tie-breaking jitter is permitted.
  - *Diversity Consequence:* Every member of an `A02` population of size $N$ receives the identical chromosome (`duplicate_rate = (N - 1)/N`, `unique_chromosomes = 1`, `hamming_diversity = 0.0`). This is an intended low-diversity deterministic control and must be reported honestly.

---

## 3. Halton & Scrambled Halton (`A04`, `A05`, `A08`, `A09`, ...) (`D02-B`: `ACCEPTED`)

- **Primary Source:** J. H. Halton, "On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals," *Numerische Mathematik*, vol. 2, pp. 84–90, 1960. DOI: `10.1007/BF01386213`.
- **Sequence Definition:** For chromosome dimension $s \in \{0, \dots, L-1\}$, assign prime base $b_s$ (the $(s+1)$-th prime: $2, 3, 5, \dots$). Point index $i \in \{1, \dots, N\}$ (skipping $i = 0$ so point 0 is not the all-zero vector that systematically maps to `JobId = 0` at every position) computes the radical inverse in base $b_s$:
  $$i = \sum_{k=0}^{K} d_k b_s^k, \qquad \phi_{b_s}(i) = \sum_{k=0}^{K} d_k b_s^{-(k+1)} \in [0, 1)$$
- **Scrambling Algorithm (`A05`, `A09`, `A19`, `A20`):** Deterministic **Faure / random digit permutation scrambling** per dimension $s$ using the `qmc_scramble` PRNG substream:
  - For each coordinate $s$ and digit position $k$ (or fixed per-base permutation $\pi_s$ over $\{0, \dots, b_s - 1\}$), compute scrambled coordinate:
    $$\tilde{\phi}_{b_s}(i) = \sum_{k=0}^{K} \pi_{s,k}(d_k)\, b_s^{-(k+1)}$$
  - *Choice over Cranley–Patterson shift:* Random-digit permutations break the severe linear correlations observed between high-prime Halton coordinates ($b_s$ large) at small $N$ much more effectively than simple modulo-1 additive shifts, while preserving $(0, m, s)$ / equidistribution of each 1D projection.
- **Precision & Dimension Support:** Precompute or sieve the first $L$ primes in `u64` and accumulate radical inverses in `f64` (53-bit significand suffices for $N \le 10^9$). Up to $L = 21,201$, the 21,201st prime is $239,977$, well within `u32`/`u64`.
- **Independent Replicates:** Each stochastic replicate seed derives a fresh `qmc_scramble` stream generating independent digit permutations $\pi_{s,k}$.

---

## 4. Sobol & Scrambled Sobol (`A06`, `A07`, `A10`, `A11`, ...) (`D02-B`: `ACCEPTED`)

- **Primary Sources:**
  - I. M. Sobol', "On the distribution of points in a cube and the approximate evaluation of integrals," *USSR Computational Mathematics and Mathematical Physics*, vol. 7, no. 4, pp. 86–112, 1967. DOI: `10.1016/0041-5553(67)90144-9`.
  - S. Joe and F. Y. Kuo, "Constructing Sobol Sequences with Better Two-Dimensional Projections," *SIAM Journal on Scientific Computing*, vol. 30, no. 5, pp. 2635–2654, 2008. DOI: `10.1137/070709359`.
  - A. B. Owen, "Randomly permuted $(t,m,s)$-nets and $(t,s)$-sequences," *Lecture Notes in Statistics*, vol. 106, pp. 299–317, 1995. DOI: `10.1007/978-1-4612-2552-2_19`.
  - J. Matoušek, "On the $L_2$-discrepancy for anchored boxes," *Journal of Complexity*, vol. 14, no. 4, pp. 527–556, 1998. DOI: `10.1006/jcom.1998.0489`.
- **Direction Numbers & Dimension Cap:** Embed Joe & Kuo (2008) `new-joe-kuo-6.21201` direction polynomials/numbers in Rust (`crates/jssp-core`), supporting **maximum dimension $d_{\max} = 21,201$** and **32-bit Gray-code sequence length up to $2^{32} - 1$** points.
- **Sequence Indexing & Power-of-Two Net Balance:**
  - **Scrambled Sobol (`A07`, `A11`, `A23`, `A24`):** Evaluated on the contiguous 0-based prefix $i \in \{0, 1, \dots, 63\}$ ($N = 64 = 2^6$). Because Matoušek/Owen scrambling randomizes point $i = 0$ to $\tilde{u}_0 \in [0, 1)^L$, the 64-point prefix forms an exact dyadic $(t, 6, s)$-net in $[0, 1)^L$.
  - **Unscrambled Sobol (`A06`, `A10`, `A21`, `A22`):** Evaluated on the contiguous shifted prefix $i \in \{1, 2, \dots, 64\}$ ($N = 64 = 2^6$). Index $i = 0$ is skipped because $(0,\dots,0)$ deterministically maps to the degenerate lexicographic job sequence $[0,\dots,0, 1,\dots,1, \dots, n-1,\dots,n-1]$; because shifting $\{0,\dots,63\} \to \{1,\dots,64\}$ drops $i = 0$ and introduces $i = 64$, unscrambled `A06`, `A10`, `A21`, `A22` do not retain strict 0-based $(t, 6, s)$-net balance at $i=64$ (this boundary is explicitly recorded).
- **Scrambling Algorithm (`A07`, `A11`, `A23`, `A24`):** Implement **Matoušek (1998) random linear matrix + XOR digital shift** over $\mathbb{F}_2^{32}$ (equivalent in second-order variance/net-preservation properties to Owen's nested uniform scrambling while running in $O(32 L)$ time and space without storing an exponential tree):
  - For each dimension $s$, draw an invertible lower-triangular $32 \times 32$ binary matrix $M_s$ (ones on diagonal, random bits below diagonal) and a random 32-bit digital shift $c_s$ from the `qmc_scramble` stream:
    $$\tilde{y}_{i,s} = (M_s \cdot y_{i,s}) \oplus c_s \pmod{2}, \qquad \tilde{u}_{i,s} = 2^{-32}\,\tilde{y}_{i,s} \in [0, 1)$$
- **Power-of-Two Population Size:** Freeze baseline total population size at **$N = 64 = 2^6$** across Phase A and Phase C so full-population Sobol (`N_S = 64`) and equal/quarter-Sobol mixes (`N_S = 32 = 2^5`, `N_S = 16 = 2^4`) realize exact dyadic sample sizes. (See Section 5 for the non-power-of-two `N_S = 48` case in `A13`.)

### 4.1 High-Dimensional Strategy ($L > 21,201$)
- Standard benchmarks in Track 1 have at most $L = 100 \times 20 = 2,000 \ll 21,201$, and Track 2a ($< 1000\times 1000$) fits within $d_{\max} = 21,201$ for sizes up to e.g. $200\times 100 = 20,000$ (note: larger Track 2a strata such as $500 \times 20$ have $10,000 \le 21,201$, while $500\times 100 = 50,000 > 21,201$ and Track 2b $1000\times 1000$ has $L = 10^6$).
- **Policy:**
  1. Standard `Sobol` (`A06`, `A07`, `A10`–`A14`, `A21`–`A24`) is **strictly restricted to $L \le 21,201$**; attempting to run standard Sobol on $L > 21,201$ returns `JsspError::UnsupportedQmcDimension`.
  2. For $L > 21,201$, any stress-track QMC test must use an explicitly distinct variant named **`Sobol-BlockRecycled`** (consuming blocks of $21,201$ coordinates with independent per-block digital shifts) and must **never** claim $(t,m,s)$-net discrepancy guarantees across recycled dimensions.

---

## 5. Mixed Halton / Sobol Populations (`A12`, `A13`, `A14`)

- **Population-Level Contiguous Prefix Allocation:** Assign $N_H$ individuals generated by **Scrambled Halton** on the 1-based contiguous prefix $i \in \{1, 2, \dots, N_H\}$ and $N_S = N - N_H$ individuals generated independently by **Scrambled Sobol** on the 0-based contiguous prefix $i \in \{0, 1, \dots, N_S - 1\}$ (using separate generator instances and `qmc_scramble` substreams). **Never** average or interleave raw coordinates.
- **Exact Counts, Sequence Indices & Digital-Net Balance Status for $N = 64$:**
  - `A12` (50:50): $N_H = 32$ (Halton $i \in \{1,\dots,32\}$) + $N_S = 32$ (Sobol $i \in \{0,\dots,31\}$). Here $N_S = 32 = 2^5$ **is a power of two**, so the 32-point Sobol subset retains exact $(t, 5, s)$ dyadic digital-net balance.
  - `A13` (25:75): $N_H = 16$ (Halton $i \in \{1,\dots,16\}$) + $N_S = 48$ (Sobol $i \in \{0,\dots,47\}$). Here $N_S = 48 = 3 \times 2^4$ is **NOT a power of two**. A 48-point contiguous prefix ($i \in \{0,\dots,47\}$) does **not** inherit the standard dyadic $(t, m, s)$-net balance guarantee of a 32-point ($2^5$) or 64-point ($2^6$) Sobol net; `A13` is retained strictly as an **empirical mixed-QMC configuration without the standard digital-net balance guarantee**.
  - `A14` (75:25): $N_H = 48$ (Halton $i \in \{1,\dots,48\}$) + $N_S = 16$ (Sobol $i \in \{0,\dots,15\}$). Here $N_S = 16 = 2^4$ **is a power of two**, so the 16-point Sobol subset retains exact $(t, 4, s)$ dyadic digital-net balance.
- **General Rounding Rule (if $N$ changes):** $N_H = \mathrm{round}(r_H \cdot N)$, $N_S = N - N_H$. Any non-$2^m$ value of $N_S$ must be explicitly flagged as lacking digital-net balance guarantees.

---

## 6. Experimental Validity & Discrepancy Caveats

Continuous low-discrepancy theorems (e.g., Koksma–Hlawka inequality) apply to functions of bounded variation on $[0, 1)^d$. Under sequential discrete OBR mapping:
1. The mapping $u \mapsto \pi$ is **stepwise discontinuous** and **state-dependent** (because $R_t$ and $W_j(t)$ depend on all preceding choices $u_0, \dots, u_{t-1}$).
2. Therefore, **low discrepancy in $[0,1)^L$ does NOT automatically imply low discrepancy or maximal diversity in OBR schedule space**.
3. Phase A explicitly evaluates empirical population quality (`best`, `mean`, `median` $C_{\max}$), **normalized pairwise Hamming diversity**, **chromosome duplicate rate**, **schedule duplicate rate**, and **empirical 1D/2D projection uniformity diagnostics** rather than assuming theoretical QMC bounds transfer.

---

## 7. Pseudocode for Sequential Discrete Inverse-CDF Mapping

```text
function qmc_point_to_obr(instance, u, alpha):
    n = instance.num_jobs(); L = instance.total_ops()
    rem_ops = [instance.job_len(j) for j in 0..n]
    next_op = [0; n]
    rem_work = [sum(instance.proc_time(j, k) for k in 0..rem_ops[j]) for j in 0..n]
    pi = [0; L]

    for t in 0..L:
        eligible = [j for j in 0..n if rem_ops[j] > 0]       // strictly ascending JobId
        if alpha == 0.0:
            weights = [1.0 for _ in eligible]
        else:
            weights = [(rem_work[j] as f64).powf(alpha) for j in eligible]

        total_w = sum(weights)
        if total_w <= 0.0:
            weights = [1.0 for _ in eligible]
            total_w = eligible.len() as f64

        // Inverse-CDF selection in [0, 1)
        target = u[t].clamp(0.0, 1.0 - f64::EPSILON) * total_w
        cum = 0.0
        chosen = eligible[eligible.len() - 1]
        for idx, j in enumerate(eligible):
            cum += weights[idx]
            if target < cum:
                chosen = j
                break

        pi[t] = chosen
        op = instance.op_id(chosen, next_op[chosen])
        rem_work[chosen] -= instance.proc_time(op)
        next_op[chosen] += 1
        rem_ops[chosen]  -= 1
    return pi
```

---

## 8. Primary References
- J. H. Halton, "On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals," *Numerische Mathematik*, vol. 2, pp. 84–90, 1960. DOI: `10.1007/BF01386213`
- I. M. Sobol', "On the distribution of points in a cube and the approximate evaluation of integrals," *USSR Computational Mathematics and Mathematical Physics*, vol. 7, no. 4, pp. 86–112, 1967. DOI: `10.1016/0041-5553(67)90144-9`
- A. B. Owen, "Randomly permuted $(t,m,s)$-nets and $(t,s)$-sequences," in *Monte Carlo and Quasi-Monte Carlo Methods in Scientific Computing*, Lecture Notes in Statistics, vol. 106, pp. 299–317, Springer, 1995. DOI: `10.1007/978-1-4612-2552-2_19`
- J. Matoušek, "On the $L_2$-discrepancy for anchored boxes," *Journal of Complexity*, vol. 14, no. 4, pp. 527–556, 1998. DOI: `10.1006/jcom.1998.0489`
- S. Joe and F. Y. Kuo, "Constructing Sobol Sequences with Better Two-Dimensional Projections," *SIAM Journal on Scientific Computing*, vol. 30, no. 5, pp. 2635–2654, 2008. DOI: `10.1137/070709359`
