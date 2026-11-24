# Dual-View Memetic Architecture: Fixed Gate & Adaptive Gate (`D06`)

**Decision ID:** `D06` (`PROVISIONAL`)
**Scope:** Proposed Dual-View representation (`C05`–`C11`), comparison of Pair-Level Selection Gate (Model A) vs. Reproductive-Slot Gate (Model B), deterministic rule-based adaptive gate, budget accounting, ablation controls (`C09`–`C11`), and literature novelty positioning.

---

## 1. Related-Work Positioning & Novelty Assessment

To ensure strict research integrity, the Dual-View architecture is positioned explicitly against established memetic and evolutionary literature:

1. **Lamarckian vs. Baldwinian Inheritance:** Whitley, Gordon & Mathias (1994, DOI `10.1007/3-540-58484-6_245`) established that Lamarckian inheritance accelerates convergence via genotype write-back while Baldwinian evaluation preserves genotypic diversity by smoothing the fitness landscape without altering the chromosome.
2. **Partial Lamarckianism:** Houck, Joines, Kay & Wilson (1997, DOI `10.1162/evco.1997.5.1.31`) hybridized Lamarckian and Baldwinian updates by probabilistically applying Lamarckian write-back to a fixed fraction of individuals ($p_L \in [0, 1]$) on a **single genotype** (`C04`).
3. **Meta-Lamarckian / Adaptive Meme Selection:** Ong & Keane (2004, DOI `10.1109/TEVC.2003.819944`) introduced Meta-Lamarckian adaptive selection among multiple local-search operators using empirical improvement rewards on single individuals.
4. **Self-Adaptive Learning Proportions:** El-Mihoub, Hopgood & Nolle (2021, online 2020, DOI `10.1007/s12065-020-00425-5`) directly addressed adaptive Lamarckian/Baldwinian mixing and local-search frequency by embedding **self-adaptive control meta-genes** inside each individual's chromosome. Unlike our proposal, El-Mihoub et al. operate on a single problem genotype per individual whose inherited locus can be overwritten by Lamarckian updates or kept by Baldwinian steps, rather than maintaining two persistent paired genotypes sharing an identity.

### Structured Novelty Assessment Table

| Category | Specific Concept / Element | Grounding & Positioning |
|---|---|---|
| **Already established ideas** | Lamarckian write-back (`C02`), Baldwinian phenotype fitness (`C03`), Partial Lamarckian mixing (`C04`), adaptive Lamarckian/Baldwinian probability control (El-Mihoub et al., 2021), multi-population co-evolution (`C10`) | Explicitly attributed to Whitley et al. (1994), Houck et al. (1997), Ong & Keane (2004), and El-Mihoub et al. (2021). |
| **Similar but distinct prior methods** | El-Mihoub et al. (2021) self-adaptive meta-genes; dual-population memetic GAs without individual identity pairing | El-Mihoub et al. adapt probabilities per-chromosome on a single genotype; dual-population GAs evolve two independent pools without persistent individual pairing (`C10` control). |
| **Potentially novel design elements** | Persistent **identity-paired** Dual-View individual $I_i = (\mathrm{ID}_i, G_i^L, G_i^B)$ with strictly **within-view genetic lineages** ($G^L \times G^L$, $G^B \times G^B$) coordinated by a **pair-level gate $p_t$** | Isolates the hypothesized advantage of co-maintaining paired Lamarckian and Baldwinian trajectories per individual without cross-view crossover contamination. |
| **Unverified novelty hypotheses** | Whether persistent pairing + adaptive gating (`C06`) outperforms single-view memetic GAs (`C02`–`C04`), Fixed Gate (`C05`), and Unpaired Dual-Population (`C10`) under equal genotype capacity and evaluation budgets | Treated strictly as a falsifiable empirical hypothesis to be tested in Phase C—not claimed as established superiority prior to confirmatory results. |

---

## 2. Formal Dual-View Individual State

A Dual-View population of $N_{\text{pairs}} = 32$ paired individuals contains **$2 N_{\text{pairs}} = 64$ total OBR genotypes** (matching the $N = 64$ single-view GA population size). Each paired individual at generation $t$ is defined as:
$$I_i(t) = \big(\mathrm{ID}_i,\; G_i^L(t),\; G_i^B(t)\big), \qquad i \in \{0, \dots, N_{\text{pairs}} - 1\}$$
- **Initialization ($t = 0$):** Both genotypes originate from the same initial OBR chromosome: $G_i^L(0) = G_i^B(0) = \pi_i^{(0)}$, assigned unique persistent identity $\mathrm{ID}_i$.
- **Lamarckian View ($G_i^L$):** Local search $\mathrm{LS}_k(G_i^L)$ produces improved schedule $S_i^L$ and re-encoded chromosome $\tilde{G}_i^L$. **Genotype write-back occurs:** $G_i^L \leftarrow \tilde{G}_i^L$, and fitness is $f_i^L = C_{\max}(S_i^L)$.
- **Baldwinian View ($G_i^B$):** Local search $\mathrm{LS}_k(G_i^B)$ evaluates locally improved phenotype $S_i^B$ for fitness $f_i^B = C_{\max}(S_i^B)$, while **$G_i^B$ remains unmodified** in the genotype pool.
- **Lineage Separation Invariant (`C05`–`C09`):** Crossover and mutation occur **strictly within view** ($G^L \times G^L \to G_{\text{child}}^L$ and $G^B \times G^B \to G_{\text{child}}^B$). Cross-view crossover is prohibited in base Dual-View designs; migration is isolated exclusively in `C11`.

---

## 3. Gate Architecture Comparison: Model A vs. Model B

We evaluate the two candidate gate architectures across all 10 required dimensions:

| Dimension | Model A — Pair-Level Selection Gate *(Selected)* | Model B — Reproductive-Slot Gate *(Rejected)* |
|---|---|---|
| **1. Fitness calculation** | Both views are locally evaluated per generation: $f_i^L = C_{\max}(\mathrm{LS}(G_i^L))$, $f_i^B = C_{\max}(\mathrm{LS}(G_i^B))$. | Both views must still be tracked/evaluated to update adaptive performance signals and maintain pair validity. |
| **2. Normalization** | Min-max normalization across the current population ($S_i^v \in [0, 1]$, higher = better makespan; constant $1.0$ if all equal). | Either separate view normalization or joint rank normalization across $2 N_{\text{pairs}}$ genotypes. |
| **3. Parent selection** | Binary tournament over **paired individuals** $I_i$ using scalar pair score $S_i(t) = p_t S_i^L(t) + (1 - p_t) S_i^B(t)$. | Independent view selection; gate $p_t$ probabilistically routes each offspring slot to emit either an $L$-offspring or $B$-offspring. |
| **4. Crossover** | Selected parent pairs $(I_a, I_b)$ execute **parallel within-view POX**: $G_a^L \times G_b^L \to G_c^L$ and $G_a^B \times G_b^B \to G_c^B$. | Variable-capacity pools require asymmetric cloning or view dropout when $p_t \neq 0.5$, distorting equalized capacity. |
| **5. Mutation** | Within-view OBR swap/insert mutation applied independently to $G_c^L$ and $G_c^B$ ($p_m = 0.15$). | Applied only to the view that won the slot (or cloned into the inactive view). |
| **6. Local Search** | Applied within-budget to both views ($b_{\text{LS}} = 50$ evals each, shared budget counter). | Unequal slot counts skew per-view local search frequency, confounding gate weight with LS budget share. |
| **7. Genotype inheritance** | Lamarckian child inherits write-back $\tilde{G}_c^L$; Baldwinian child preserves pre-LS $G_c^B$. | Non-selected view in a slot either stalls or duplicates, violating balanced dual-lineage evolution. |
| **8. Baldwinian phenotype eval** | $f_c^B = C_{\max}(\mathrm{LS}(G_c^B))$ recorded for next generation's $S_c^B$ without altering $G_c^B$. | Same, except unallocated view slots lose selection feedback. |
| **9. Pair identity preservation** | **Exact 1:1:** Every offspring pair $(G_c^L, G_c^B)$ inherits a coherent child pair $\mathrm{ID}_c$ co-descended from $(I_a, I_b)$. | **Broken or asymmetric:** Unequal slot counts break 1:1 `(ID, L, B)` pairing after generation 1. |
| **10. Computational budget** | Deterministic fixed cost per generation ($2 N_{\text{pairs}} = 64$ genotypes evaluated), identical across `C05`–`C09`, `C11`. | Budget per view drifts dynamically with $p_t$, confounding gate effects with search allocation. |

**Recommendation (`D06`, `PROVISIONAL`):** Adopt **Model A (Pair-Level Selection Gate)** as the sole Dual-View base architecture because it preserves exact `1:1` `(ID_i, G_i^L, G_i^B)` pairing, strictly forbids cross-view crossover, prevents either view from going extinct, and keeps genotype capacity and evaluation budget strictly invariant across `p_t`.

---

## 4. Exact Equations for Model A Gate & Adaptive Controller

### 4.1 Normalized View Scores & Pair-Level Score
At generation $t$, let $f_{\min}^v(t) = \min_i f_i^v(t)$ and $f_{\max}^v(t) = \max_i f_i^v(t)$ for view $v \in \{L, B\}$ (or global bounds $f_{\min}(t) = \min(f_{\min}^L(t), f_{\min}^B(t))$ and $f_{\max}(t) = \max(f_{\max}^L(t), f_{\max}^B(t))$ across both views so $S_i^L$ and $S_i^B$ remain on a **commensurable scale**!):
$$S_i^v(t) = \begin{cases}
1.0, & \text{if } f_{\max}(t) = f_{\min}(t), \\[0.6em]
\dfrac{f_{\max}(t) - f_i^v(t)}{f_{\max}(t) - f_{\min}(t)}, & \text{otherwise,}
\end{cases} \qquad v \in \{L, B\}$$
*(Using shared population-wide $(f_{\min}(t), f_{\max}(t))$ ensures that if one view consistently discovers strictly lower $C_{\max}$, its normalized score is strictly higher rather than masked by per-view rescaling.)*

The **pair-level selection score** is:
$$S_i(t) = p_t\, S_i^L(t) + (1 - p_t)\, S_i^B(t), \qquad p_t \in [p_{\min}, p_{\max}] \subseteq [0, 1]$$
- **Unambiguous Gate Convention:** $p_t$ always weights the **Lamarckian view ($G^L$)**, and $(1 - p_t)$ weights the **Baldwinian view ($G^B$)**.
- **Parent Selection Tie-Break:** In binary tournament between $I_a$ and $I_b$, select larger $S_i(t)$; tie-break by smaller $\min(f_i^L(t), f_i^B(t))$, then smaller $\mathrm{ID}_i$.
- **Elitism:** Top $E_{\text{pairs}} = 1$ pair ($= 2$ genotypes, matching `C01`–`C04` elitism $E = 2$) ordered by $\min(f_i^L(t), f_i^B(t))$ (tie-break larger $S_i(t)$, smaller $\mathrm{ID}_i$) is copied unchanged to generation $t+1$.

### 4.2 Fixed Gate Configurations (`C05`, `C07`, `C08`)
- **`C05`:** $p_t \equiv 0.50$ (equal weight on Lamarckian and Baldwinian views)
- **`C07`:** $p_t \equiv 0.25$ (Baldwinian-favored fixed sensitivity)
- **`C08`:** $p_t \equiv 0.75$ (Lamarckian-favored fixed sensitivity)

### 4.3 Deterministic Rule-Based Adaptive Gate (`C06` & `C09`)
No machine learning, bandits, or neural policies are used. The controller computes two bounded dimensionless signals at every generation $t \ge 1$ ($\Delta t = 1$):

1. **Performance-Efficiency Signal $\Delta_{\text{perf}}(t) \in [-1, 1]$:**
   Let $\bar{\delta}^v(t)$ be the mean makespan reduction achieved per local-search evaluation in view $v \in \{L, B\}$ at generation $t$:
   $$\bar{\delta}^v(t) = \frac{\sum_{i=0}^{N_{\text{pairs}}-1} \max\!\big(0,\, C_{\max}(G_i^v) - f_i^v(t)\big)}{\max\!\big(1,\, \mathrm{LS\_Evals}^v(t)\big)} \quad \text{combined with view mean advantage } \bar{S}^L(t) - \bar{S}^B(t)$$
   Specifically, define raw relative advantage combining mean quality advantage and LS improvement yield:
   $$\Delta_{\text{perf}}(t) = \frac{1}{2}\Big(\bar{S}^L(t) - \bar{S}^B(t)\Big) + \frac{1}{2}\left(\frac{\bar{\delta}^L(t) - \bar{\delta}^B(t)}{\bar{\delta}^L(t) + \bar{\delta}^B(t) + \epsilon}\right) \in [-1, 1]$$
   where $\epsilon = 10^{-9}$, $\bar{S}^v(t) = \frac{1}{N_{\text{pairs}}}\sum_i S_i^v(t) \in [0, 1]$, and positive values favor $G^L$ while negative values favor $G^B$.
2. **Genotypic Diversity Signal $\Delta_{\text{div}}(t) \in [-1, 1]$:**
   Let $D_{\text{Ham}}(G^v(t)) \in [0, 1]$ be the mean normalized pairwise Hamming distance among genotypes in view $v$:
   $$D_{\text{Ham}}(G^v(t)) = \frac{2}{N_{\text{pairs}}(N_{\text{pairs}} - 1)} \sum_{0 \le i < j < N_{\text{pairs}}} \frac{1}{L}\sum_{\ell=0}^{L-1} \mathbb{I}\!\big[G_i^v[\ell] \neq G_j^v[\ell]\big]$$
   $$\Delta_{\text{div}}(t) = D_{\text{Ham}}(G^L(t)) - D_{\text{Ham}}(G^B(t)) \in [-1, 1]$$
   When Lamarckian write-back collapses $G^L$ diversity ($D_{\text{Ham}}(G^L) < D_{\text{Ham}}(G^B)$), $\Delta_{\text{div}}(t) < 0$, steering $p_t$ toward the more diverse Baldwinian view.
3. **Smoothed Bounded Update:**
   $$g(t) = w_{\text{perf}}\, \Delta_{\text{perf}}(t) + w_{\text{div}}\, \Delta_{\text{div}}(t), \qquad \tilde{g}(t) = (1 - \lambda)\,\tilde{g}(t-1) + \lambda\, g(t), \quad \tilde{g}(0) = 0$$
   $$p_{t+1} = \begin{cases}
   p_t, & \text{if } |\tilde{g}(t)| < 10^{-12} \text{ (deterministic tie-break hold)}, \\
   \mathrm{clip}\!\big(p_t + \eta\, \tilde{g}(t),\; p_{\min},\; p_{\max}\big), & \text{otherwise.}
   \end{cases}$$

**Provisional Parameters (Frozen Pre-Confirmatory unless adjusted on Dev Split):**
- $p_0 = 0.50$, $[p_{\min}, p_{\max}] = [0.15, 0.85]$ (prevents view extinction while permitting strong $85:15$ emphasis).
- EWMA smoothing $\lambda = 0.30$ (effective memory window $\approx 5.7$ generations); step rate $\eta = 0.05$ (prevents bang-bang oscillations).
- **`C06` Weights:** $w_{\text{perf}} = 0.70$, $w_{\text{div}} = 0.30$.
- **`C09` (Diversity-Signal Ablation):** $w_{\text{perf}} = 1.00$, $w_{\text{div}} = 0.00$ (isolates the causal effect of $\Delta_{\text{div}}(t)$).

---

## 5. Control & Exploratory Configurations (`C10` and `C11`)

- **`C10` (Unpaired Dual-Population Control):** Runs two independent subpopulations of size $32$ each ($32$ LGA + $32$ BGA $= 64$ genotypes total). No shared $\mathrm{ID}_i$, no pair-level selection gate, and no cross-population crossover; parent selection occurs strictly within each subpopulation, sharing the exact same total evaluation budget $B_{\text{eval}}$. Directly tests whether **identity pairing + pair-level gating** improves upon merely splitting a budget into two independent memetic pools.
- **`C11` (Controlled Cross-View Migration — Exploratory Ablation):** Uses Model A (`C06` adaptive gate) plus periodic cross-view transfer every $T_{\text{mig}} = 10$ generations: identify the stronger view $v^*$ at generation $t$ (smaller best $C_{\max}$, tie-breaking toward $L$) and copy the top $k_{\text{mig}} = \max(1, \lfloor 0.10\, N_{\text{pairs}} \rfloor) = 3$ unique genotypes from $v^*$ into the worst $k_{\text{mig}}$ slots of the complementary view $\bar{v}$ (evaluating migrated chromosomes cleanly within budget). Strictly excluded from base Dual-View claims.

---

## 6. Complexity & Equalized Budget Accounting
- Per-generation time complexity of `C05`–`C09`: $O(N_{\text{pairs}} \cdot (L + b_{\text{LS}} \cdot C_{\text{LS}}))$, identical in order and genotype count ($64$) to `C01`–`C04` ($N = 64$).
- **Invariant:** Every genotype decode and every neighborhood move evaluation in both $G^L$ and $G^B$ increments the **single shared `eval_counter`**. Run termination checks `eval_counter >= B_eval`, ensuring Dual-View does **not** perform $2\times$ evaluations relative to `C01`–`C04`.

---

## 7. Primary References
- T. A. El-Mihoub, A. A. Hopgood, and L. Nolle, "Self-adaptive learning for hybrid genetic algorithms," *Evolutionary Intelligence*, vol. 14, no. 4, pp. 1565–1579, 2021 (online 2020). DOI: `10.1007/s12065-020-00425-5`
- D. Whitley, V. S. Gordon, and K. Mathias, "Lamarckian evolution, the Baldwin effect and function optimization," in *Parallel Problem Solving from Nature — PPSN III*, LNCS vol. 866, pp. 5–15, Springer, 1994. DOI: `10.1007/3-540-58484-6_245`
- C. R. Houck, J. A. Joines, M. G. Kay, and J. R. Wilson, "Empirical Investigation of the Benefits of Partial Lamarckianism," *Evolutionary Computation*, vol. 5, no. 1, pp. 31–60, 1997. DOI: `10.1162/evco.1997.5.1.31`
- Y. S. Ong and A. J. Keane, "Meta-Lamarckian learning in memetic algorithms," *IEEE Transactions on Evolutionary Computation*, vol. 8, no. 2, pp. 99–110, 2004. DOI: `10.1109/TEVC.2003.819944`
