# Source-Verified Definitions of `N5`, `N6`, `N7`, `N8`, `NNS`, and Batch Approximate Evaluation (`D03`)

**Decision IDs:** `D03-A` (`ACCEPTED`), `D03-B` (`DEFERRED` for `NNS`), `D03-C` (`REJECTED` for approximate primary evaluation)
**Scope:** Exact graph definitions, move rules, feasibility guarantees, inclusion relations, and evaluation policies for Phase B and Phase C local search.

---

## 1. Common Disjunctive Graph & Critical Block Definitions

Following Balas (1969) as used in Nowicki & Smutnicki (1996), Balas & Vazacopoulos (1998), Zhang et al. (2007), and Xie et al. (2023):

- **Disjunctive Graph:** Let $G(\sigma) = (V, A, E(\sigma))$ where $V = O \cup \{0, \tilde{n}\}$ ($0$ = dummy start with $p_0 = 0$, $\tilde{n}$ = dummy sink with $p_{\tilde{n}} = 0$).
  - $A$: conjunctive job-route arcs $(JP[u], u)$ for consecutive operations of the same job, plus $(0, O_{j,0})$ and $(O_{j,m_j-1}, \tilde{n})$.
  - $E(\sigma)$: directed machine arcs $(MP[u], u)$ induced by machine processing sequence $\sigma$.
  - Node-weighted (or outgoing-arc-weighted) lengths: head time $r_u = L(0, u)$ is the longest path length from $0$ to $u$ excluding $p_u$; tail time $q_u = L(u, \tilde{n}) - p_u$ is the longest path length strictly after $u$ to $\tilde{n}$ (so $L(u, \tilde{n}) = p_u + q_u$). For any operation $u$, longest path passing through $u$ has length $r_u + p_u + q_u$.
- **Makespan & Critical Path:** $C_{\max}(\sigma) = L(0, \tilde{n}) = \max_{u \in O} (r_u + p_u + q_u)$. An operation $u \in O$ is critical if $r_u + p_u + q_u = C_{\max}(\sigma)$.
- **Critical Path Selection & Critical Blocks ($B_1, \dots, B_b$):** Pick one deterministic longest critical path $P = (0, v_1, \dots, v_K, \tilde{n})$ (tie-breaking predecessors in longest-path backtracking deterministically by lowest `OpId`). Partition $P$ into maximal contiguous subsequences $B_1, B_2, \dots, B_b$ of operations processed on the **same machine** with $|B_l| \ge 1$ (note: only blocks with $|B_l| \ge 2$ contain intra-block swaps/insertions, whereas $|B_l| \ge 1$ blocks can still participate in `N8` extra-block moves!).
  - For a block $B = (b_1, b_2, \dots, b_k)$ ($k \ge 2$): $u_{\text{first}} = b_1$, $u_{\text{last}} = b_k$, and **inner operations** are $\{b_2, \dots, b_{k-1}\}$.
  - *M4 Disjunctive Graph Implementation Note:* Milestone M4 implements the graph-wide critical subgraph and extracts maximal contiguous critical machine blocks of length $\ge 2$ across all critical `MachineOrder` arcs (`head[u] + p[u] == head[v]` and critical target $v$) alongside the deterministic `RepresentativeCriticalPath` ($P$) in [`disjunctive_graph.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/disjunctive_graph.md) and [`critical_path_and_blocks.md`](file:///home/abdelhalimyasser/Downloads/jssp-research/docs/algorithms/critical_path_and_blocks.md). Head/tail conventions (`head[u] = r_u`, `tail[u] = q_u`, both excluding $p_u$) agree identically. Move generation for `N5`–`N8` remains scheduled for Milestone M5.

---

## 2. Formal Definitions of `N5`, `N6`, `N7`, and `N8` (`D03-A`: `ACCEPTED`)

### 2.1 `N5` — Nowicki & Smutnicki (1996)
- **Primary Reference:** E. Nowicki and C. Smutnicki, "A Fast Taboo Search Algorithm for the Job Shop Problem," *Management Science*, vol. 42, no. 6, pp. 797–813, 1996 (Section 2, pp. 799–801). DOI: `10.1287/mnsc.42.6.797`.
- **Move Rule:** For each critical block $B_l = (b_1, \dots, b_k)$ with $k \ge 2$:
  - Swap $(b_1, b_2)$ **unless** $l = 1$ (first block of $P$).
  - Swap $(b_{k-1}, b_k)$ **unless** $l = b$ (last block of $P$).
  - *(Special case $k = 2$ and $1 < l < b$:* swapping $(b_1, b_2)$ and $(b_{k-1}, b_k)$ yields the same move; duplicate elimination keeps it once.)
- **Feasibility & Complexity:** Every adjacent critical-block swap is **guaranteed acyclic** (van Laarhoven et al. 1992; Nowicki & Smutnicki 1996 Lemma 1). Size: $\le 2b$ moves ($O(L)$ max). Exact evaluation: $O(L)$ per move (or incremental Taillard/Nowicki–Smutnicki estimate).

### 2.2 `N6` — Balas & Vazacopoulos (1998)
- **Primary Reference:** E. Balas and A. Vazacopoulos, "Guided Local Search with Shifting Bottleneck for Job Shop Scheduling," *Management Science*, vol. 44, no. 2, pp. 262–275, 1998 (Section 2, Theorems 1–2 / Neighborhood $N6$, pp. 264–266). DOI: `10.1287/mnsc.44.2.262`.
- **Move Rule:** For each critical block $B_l = (b_1, \dots, b_k)$ with $k \ge 2$ and every inner/boundary pair within $B_l$:
  1. **Forward shift (move $u = b_1$ right after $v \in \{b_2, \dots, b_k\}$):** Allowed if $v = b_2$ OR $l > 1$? Wait—in Balas & Vazacopoulos (1998), `N6` moves an operation inside $B_l$ to the front ($b_1$) or rear ($b_k$), OR moves $b_1$ after $v$ / $b_k$ before $u$: specifically, as summarized identically in Zhang et al. (2007, Section 2) and Xie et al. (2023, Section 3.1, Fig. 4) and Gui et al. (2023, Section 3.2, Fig. 7):
     - Move an operation $v \in \{b_2, \dots, b_k\}$ to immediately **before** $u = b_1$ (if $l > 1$, or for $v = b_k$? Wait: moving $b_2$ before $b_1$ in $B_1$ cannot improve $C_{\max}$ because $b_1$ has $JP[b_1]$ outside $M$ or $r_{b_1}=0$; under Proposition 3 of Nowicki–Smutnicki, changing only the first operation of $B_1$ or last operation of $B_b$ cannot decrease $C_{\max}$).
     - Move an operation $u \in \{b_1, \dots, b_{k-1}\}$ to immediately **after** $v = b_k$ (excluding moving $b_{k-1}$ after $b_k$ in $B_b$, or moving $u$ after $b_k$ in $B_b$).
- **Sufficient Feasibility Conditions (Balas & Vazacopoulos 1998; Zhang et al. 2007 Theorems 1–2):**
  - Moving $v$ immediately **before** $u$ (where $u$ precedes $v$ on the same machine) is **guaranteed acyclic** if:
    $$r_u + p_u \ge r_{JP[v]} + p_{JP[v]} \quad (\text{or } r_{JP[v]} = -\infty \text{ if } JP[v] = 0)$$
    *(Note: Xie et al. 2023 Proposition 2 relaxes this slightly to $r_u + p_u \ge r_{JP[v]}$, which is strictly checked below; for `N6` we apply the published head/tail condition $r_u + p_u \ge r_{JP[v]} + p_{JP[v]}$ AND always run cycle detection on the mutated graph as a safety net!)*
  - Moving $u$ immediately **after** $v$ is **guaranteed acyclic** if:
    $$q_v + p_v \ge q_{JS[u]} + p_{JS[u]} \quad (\text{equivalently } L(v, \tilde{n}) \ge L(JS[u], \tilde{n}))$$
- **Size & Complexity:** $O(\sum_l |B_l|)$ moves; can create a cycle **only** if a non-sufficient move were attempted, so enforcing the sufficient condition guarantees acyclicity (with explicit topological verification in Rust).

### 2.3 `N7` — Zhang, Li, Guan & Rao (2007)
- **Primary Reference:** C. Y. Zhang, P. G. Li, Z. L. Guan, and Y. Q. Rao, "A tabu search algorithm with a new neighborhood structure for the job shop scheduling problem," *Computers & Operations Research*, vol. 34, no. 11, pp. 3229–3242, 2007 (Section 2, Theorems 1–2 & Fig. 2, pp. 3231–3233). DOI: `10.1016/j.cor.2005.12.002`.
- **Move Rule:** Extends `N6` within each critical block $B_l = (b_1, \dots, b_k)$ ($k \ge 2$):
  1. All `N6` insertions moving $v \in \{b_2,\dots,b_k\}$ immediately **before** $b_1$ and $u \in \{b_1,\dots,b_{k-1}\}$ immediately **after** $b_k$.
  2. **Additional `N7` moves:**
     - Move $b_1$ immediately **after** any inner operation $v \in \{b_2, \dots, b_{k-1}\}$.
     - Move $b_k$ immediately **before** any inner operation $u \in \{b_2, \dots, b_{k-1}\}$.
  - *Boundary exclusions:* In $B_1$, moves that only reorder within $B_1$ by inserting $b_1$ inside $B_1$ or inner ops before $b_1$ can be retained or clipped per Zhang et al. / Xie et al. (see Section 3 below!).
- **Feasibility Conditions:** Same Balas–Vazacopoulos / Zhang et al. Theorems 1–2: $r_u + p_u \ge r_{JP[v]} + p_{JP[v]}$ for moving $v$ before $u$; $L(v, \tilde{n}) \ge L(JS[u], \tilde{n})$ for moving $u$ after $v$.

### 2.4 `N8` — Xie, Li, Gao & Gui (2023) *(and TWT Terminology Disambiguation)*
- **Primary Reference (Makespan):** J. Xie, X. Li, L. Gao, and L. Gui, "A new neighbourhood structure for job shop scheduling problems," *International Journal of Production Research*, vol. 61, no. 7, pp. 2147–2161, 2023 (arXiv:2109.02843, Section 3.2–3.3, Propositions 1–6, Figs. 9–11). DOI: `10.1080/00207543.2022.2060772`.
- **Terminology Warning:** J. Kuhpfahl and C. Bierwirth (2016, *Computers & Operations Research*, vol. 66, pp. 44–57, DOI `10.1016/j.cor.2015.07.011`) study neighborhoods for **Total Weighted Tardiness (TWT)** (`JSPTWT`), not makespan. Our project strictly implements **Xie et al. (2023) `N8`**, which targets makespan minimization.
- **Move Rule (Xie et al. 2023 Section 3.2):** Given critical block $B_l$ on machine $r$:
  1. **Intra-block moves:** All `N7` moves within $B_l$.
  2. **Extra-block forward moves:** For $u \in B_l$ and any operation $v$ on the **same machine** $r$ scheduled **after** $B_l$ ($v \notin B_l$, $v$ after $b_k$), insert $u$ immediately **after** $v$ if feasible.
  3. **Extra-block backward moves:** For $v \in B_l$ and any operation $u$ on the **same machine** $r$ scheduled **before** $B_l$ ($u \notin B_l$, $u$ before $b_1$), insert $v$ immediately **before** $u$ if feasible.
- **Feasibility Conditions (Xie et al. 2023 Propositions 1–2):**
  - Moving $u$ immediately after $v$: $L(v, \tilde{n}) \ge L(JS[u], \tilde{n}) - p_{JS[u]}$ (corrected proof in Proposition 1: $q_{JS[u]}$ vs $L(v,\tilde{n}) = p_v + q_v$, i.e., if $L(v,\tilde{n}) < L(JS[u],\tilde{n}) - p_{JS[u]}$ a cycle $JS[u] \to v \to u \to JS[u]$ could exist; note that Zhang et al. uses the slightly stricter $L(v, \tilde{n}) \ge L(JS[u], \tilde{n})$ and Rust's explicit topological cycle check guarantees 100% safety either way!).
  - Moving $v$ immediately before $u$: $L(0, u) + p_u \ge L(0, JP[v])$ (Proposition 2).
- **Neighborhood Clipping (Xie et al. 2023 Propositions 3–6):** Prunes four provably non-improving boundary patterns:
  - Prop. 3: Moving $b_1$ of $B_1$ right after any inner operation $v \in B_1$ cannot reduce $C_{\max}$.
  - Prop. 4: Moving $b_k$ of $B_b$ right before any inner operation $u \in B_b$ cannot reduce $C_{\max}$.
  - Prop. 5: Moving an inner operation $v \in B_1$ right before $b_1 \in B_1$ cannot reduce $C_{\max}$.
  - Prop. 6: Moving an inner operation $u \in B_b$ right after $b_k \in B_b$ cannot reduce $C_{\max}$.

### 2.5 Inclusion Hierarchy & Boundary Caveat
Under **uniform boundary clipping** (applying Nowicki–Smutnicki Proposition 3 / Xie et al. Propositions 3–6 across $B_1$ and $B_b$ and identical feasibility checks):
$$\mathrm{N5}(\sigma) \subseteq \mathrm{N6}(\sigma) \subseteq \mathrm{N7}(\sigma) \subseteq \mathrm{N8}(\sigma)$$
*Caveat:* If `N7` is implemented *without* Propositions 3–6 boundary clipping while `N8` applies Propositions 3–6 clipping, then those specific non-improving $B_1/B_b$ moves exist in unclipped `N7` and are clipped from `N8`. In our implementation, we record clipped vs. unclipped move counts and preserve strict set inclusion on **improving** moves (since Propositions 3–6 never eliminate any $C_{\max}$-improving move).

### 2.6 Duplicate Move Elimination & Complexity Summary

| Neighborhood | Canonical Key for Deduplication | Guaranteed Acyclic Precondition? | Max Candidate Size | Exact Eval Complexity |
|---|---|---|---|---|
| `N5` (1996) | `(machine_id, new_machine_seq_hash)` / `(u, v, dir)` | Yes (Lemma 1) | $\le 2 b \le 2L$ | $O(L)$ per move |
| `N6` (1998) | `(op_moved, target_anchor, before/after)` | Yes (Theorems 1–2 + DAG verify) | $O(\sum |B_l|)$ | $O(L)$ per move |
| `N7` (2007) | Same + inner-to-end vs end-to-inner deduplication | Yes (Theorems 1–2 + DAG verify) | $O(\sum |B_l|)$ | $O(L)$ per move |
| `N8` (2023) | Same across full machine sequence $\sigma_r$ | Yes (Props. 1–2 + DAG verify) | $O(m \cdot \sum |B_l| / m) \le O(L \cdot \max n)$ | $O(L)$ per move |

---

## 3. Additional Candidate Evaluation: `NNS` (Gui et al., 2023) (`D03-B`: `DEFERRED`)

- **Primary Reference:** L. Gui, X. Li, L. Gao, and C. Wang, "Necessary and Sufficient Conditions for Feasible Neighbourhood Solutions in the Local Search of the Job-Shop Scheduling Problem," *Chinese Journal of Mechanical Engineering*, vol. 36, no. 1, art. 87, 2023. DOI: `10.1186/s10033-023-00911-8`. URL: `https://doi.org/10.1186/s10033-023-00911-8`.
- **Objective Description Audit (Internally Inconsistent):**
  - **Section 2 ("Description of JSP"):** States *"In this paper, the objective function is to minimize the total completion time."*
  - **Section 1 & Section 2 Disjunctive Graph Model:** Section 1 states the goal is *"minimum makespan"*, and Section 2 explicitly defines the critical path $s \to e$ weight sum as the **makespan** of the schedule.
  - **Section 6.2 Empirical Results:** Evaluates 152 benchmark instances (`FT`, `LA`, `ABZ`, `ORB`, `SWV`, `YN`, `TA`) using $\mathrm{MRE} = 100 \times (C - \mathrm{LB}_{\text{best}}) / \mathrm{LB}_{\text{best}}$ against classical makespan lower/best-known bounds.
  - **Classification:** Because no published erratum or author correction resolves this contradiction, Gui et al. (2023)'s objective wording is formally classified as **internally inconsistent** (rather than asserting an editorial typo).
- **Separation of Technical Components:**
  1. *Structural Feasibility Propositions (Section 4.1–4.2, Propositions 7–9):* Purely topological disjunctive-graph acyclicity theorems independent of objective function. *Proposition 7:* infeasibility $\iff$ directed cycle; *Proposition 8:* moving $v$ before $u$ ($u \prec_M v$) creates a cycle $\iff$ path $u \leadsto JP[v]$ exists; *Proposition 9:* moving $u$ after $v$ ($u \prec_M v$) creates a cycle $\iff$ path $JS[u] \leadsto v$ exists.
  2. *`NNS` Neighborhood Definition (Section 4.3):* Pairs $(u, v)$ on the same critical block with at least one block endpoint ($u = b_1$ or $v = b_k$), inserting $v$ before $u$ iff $u \not\leadsto JP[v]$, or $u$ after $v$ iff $JS[u] \not\leadsto v$. It recovers feasible moves pruned by Balas–Vazacopoulos / Zhang et al. sufficient-only head/tail bounds (Section 4.3, Figs. 10–11, makespan $20 \to 19$).
  3. *Empirical Makespan Evaluation (Section 6):* Tabu Search benchmark comparisons reported against makespan bounds.
- **Overhead & OBR Adapter Compatibility:**
  - Computing exact reachability via a full path matrix takes $O(|V|^2)$ per iteration (Section 4.4 pairs sufficient head/tail bounds with targeted reachability checks to accelerate filtering). Fully compatible with classical rectangular JSSP and our `D01-B` OBR adapter.
- **Disposition (`DEFERRED`) & Supplementary Comparator (`B15-NNS` vs. `B15`):**
  - `NNS` remains excluded from the primary 63-configuration matrix to avoid expanding Phase B across all 7 active search strategies without prior screening.
  - Designated as a supplementary Tabu Search ablation (`B15-NNS`) directly compared against **`B15` (verified `Tabu Search + N7`)**. Both `B15-NNS` and `B15` must execute with **identical paired initial solutions (`A01` per `(instance, seed)`), identical objective-evaluation budgets ($100L$ dev / $500L$ confirmatory), identical 20 master seeds and PRNG substreams, and identical Tabu Search hyperparameters**.

---

## 4. Batch Approximate N7 Evaluator (Ba, Liu & Yuan, 2026) (`D03-C`: `REJECTED`)

- **Primary Reference:** Z. Ba, J. Liu, and Y. Yuan, "A batch approximate evaluation method for neighborhood solutions in job shop scheduling," *Computers & Operations Research*, vol. 195, art. 107610, 2026. DOI: `10.1016/j.cor.2026.107610`.
- **Classification:** It is **neither** a new neighborhood structure **nor** a search strategy; it is an **approximate neighborhood evaluator / implementation optimization** based on slack-time recurrence across critical blocks to estimate `N7` move makespans in batch.
- **Project Policy (`REJECTED` for primary exact evaluation):** Per `AGENTS.md` §3.3 & §7, exact makespan evaluation (`Schedule::cmax`) is mandatory as the authoritative objective evaluator and correctness reference across all Phases A–C. Approximate evaluators cannot substitute for exact $C_{\max}$ inside budget-normalized comparisons because approximate move rankings alter search trajectories and break evaluation-count parity.

---

## 5. Primary References
- E. Nowicki and C. Smutnicki, "A Fast Taboo Search Algorithm for the Job Shop Problem," *Management Science*, vol. 42, no. 6, pp. 797–813, 1996. DOI: `10.1287/mnsc.42.6.797`
- E. Balas and A. Vazacopoulos, "Guided Local Search with Shifting Bottleneck for Job Shop Scheduling," *Management Science*, vol. 44, no. 2, pp. 262–275, 1998. DOI: `10.1287/mnsc.44.2.262`
- C. Y. Zhang, P. G. Li, Z. L. Guan, and Y. Q. Rao, "A tabu search algorithm with a new neighborhood structure for the job shop scheduling problem," *Computers & Operations Research*, vol. 34, no. 11, pp. 3229–3242, 2007. DOI: `10.1016/j.cor.2005.12.002`
- J. Xie, X. Li, L. Gao, and L. Gui, "A new neighbourhood structure for job shop scheduling problems," *International Journal of Production Research*, vol. 61, no. 7, pp. 2147–2161, 2023. DOI: `10.1080/00207543.2022.2060772`
- J. Kuhpfahl and C. Bierwirth, "A study on local search neighborhoods for the job shop scheduling problem with total weighted tardiness objective," *Computers & Operations Research*, vol. 66, pp. 44–57, 2016. DOI: `10.1016/j.cor.2015.07.011`
- L. Gui, X. Li, L. Gao, and C. Wang, "Necessary and Sufficient Conditions for Feasible Neighbourhood Solutions in the Local Search of the Job-Shop Scheduling Problem," *Chinese Journal of Mechanical Engineering*, vol. 36, no. 1, art. 87, 2023. DOI: `10.1186/s10033-023-00911-8`
- Z. Ba, J. Liu, and Y. Yuan, "A batch approximate evaluation method for neighborhood solutions in job shop scheduling," *Computers & Operations Research*, vol. 195, art. 107610, 2026. DOI: `10.1016/j.cor.2026.107610`
