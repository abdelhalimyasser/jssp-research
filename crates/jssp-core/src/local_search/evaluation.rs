//! High-speed exact disjunctive-graph evaluator operating directly on machine sequences (`D03-C`).

use crate::decoder::SerialDecoderWorkspace;
use crate::error::JsspError;
use crate::feasibility::validate_schedule;
use crate::model::{JobId, JsspInstance, Schedule};
use crate::neighborhoods::moves::{GraphMove, apply_move_to_slice, undo_move_on_slice};
use crate::representation::ObrChromosome;

pub(crate) const NONE_OP: u32 = u32::MAX;

/// One maximal contiguous same-machine block along the deterministic representative critical path.
///
/// Includes `len == 1` blocks so block indices `0..total_blocks` distinguish the true first
/// critical block `B_1` (`index == 0`) and last critical block `B_b` (`index + 1 == total_blocks`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalBlockSlice {
    pub block_index: usize,
    pub total_blocks: usize,
    pub machine: usize,
    pub operations: Vec<u32>,
}

/// Immutable analyzed snapshot of head/tail times, makespan, and critical blocks for a DAG state.
#[derive(Debug, Clone)]
pub struct EvaluatedGraphState {
    pub makespan: u64,
    pub heads: Vec<u64>,
    pub tails: Vec<u64>,
    pub critical_path: Vec<u32>,
    pub blocks: Vec<CriticalBlockSlice>,
}

/// Zero-allocation exact disjunctive-graph evaluator for local search and metaheuristics.
///
/// Evaluates complete DAG longest paths cleanly in `O(L)` flat array passes while detecting any
/// directed cycles immediately (`Ok(None)`), ensuring no cyclic graph ever enters search state.
#[derive(Debug, Clone)]
pub struct FastGraphEvaluator {
    num_jobs: usize,
    num_machines: usize,
    num_ops: usize,
    op_machine: Vec<usize>,
    proc_time: Vec<u64>,
    job_pred: Vec<u32>,
    job_succ: Vec<u32>,
    mean_proc_time: f64,
    // Mutable machine sequence state
    machine_seqs: Vec<Vec<u32>>,
    // Scratch buffers reused across topological DP evaluations
    mach_pred: Vec<u32>,
    mach_succ: Vec<u32>,
    mach_pos: Vec<usize>,
    in_deg: Vec<u8>,
    queue: Vec<u32>,
    topo: Vec<u32>,
    heads_buf: Vec<u64>,
    tails_buf: Vec<u64>,
}

impl FastGraphEvaluator {
    /// Initializes static instance topology arrays sized for `instance`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if instance operation lookups fail.
    pub fn new(instance: &JsspInstance) -> Result<Self, JsspError> {
        let num_jobs = instance.num_jobs();
        let num_machines = instance.num_machines();
        let num_ops = instance.num_operations();

        let mut op_machine = Vec::with_capacity(num_ops);
        let mut proc_time = Vec::with_capacity(num_ops);
        let mut job_pred = vec![NONE_OP; num_ops];
        let mut job_succ = vec![NONE_OP; num_ops];

        for j in 0..num_jobs {
            for k in 0..num_machines {
                let op = instance.operation_by_indices(j, k)?;
                let flat = j * num_machines + k;
                op_machine.push(op.machine.0);
                proc_time.push(op.processing_time);
                if k > 0 {
                    job_pred[flat] = (flat - 1) as u32;
                }
                if k + 1 < num_machines {
                    job_succ[flat] = (flat + 1) as u32;
                }
            }
        }

        let mean_proc_time = if num_ops == 0 {
            1.0
        } else {
            (instance.total_processing_time() as f64) / (num_ops as f64)
        };

        Ok(Self {
            num_jobs,
            num_machines,
            num_ops,
            op_machine,
            proc_time,
            job_pred,
            job_succ,
            mean_proc_time,
            machine_seqs: (0..num_machines)
                .map(|_| Vec::with_capacity(num_jobs))
                .collect(),
            mach_pred: vec![NONE_OP; num_ops],
            mach_succ: vec![NONE_OP; num_ops],
            mach_pos: vec![0; num_ops],
            in_deg: vec![0; num_ops],
            queue: Vec::with_capacity(num_ops),
            topo: Vec::with_capacity(num_ops),
            heads_buf: vec![0; num_ops],
            tails_buf: vec![0; num_ops],
        })
    }

    #[must_use]
    pub const fn num_jobs(&self) -> usize {
        self.num_jobs
    }

    #[must_use]
    pub const fn num_machines(&self) -> usize {
        self.num_machines
    }

    #[must_use]
    pub const fn num_ops(&self) -> usize {
        self.num_ops
    }

    #[must_use]
    pub const fn mean_proc_time(&self) -> f64 {
        self.mean_proc_time
    }

    #[inline]
    #[must_use]
    pub fn proc_time(&self, op: u32) -> u64 {
        self.proc_time[op as usize]
    }

    #[inline]
    #[must_use]
    pub fn job_pred(&self, op: u32) -> Option<u32> {
        let p = self.job_pred[op as usize];
        if p == NONE_OP { None } else { Some(p) }
    }

    #[inline]
    #[must_use]
    pub fn job_succ(&self, op: u32) -> Option<u32> {
        let s = self.job_succ[op as usize];
        if s == NONE_OP { None } else { Some(s) }
    }

    #[inline]
    #[must_use]
    pub fn mach_pos(&self, op: u32) -> usize {
        self.mach_pos[op as usize]
    }

    #[must_use]
    pub fn machine_seq(&self, machine: usize) -> &[u32] {
        &self.machine_seqs[machine]
    }

    /// Loads chronological machine sequences directly from a validated [`Schedule`].
    ///
    /// # Errors
    /// Returns [`JsspError`] if operation IDs fall outside instance bounds.
    pub fn load_from_schedule(&mut self, schedule: &Schedule) -> Result<(), JsspError> {
        for seq in &mut self.machine_seqs {
            seq.clear();
        }
        for m_seq in schedule.machine_sequences() {
            let m = m_seq.machine.0;
            for op in m_seq.operations {
                let flat = op.flat_index(self.num_jobs, self.num_machines)?;
                self.machine_seqs[m].push(flat as u32);
            }
        }
        Ok(())
    }

    fn sync_machine_slot(&mut self, machine: usize) {
        let seq = &self.machine_seqs[machine];
        let len = seq.len();
        for (pos, &op) in seq.iter().enumerate() {
            let u = op as usize;
            self.mach_pos[u] = pos;
            self.mach_pred[u] = if pos > 0 { seq[pos - 1] } else { NONE_OP };
            self.mach_succ[u] = if pos + 1 < len { seq[pos + 1] } else { NONE_OP };
        }
    }

    /// Replaces the internal machine sequence state from a saved snapshot (`clone`-style restore).
    pub fn restore_machine_seqs(&mut self, saved: &[Vec<u32>]) {
        for (m, (dst, src)) in self.machine_seqs.iter_mut().zip(saved.iter()).enumerate() {
            dst.clear();
            dst.extend_from_slice(src);
            let len = dst.len();
            for (pos, &op) in dst.iter().enumerate() {
                let u = op as usize;
                self.mach_pos[u] = pos;
                self.mach_pred[u] = if pos > 0 { dst[pos - 1] } else { NONE_OP };
                self.mach_succ[u] = if pos + 1 < len { dst[pos + 1] } else { NONE_OP };
            }
            let _ = m;
        }
    }

    /// Clones current machine sequences into a compact restart checkpoint.
    #[must_use]
    pub fn snapshot_machine_seqs(&self) -> Vec<Vec<u32>> {
        self.machine_seqs.clone()
    }

    /// Applies `mv` permanently to the current machine sequences.
    pub fn commit_move(&mut self, mv: &GraphMove) {
        apply_move_to_slice(&mut self.machine_seqs[mv.machine], mv);
        self.sync_machine_slot(mv.machine);
    }

    fn sync_machine_links_and_forward_dp(&mut self) -> Result<Option<u64>, JsspError> {
        for seq in &self.machine_seqs {
            let len = seq.len();
            for (pos, &op) in seq.iter().enumerate() {
                let u = op as usize;
                self.mach_pos[u] = pos;
                self.mach_pred[u] = if pos > 0 { seq[pos - 1] } else { NONE_OP };
                self.mach_succ[u] = if pos + 1 < len { seq[pos + 1] } else { NONE_OP };
            }
        }

        self.queue.clear();
        self.topo.clear();
        self.heads_buf.fill(0);

        for u in 0..self.num_ops {
            let deg =
                u8::from(self.job_pred[u] != NONE_OP) + u8::from(self.mach_pred[u] != NONE_OP);
            self.in_deg[u] = deg;
            if deg == 0 {
                self.queue.push(u as u32);
            }
        }

        let mut head_idx = 0usize;
        let mut makespan = 0u64;

        while head_idx < self.queue.len() {
            let u_op = self.queue[head_idx];
            head_idx += 1;
            self.topo.push(u_op);
            let u = u_op as usize;

            let finish = self.heads_buf[u]
                .checked_add(self.proc_time[u])
                .ok_or(JsspError::TimeOverflow)?;
            if finish > makespan {
                makespan = finish;
            }

            let js = self.job_succ[u];
            if js != NONE_OP {
                let v = js as usize;
                if finish > self.heads_buf[v] {
                    self.heads_buf[v] = finish;
                }
                self.in_deg[v] -= 1;
                if self.in_deg[v] == 0 {
                    self.queue.push(js);
                }
            }

            let ms = self.mach_succ[u];
            if ms != NONE_OP {
                let v = ms as usize;
                if finish > self.heads_buf[v] {
                    self.heads_buf[v] = finish;
                }
                self.in_deg[v] -= 1;
                if self.in_deg[v] == 0 {
                    self.queue.push(ms);
                }
            }
        }

        if self.topo.len() != self.num_ops {
            return Ok(None);
        }
        Ok(Some(makespan))
    }

    /// Probes candidate move `mv` via temporary in-place application + full exact forward DAG DP
    /// (verifying acyclicity and exact makespan), then restores original machine sequence.
    /// Returns `Ok(None)` iff `mv` induces a directed cycle.
    ///
    /// # Errors
    /// Returns [`JsspError::TimeOverflow`] if longest-path addition overflows `u64`.
    pub fn probe_move_makespan(&mut self, mv: &GraphMove) -> Result<Option<u64>, JsspError> {
        apply_move_to_slice(&mut self.machine_seqs[mv.machine], mv);
        let res = self.sync_machine_links_and_forward_dp();
        undo_move_on_slice(&mut self.machine_seqs[mv.machine], mv);
        self.sync_machine_slot(mv.machine);
        res
    }

    /// Runs full forward + backward head/tail DP, deterministic representative critical-path
    /// reconstruction (identical tie-breaking to M4), and maximal critical-block extraction.
    /// Returns `Ok(None)` if the current machine sequences form a cycle.
    ///
    /// # Errors
    /// Returns [`JsspError`] on arithmetic overflow or internal critical-path inconsistency.
    pub fn analyze_current(&mut self) -> Result<Option<EvaluatedGraphState>, JsspError> {
        let Some(makespan) = self.sync_machine_links_and_forward_dp()? else {
            return Ok(None);
        };

        self.tails_buf.fill(0);
        for &u_op in self.topo.iter().rev() {
            let u = u_op as usize;
            let cand = self.proc_time[u]
                .checked_add(self.tails_buf[u])
                .ok_or(JsspError::TimeOverflow)?;
            let jp = self.job_pred[u];
            if jp != NONE_OP && cand > self.tails_buf[jp as usize] {
                self.tails_buf[jp as usize] = cand;
            }
            let mp = self.mach_pred[u];
            if mp != NONE_OP && cand > self.tails_buf[mp as usize] {
                self.tails_buf[mp as usize] = cand;
            }
        }

        // Verify zero negative slack across every operation
        for u in 0..self.num_ops {
            let total = self.heads_buf[u]
                .checked_add(self.proc_time[u])
                .and_then(|x| x.checked_add(self.tails_buf[u]))
                .ok_or(JsspError::TimeOverflow)?;
            if total > makespan {
                return Err(JsspError::NegativeOperationSlack {
                    job: u / self.num_machines,
                    op_index: u % self.num_machines,
                    head_plus_proc_plus_tail: total,
                    graph_makespan: makespan,
                });
            }
        }

        // Reconstruct M4-identical representative critical path by minimum flat op tie-break.
        let mut current: Option<u32> = None;
        for u in 0..self.num_ops {
            if self.heads_buf[u] + self.proc_time[u] == makespan {
                current = Some(u as u32);
                break;
            }
        }
        let mut path_rev = Vec::new();
        while let Some(u_op) = current {
            path_rev.push(u_op);
            let u = u_op as usize;
            let target_head = self.heads_buf[u];
            let jp = self.job_pred[u];
            let mp = self.mach_pred[u];

            let mut best_pred: Option<u32> = None;
            if jp != NONE_OP
                && self.heads_buf[jp as usize] + self.proc_time[jp as usize] == target_head
            {
                best_pred = Some(jp);
            }
            if mp != NONE_OP
                && self.heads_buf[mp as usize] + self.proc_time[mp as usize] == target_head
            {
                best_pred = Some(match best_pred {
                    Some(existing) => existing.min(mp),
                    None => mp,
                });
            }
            current = if target_head == 0 && best_pred.is_none() {
                None
            } else {
                best_pred
            };
        }
        path_rev.reverse();

        // Partition representative critical path into ALL maximal same-machine runs (len >= 1).
        let mut raw_blocks: Vec<(usize, Vec<u32>)> = Vec::new();
        for &op in &path_rev {
            let m = self.op_machine[op as usize];
            if let Some((last_m, vec)) = raw_blocks.last_mut()
                && *last_m == m
            {
                vec.push(op);
                continue;
            }
            raw_blocks.push((m, vec![op]));
        }
        let total_blocks = raw_blocks.len();
        let blocks = raw_blocks
            .into_iter()
            .enumerate()
            .map(|(block_index, (machine, operations))| CriticalBlockSlice {
                block_index,
                total_blocks,
                machine,
                operations,
            })
            .collect();

        Ok(Some(EvaluatedGraphState {
            makespan,
            heads: self.heads_buf.clone(),
            tails: self.tails_buf.clone(),
            critical_path: path_rev,
            blocks,
        }))
    }

    /// Encodes an analyzed graph state into its canonical `D01-B` OBR chromosome ordered by
    /// `(head[u], head[u] + proc_time[u], job_id, op_index)`.
    #[must_use]
    pub fn encode_canonical_obr(&self, state: &EvaluatedGraphState) -> ObrChromosome {
        let mut keyed: Vec<(u64, u64, usize)> = (0..self.num_ops)
            .map(|u| {
                let h = state.heads[u];
                let c = h.saturating_add(self.proc_time[u]);
                (h, c, u)
            })
            .collect();
        keyed.sort_unstable();
        let genes = keyed
            .into_iter()
            .map(|(_, _, u)| JobId(u / self.num_machines))
            .collect();
        ObrChromosome::new(genes)
    }

    /// Projects `self` back to an M2-validated active [`Schedule`] via canonical `D01-B` OBR
    /// encoding, returning `(obr, validated_schedule, adapter_left_shift_count)`.
    ///
    /// # Errors
    /// Returns [`JsspError`] if analysis detects a cycle, decoding fails, or schedule validation fails.
    pub fn project_and_validate(
        &mut self,
        instance: &JsspInstance,
        decoder_ws: &mut SerialDecoderWorkspace,
    ) -> Result<(ObrChromosome, Schedule, usize), JsspError> {
        let state = self
            .analyze_current()?
            .ok_or_else(|| JsspError::InvalidDisjunctiveGraph {
                detail: "cannot project cyclic disjunctive graph state".to_owned(),
            })?;
        let obr = self.encode_canonical_obr(&state);
        let schedule = decoder_ws.decode(instance, &obr, true)?;
        validate_schedule(instance, &schedule)?;
        let mut left_shifts = 0usize;
        for sched_op in schedule.operations() {
            let flat = sched_op
                .op_id
                .flat_index(self.num_jobs, self.num_machines)?;
            if sched_op.start_time < state.heads[flat] {
                left_shifts += 1;
            }
        }
        Ok((obr, schedule, left_shifts))
    }
}
