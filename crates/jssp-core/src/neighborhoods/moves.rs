//! Neighborhood identifiers and canonical move representations (`D03-A`, `N5`–`N8`).

use crate::error::JsspError;

/// Canonical identifiers for the four Phase B disjunctive-graph neighborhoods (`D03-A`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NeighborhoodId {
    /// Nowicki & Smutnicki (1996) critical-block boundary adjacent swaps.
    N5,
    /// Balas & Vazacopoulos (1998) single-block shifts to/from block endpoints.
    N6,
    /// Zhang, Li, Guan & Rao (2007) extended critical-block endpoint-and-interior shifts.
    N7,
    /// Xie, Li, Gao & Gui (2023) `N7` intra-block + extra-block same-machine shifts.
    N8,
}

impl NeighborhoodId {
    /// Ordered array of all four approved neighborhoods.
    pub const ALL: [Self; 4] = [Self::N5, Self::N6, Self::N7, Self::N8];

    /// Parses `"N5"`, `"N6"`, `"N7"`, or `"N8"`.
    ///
    /// # Errors
    /// Returns [`JsspError::InvalidDisjunctiveGraph`] on unknown identifiers.
    pub fn from_code(code: &str) -> Result<Self, JsspError> {
        match code {
            "N5" => Ok(Self::N5),
            "N6" => Ok(Self::N6),
            "N7" => Ok(Self::N7),
            "N8" => Ok(Self::N8),
            other => Err(JsspError::InvalidDisjunctiveGraph {
                detail: format!("unknown neighborhood ID '{other}', expected N5..=N8"),
            }),
        }
    }

    /// Canonical string identifier (`"N5"`..`"N8"`).
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::N5 => "N5",
            Self::N6 => "N6",
            Self::N7 => "N7",
            Self::N8 => "N8",
        }
    }
}

/// Structural classification of a single-machine insertion/swap move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MoveKind {
    AdjacentSwap,
    InsertBefore,
    InsertAfter,
}

/// One candidate disjunctive-graph shift move on a single machine sequence.
///
/// Applying the move relocates operation at position `from_pos` on `machine`
/// to slot `to_pos` via `rotate_left` (`from_pos < to_pos`) or `rotate_right`
/// (`to_pos < from_pos`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphMove {
    pub machine: usize,
    pub from_pos: usize,
    pub to_pos: usize,
    pub moved_op: u32,
    pub anchor_op: u32,
    pub kind: MoveKind,
}

impl GraphMove {
    /// Canonical sequence-equivalence key eliminating duplicate adjacent swaps represented
    /// both as moving `a` after `b` and moving `b` before `a`.
    #[must_use]
    pub fn canonical_dedup_key(&self) -> (usize, usize, usize, u8) {
        if self.from_pos.abs_diff(self.to_pos) == 1 {
            let lo = self.from_pos.min(self.to_pos);
            (self.machine, lo, lo + 1, 0)
        } else {
            (self.machine, self.from_pos, self.to_pos, 1)
        }
    }

    /// Returns the directed machine-pair ordering `(first, second)` **created** by this move
    /// (`restored` order tested for tabu status) and its **broken** inverse pair to record
    /// into the tabu list upon acceptance.
    #[must_use]
    pub const fn tabu_orders(&self) -> ((u32, u32), (u32, u32)) {
        if self.from_pos < self.to_pos {
            // `moved_op` moves after `anchor_op`: creates (anchor -> moved), breaks (moved -> anchor)
            let created = (self.anchor_op, self.moved_op);
            let broken = (self.moved_op, self.anchor_op);
            (created, broken)
        } else {
            // `moved_op` moves before `anchor_op`: creates (moved -> anchor), breaks (anchor -> moved)
            let created = (self.moved_op, self.anchor_op);
            let broken = (self.anchor_op, self.moved_op);
            (created, broken)
        }
    }
}

/// Applies `mv` in place to `machine_seq`.
#[inline]
pub fn apply_move_to_slice<T>(machine_seq: &mut [T], mv: &GraphMove) {
    if mv.from_pos < mv.to_pos {
        machine_seq[mv.from_pos..=mv.to_pos].rotate_left(1);
    } else if mv.to_pos < mv.from_pos {
        machine_seq[mv.to_pos..=mv.from_pos].rotate_right(1);
    }
}

/// Restores `machine_seq` immediately after [`apply_move_to_slice`].
#[inline]
pub fn undo_move_on_slice<T>(machine_seq: &mut [T], mv: &GraphMove) {
    if mv.from_pos < mv.to_pos {
        machine_seq[mv.from_pos..=mv.to_pos].rotate_right(1);
    } else if mv.to_pos < mv.from_pos {
        machine_seq[mv.to_pos..=mv.from_pos].rotate_left(1);
    }
}
