use rustfst::prelude::*;
use crate::StdVectorFst;

// Default normalization constant (e.g., for checks)
const KNORMEPS: f64 = 0.001;
const KFLOATEPS: f64 = 0.000001;
const KINFBACKOFF: f64 = 99.00;

pub struct NGramModel {
    /// Underlying FST
    fst: StdVectorFst,
    /// Highest order in the modle
    hi_order: u8,
    /// Order of each state
    state_orders: Vec<u8>,
    /// N-Gram which reaches a given state
    state_ngrams: Vec<Vec<Label>>,
}

impl NGramModel {
    pub fn new(fst: StdVectorFst) -> Self {
        let hi_order: u8 = 0;
        let state_orders = Vec::<u8>::new();
        let state_ngrams = Vec::<Vec<Label>>::new();
        Self {
            fst,
            hi_order,
            state_orders,
            state_ngrams,
        }
    }
}
