use anyhow::{anyhow, Result};
use rustfst::fst_impls::VectorFst;
use rustfst::prelude::*;
use rustfst::semirings::Semiring;
use std::collections::HashMap;
use std::sync::Arc;

use crate::StdVectorFst;

/// Type for counts
type Count = LogWeight;
/// ID for an arc being counted
type TrId = StateId;
/// There is no arc here (FIXME: Option would be safer...)
pub static NO_TR_ID: TrId = NO_STATE_ID;
/// Keep track of transitions out of a given state
type PairTrMap = HashMap<(Label, StateId), TrId>;

/// Metadata for states
#[derive(Debug)]
struct CountState {
    /// ID of the backoff state for the current state.
    backoff_state: StateId,
    /// N-gram order of the state (of the outgoing arcs).
    order: u8,
    /// Count for n-gram corresponding to superfinal arc.
    final_count: Count,
    /// ID of the first outgoing arc at that state.
    first_tr: TrId,
}

/// Metadata for transitions
#[derive(Debug)]
struct CountTr {
    /// ID of the origin state for this arc.
    origin: StateId,
    /// ID of the destination state for this arc.
    destination: StateId,
    /// Label.
    label: Label,
    /// Count of the n-gram corresponding to this arc.
    count: Count,
    /// ID of backoff arc.
    backoff_tr: TrId,
}

/// Count N-Grams in a list of linear FSTs.
#[derive(Debug)]
pub struct NGramCounter {
    /// Maximum order of N-Grams to count
    pub order: u8,
    /// CountStates for each state
    states: Vec<CountState>,
    /// CountTrs for each transition
    trs: Vec<CountTr>,
    /// ID of start state
    initial: StateId,
    /// ID of unigram/backoff state
    backoff: StateId,
    /// Map (label, state_id) pairs to arc IDs.
    pair_tr_maps: Vec<PairTrMap>,
}

impl NGramCounter {
    pub fn new(order: u8) -> Self {
        let mut states = Vec::<_>::new();
        let backoff = states.len() as StateId;
        assert_eq!(backoff, 0); // Why would it be anything else?
        states.push(CountState {
            backoff_state: NO_STATE_ID,
            order: 1,
            final_count: Count::zero(),
            first_tr: NO_TR_ID,
        });
        let initial = if order == 1 {
            backoff
        } else {
            let unigram = states.len() as StateId;
            states.push(CountState {
                backoff_state: backoff,
                order: 2,
                final_count: Count::zero(),
                first_tr: NO_TR_ID,
            });
            unigram
        };
        let trs = Vec::<_>::new();
        let pair_tr_maps = vec![PairTrMap::new(); order as usize];
        Self {
            order,
            states,
            trs,
            initial,
            backoff,
            pair_tr_maps,
        }
    }

    /// Sum the counts of non-backoff (i.e. non-epsilon) arcs onto the backoff arc
    fn state_counts(&self, fst: &mut StdVectorFst) -> Result<()> {
        for (s, state) in self.states.iter().enumerate() {
            let s: StateId = s.try_into().unwrap();
            let mut state_count = state.final_count.clone();
            if state.backoff_state != NO_STATE_ID {
                let mut bo_pos: Option<usize> = None;
                let mut trs = fst.tr_iter_mut(s)?;
                for idx in 0..trs.len() {
                    if trs[idx].ilabel != EPS_LABEL {
                        state_count.plus_assign(Count::from(*trs[idx].weight.value()))?;
                    } else {
                        bo_pos = Some(idx);
                    }
                }
                match bo_pos {
                    None => return Err(anyhow!("backoff arc not found")),
                    Some(idx) => trs.set_weight(idx, TropicalWeight::from(*state_count.value()))?,
                }
            }
        }
        Ok(())
    }

    fn add_tr(&mut self, state_id: StateId, label: Label) -> TrId {
        // ID of the new arc we will create
        let tr_id: TrId = self.trs.len().try_into().unwrap();
        // Update the origin state's information
        let CountState {
            first_tr,
            backoff_state,
            order,
            ..
        } = self.states[state_id as usize];
        if first_tr == NO_TR_ID {
            self.states[state_id as usize].first_tr = tr_id;
        } else {
            let backoff_order = order as usize - 1;
            self.pair_tr_maps[backoff_order].insert((label, state_id), tr_id);
        }
        // Create with default values (if we are counting unigrams
        // nothing else needs to be done)
        self.trs.push(CountTr {
            origin: state_id,
            destination: self.initial,
            label: label,
            count: Count::zero(),
            backoff_tr: NO_TR_ID,
        });
        if self.order == 1 {
            return tr_id;
        }
        // Compute the backoff arc
        let backoff_tr = if backoff_state == NO_STATE_ID {
            NO_STATE_ID
        } else {
            self.find_tr(backoff_state, label)
        };
        // Compute the destinatoin state
        let destination = if order == self.order {
            self.trs[backoff_tr as usize].destination
        } else {
            let nextstate: StateId = self.states.len().try_into().unwrap();
            self.states.push(CountState {
                backoff_state: if backoff_tr == NO_TR_ID {
                    self.backoff
                } else {
                    self.trs[backoff_tr as usize].destination
                },
                order: order + 1,
                final_count: Count::zero(),
                first_tr: NO_TR_ID,
            });
            nextstate
        };
        // Update the arc we created above and return
        self.trs[tr_id as usize].destination = destination;
        self.trs[tr_id as usize].backoff_tr = backoff_tr;
        return tr_id;
    }

    fn find_tr(&mut self, state_id: StateId, label: Label) -> TrId {
        let count_state = &self.states[state_id as usize];
        if count_state.first_tr != NO_TR_ID {
            if self.trs[count_state.first_tr as usize].label == label {
                return count_state.first_tr;
            }
            let tr_map = &self.pair_tr_maps[count_state.order as usize - 1];
            if let Some(tr_id) = tr_map.get(&(label, state_id)) {
                return *tr_id;
            }
        }
        self.add_tr(state_id, label)
    }

    fn update_count(&mut self, state_id: StateId, label: Label, count: &Count) -> Result<StateId> {
        let mut tr_id = self.find_tr(state_id, label);
        let nextstate_id = self.trs[tr_id as usize].destination;
        while tr_id != NO_TR_ID {
            self.trs[tr_id as usize].count.plus_assign(count)?;
            tr_id = self.trs[tr_id as usize].backoff_tr;
        }
        Ok(nextstate_id)
    }

    fn update_final_count(&mut self, mut state_id: StateId, count: &Count) -> Result<()> {
        while state_id != NO_STATE_ID {
            self.states[state_id as usize]
                .final_count
                .plus_assign(count)?;
            state_id = self.states[state_id as usize].backoff_state;
        }
        Ok(())
    }

    fn get_fst(&self) -> Result<StdVectorFst> {
        let mut fst = StdVectorFst::new();
        for (s, state) in self.states.iter().enumerate() {
            let s: StateId = s.try_into().unwrap();
            fst.add_state();
            // rustfst and openfst handle final states differently
            if state.final_count != Count::zero() {
                fst.set_final(s, *state.final_count.value())?;
            }
            if state.backoff_state != NO_STATE_ID {
                fst.add_tr(
                    s,
                    Tr::new(
                        EPS_LABEL,
                        EPS_LABEL,
                        TropicalWeight::zero(),
                        state.backoff_state,
                    ),
                )?;
            }
        }
        for tr in self.trs.iter() {
            fst.add_tr(
                tr.origin,
                Tr::new(tr.label, tr.label, *tr.count.value(), tr.destination),
            )?;
        }
        fst.set_start(self.initial)?;
        self.state_counts(&mut fst)?;
        Ok(fst)
    }

    pub fn count_from_string_fst<W: Semiring>(&mut self, fst: &VectorFst<W>) -> Result<()> {
        let mut count_state = self.initial;
        // FIXME: Should assert that it's actually a string FST
        let mut fst_state = fst
            .start()
            .ok_or_else(|| anyhow!("FST has no start state"))?;
        let weight = Count::one();
        while !fst.is_final(fst_state)? {
            // !LOL?
            let trs = fst.get_trs(fst_state)?;
            let trs = trs.trs(); // WTF
            if trs.len() > 1 {
                return Err(anyhow!("More than one arc leaving state {}", fst_state));
            }
            if trs[0].ilabel != NO_LABEL {
                count_state = self.update_count(count_state, trs[0].ilabel, &weight)?;
            }
            fst_state = trs[0].nextstate;
        }
        self.update_final_count(count_state, &weight)?;
        Ok(())
    }

    pub fn get_ngram_counts<W: Semiring>(
        &mut self,
        sequences: Vec<VectorFst<W>>,
        syms: SymbolTable,
    ) -> Result<StdVectorFst> {
        for fst in sequences {
            self.count_from_string_fst(&fst)?;
        }
        let mut fst = self.get_fst()?;
        tr_sort(&mut fst, ILabelCompare {});
        if syms.len() > 0 {
            let syms = Arc::new(syms);
            fst.set_input_symbols(Arc::clone(&syms));
            fst.set_output_symbols(Arc::clone(&syms));
        }
        Ok(fst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_sequences;
    use rustfst::utils::decode_linear_fst;
    use std::iter::zip;
    use std::path::PathBuf;

    #[test]
    fn it_reads_sequences() {
        let (data, syms) = read_sequences(&PathBuf::from("testdata/austen.txt")).unwrap();
        assert_eq!(data.len(), 5);
        assert!(syms.contains_symbol("and"));
        assert!(syms.contains_symbol("dashwood"));
        assert!(syms.contains_symbol("amiable"));
        assert!(syms.contains_symbol("himself"));
        let path = decode_linear_fst(&data[1]).unwrap();
        let words: Vec<&str> = path
            .olabels
            .into_iter()
            .map(|label| syms.get_symbol(label))
            .flatten()
            .collect();
        assert_eq!("he was not an ill disposed young man", words.join(" "));
    }

    #[test]
    fn it_counts_ngrams() {
        let (data, _syms) = read_sequences(&PathBuf::from("testdata/austen.txt")).unwrap();
        let mut ngram = NGramCounter::new(3);
        ngram.count_from_string_fst(&data[0]).unwrap();
    }

    #[test]
    fn it_makes_an_fst() {
        let (data, syms) = read_sequences(&PathBuf::from("testdata/austen.txt")).unwrap();
        let mut ngram = NGramCounter::new(3);
        let fst = ngram.get_ngram_counts(data, syms).unwrap();
        //fst.write(&PathBuf::from("austen.fst")).unwrap();
        let ref_fst = StdVectorFst::read(&PathBuf::from("testdata/austen.count.3.fst")).unwrap();
        // Don't compare directly because it is impossible to read the resulting error message
        assert_eq!(fst.start(), ref_fst.start());
        let states: Vec<_> = fst.states_iter().collect();
        let ref_states: Vec<_> = ref_fst.states_iter().collect();
        assert_eq!(states, ref_states);
        for (state, ref_state) in zip(fst.fst_iter(), ref_fst.fst_iter()) {
            assert_eq!(state.state_id, ref_state.state_id);
            assert_eq!(state.trs, ref_state.trs);
            assert_eq!(state.num_trs, ref_state.num_trs);
            assert_eq!(state.final_weight, ref_state.final_weight);
        }
    }
}
