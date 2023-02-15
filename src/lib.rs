use anyhow::Result;
use rustfst::fst_impls::VectorFst;
use rustfst::semirings::TropicalWeight;
use rustfst::utils::acceptor;
use rustfst::Semiring;
use rustfst::{Label, SymbolTable};
use std::convert::identity;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::PathBuf;

/// rustfst does not define this but we will
type StdVectorFst = VectorFst<TropicalWeight>;

/// Functions for counting N-Grams from text
pub mod count;

/// N-Gram counter
pub use count::NGramCounter;

/// Read input sequences as whitespace-separated lines
pub fn read_sequences(input: &PathBuf) -> Result<(Vec<StdVectorFst>, SymbolTable)> {
    let fh = File::open(input)?;
    let mut syms = SymbolTable::new();
    let data: Vec<StdVectorFst> = BufReader::new(fh)
        .lines()
        .flat_map(identity) // let IntoIter remove None for us
        .map(|spam| {
            let labels: Vec<Label> = spam
                .trim()
                .split_whitespace()
                .map(|s| syms.add_symbol(s))
                .collect();
            // FIXME: We should assert that this is a string FST
            acceptor(&labels, TropicalWeight::one())
        })
        .collect();
    Ok((data, syms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfst::utils::decode_linear_fst;

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
}
