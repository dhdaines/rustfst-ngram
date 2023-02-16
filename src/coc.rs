use anyhow::{Result, anyhow};

/// Compute count-of-count bins for e.g. absolute discounting (used
/// by Kneser-Ney smoothing)
#[derive(Debug)]
pub struct NGramCountOfCounts {
}

impl NGramCountOfCounts {
    pub const MAX_BINS: u8 = 32;

    pub fn new(bins: u8) -> Result<Self> {
        if bins > Self::MAX_BINS {
            Err(anyhow!("NGramCountOfCounts: Number of bins too large: {}", bins))
        }
        else {
            let coc = NGramCountOfCounts {
            };
            Ok(coc)
        }
    }
}

