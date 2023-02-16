/// Make an N-Gram model with Kneser-Ney smoothing from a count FST.
#[derive(Debug)]
pub struct NGramMaker {
    /// Absolute Discounting D
    parameter: f32,
    /// number of bins for discounting
    bins: u8,
    /// count bins for orders
    count_of_counts: NGramCountOfCounts,
    /// discount for bins
    discount: Vec<Vec<f32>>,
}

impl NGramMaker {
    /// Generalized rule of thumb: Y = k n_k / ( k n_k + (k+1) * n_{k+1} )
    /// where n_k is the total count mass for items that occurred k times
    /// Note: method generalized to allow for zeros in low count bins:
    ///       find lowest non-empty count bins, then use rule of thumb
    fn abs_discount_rule_of_thumb(&self, order: u8) -> f32 {
    }

    /// Calculate absolute discounting parameter according to histogram formula.
    /// Using Chen and Goodman version from equation (26) of paper
    /// For count i, discount: i - ( (i+1) Y n_{i+1} / n_{i} ) for a given Y
    fn absolute_discount_formula(&self, order: u8, bin: u8, y: f32) -> f32 {
        0.0
    }

    /// Calculate absolute discount parameter for count i
    /// Note: discounts stored with bin indices starting at 0, bin k is count k+1
    fn calculate_absolute_discount(&self, order: u8, bin: u8) {
    }

    /// Return negative log discounted count for provided negative log count
    fn calculate_discounts(&self) {
    }

    fn get_discount(&self, neglogcount: Count, order: u8) -> f32 {
        0.0
    }

    fn update_kn_counts(&self, st: StateId, increment: bool) -> Result<()> {
        Ok(())
    }

    /// Calculate update value, either for incrementing or removing hi order
    fn calc_kn_value(&self, increment: bool, hi_order_value: f32,
                     lo_order_value: f32) -> f32 {
        0.0
    }

    /// Update the backoff arc with total count
    fn update_total_count(&self, st: StateId) -> Result<()> {
        Ok(())
    }

    /// Modify lower order counts according to Kneser Ney formula
    fn assign_kn_counts(&self) -> Result<()> {
        Ok(())
    }

    /// Normalizes n-gram counts and smoothes to create an n-gram model.
    fn make_ngram_model(&self) -> Result<()> {
        Ok(())
    }
}
