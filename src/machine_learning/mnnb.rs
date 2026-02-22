use rustc_hash::FxBuildHasher;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;

// Your defined type
pub type SparseVector = HashMap<u16, u16, FxBuildHasher>;

pub struct MultinomialNB {
    // How many documents belong to each class?
    class_doc_counts: HashMap<u16, usize, FxBuildHasher>,
    
    // Map<Class, Map<WordID, TotalCount>>
    // Stores how many times a specific word appears in a specific class
    feature_counts: HashMap<u16, HashMap<u16, u64, FxBuildHasher>, FxBuildHasher>,
    
    // Total count of ALL words in a specific class (for the denominator)
    class_total_word_counts: HashMap<u16, u64, FxBuildHasher>,
    
    // The set of all unique words seen across all data (Vocabulary size)
    vocab: HashSet<u16, FxBuildHasher>,
    
    // Total number of documents processed
    total_docs: usize,
    
    // Smoothing parameter (usually 1.0)
    alpha: f64,
}

impl MultinomialNB {
    pub fn new(alpha: f64) -> Self {
        Self {
            class_doc_counts: HashMap::with_hasher(FxBuildHasher::default()),
            feature_counts: HashMap::with_hasher(FxBuildHasher::default()),
            class_total_word_counts: HashMap::with_hasher(FxBuildHasher::default()),
            vocab: HashSet::with_hasher(FxBuildHasher::default()),
            total_docs: 0,
            alpha,
        }
    }

    /// Trains the model with the provided vectors and labels.
    pub fn train(&mut self, docs: &[SparseVector], labels: &[u16]) {
        assert_eq!(docs.len(), labels.len(), "Docs and Labels length mismatch");

        self.total_docs += docs.len();

        for (doc, &label) in docs.iter().zip(labels.iter()) {
            // 1. Increment document count for this class (Prior)
            *self.class_doc_counts.entry(label).or_insert(0) += 1;

            // Ensure the class exists in our feature maps
            let class_features = self
                .feature_counts
                .entry(label)
                .or_insert_with(|| HashMap::with_hasher(FxBuildHasher::default()));
            
            let class_word_total = self.class_total_word_counts.entry(label).or_insert(0);

            // 2. Aggregate word counts (Likelihood)
            for (&word_id, &count) in doc {
                self.vocab.insert(word_id);
                
                // Add to specific word count for this class
                *class_features.entry(word_id).or_insert(0) += count as u64;
                
                // Add to total word count for this class
                *class_word_total += count as u64;
            }
        }
    }

    /// Predicts the class for a single document.
    pub fn predict(&self, doc: &SparseVector) -> Option<u16> {
        let vocab_size = self.vocab.len() as f64;
        let mut best_class = None;
        let mut max_score = f64::NEG_INFINITY;

        // Iterate over all known classes to find the one with the highest probability
        for (&class_id, &class_n_docs) in &self.class_doc_counts {
            // --- Calculate Prior Log Probability ---
            // P(Class) = count(Class) / total_docs
            let log_prior = (class_n_docs as f64).ln() - (self.total_docs as f64).ln();

            // --- Calculate Likelihood Log Probability ---
            // We only need to sum log probs for words present in the input doc.
            // P(w|c) = (count(w,c) + alpha) / (count(c) + alpha * |V|)
            
            let total_words_in_class = *self.class_total_word_counts.get(&class_id).unwrap_or(&0);
            let denominator = (total_words_in_class as f64 + self.alpha * vocab_size).ln();
            
            let mut sum_log_likelihood = 0.0;

            if let Some(class_features) = self.feature_counts.get(&class_id) {
                for (&word_id, &word_count_in_doc) in doc {
                    // Check how many times this word appeared in this class during training
                    let count_in_class = *class_features.get(&word_id).unwrap_or(&0);
                    
                    let numerator = (count_in_class as f64 + self.alpha).ln();
                    
                    // Add to score: (frequency of word in doc) * log(P(word|class))
                    sum_log_likelihood += (word_count_in_doc as f64) * (numerator - denominator);
                }
            } else {
                // If class has no features (edge case), likelihood relies purely on smoothing
                 for (_, &word_count_in_doc) in doc {
                    let numerator = self.alpha.ln();
                    sum_log_likelihood += (word_count_in_doc as f64) * (numerator - denominator);
                 }
            }

            let final_score = log_prior + sum_log_likelihood;

            if final_score > max_score {
                max_score = final_score;
                best_class = Some(class_id);
            }
        }

        best_class
    }
}
