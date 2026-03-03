use rkyv::{Archive, Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::SparseVector;



/// Multinomial Naive Bayes that supports an arbitrary number of classes.
/// Labels are `u8` category IDs (matching the rest of the pipeline).
/// `predict` returns a `Vec<u8>` of all classes whose posterior score
/// exceeds a relative threshold (multi-label output).
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct MultinomialNB {
    // How many documents belong to each class?
    class_doc_counts: HashMap<u8, usize>,

    // Map<Class, Map<WordID, TotalCount>>
    feature_counts: HashMap<u8, HashMap<u16, u64>>,

    // Total count of ALL words in a specific class (denominator)
    class_total_word_counts: HashMap<u8, u64>,

    // Vocabulary: all unique word-ids seen during training
    vocab_size: usize,

    // Total number of documents processed
    total_docs: usize,

    // Laplace smoothing parameter (usually 1.0)
    alpha: f64,

    // Relative threshold: a class is included in the output when its
    // log-posterior is within `threshold` nats of the best class.
    // 0.0 → only the single best class; higher → more permissive multi-label.
    pub threshold: f64,
}

impl MultinomialNB {
    pub fn new(alpha: f64) -> Self {
        Self {
            class_doc_counts: HashMap::new(),
            feature_counts: HashMap::new(),
            class_total_word_counts: HashMap::new(),
            vocab_size: 0,
            total_docs: 0,
            alpha,
            threshold: 2.0, // default: return classes within 2 nats of best
        }
    }

    /// Trains the model.
    /// `docs`   – sparse BoW vectors (word_id → frequency)
    /// `labels` – multi-label: each entry is a `Vec<u8>` of category IDs
    pub fn train(&mut self, docs: &[SparseVector], labels: &[Vec<u8>]) {
        assert_eq!(docs.len(), labels.len(), "Docs and Labels length mismatch");

        self.total_docs += docs.len();

        // We need to track vocab to compute vocab_size
        let mut vocab: HashSet<u16> = HashSet::new();

        for (doc, label_vec) in docs.iter().zip(labels.iter()) {
            for &label in label_vec {
                *self.class_doc_counts.entry(label).or_insert(0) += 1;

                let class_features = self
                    .feature_counts
                    .entry(label)
                    .or_insert_with(HashMap::new);

                let class_word_total =
                    self.class_total_word_counts.entry(label).or_insert(0);

                for (&word_id, &count) in doc {
                    vocab.insert(word_id);
                    *class_features.entry(word_id).or_insert(0) += count as u64;
                    *class_word_total += count as u64;
                }
            }
        }

        self.vocab_size = vocab.len().max(self.vocab_size);
    }

    /// Returns the log-posterior score for every known class.
    pub fn score_all(&self, doc: &SparseVector) -> Vec<(u8, f64)> {
        let vocab_size = self.vocab_size as f64;
        let mut scores: Vec<(u8, f64)> = Vec::with_capacity(self.class_doc_counts.len());

        for (&class_id, &class_n_docs) in &self.class_doc_counts {
            let log_prior =
                (class_n_docs as f64).ln() - (self.total_docs as f64).ln();

            let total_words =
                *self.class_total_word_counts.get(&class_id).unwrap_or(&0);
            let denominator =
                (total_words as f64 + self.alpha * vocab_size).ln();

            let mut log_likelihood = 0.0;

            match self.feature_counts.get(&class_id) {
                Some(class_features) => {
                    for (&word_id, &word_count_in_doc) in doc {
                        let count_in_class =
                            *class_features.get(&word_id).unwrap_or(&0);
                        let numerator =
                            (count_in_class as f64 + self.alpha).ln();
                        log_likelihood +=
                            (word_count_in_doc as f64) * (numerator - denominator);
                    }
                }
                None => {
                    for (_, &word_count_in_doc) in doc {
                        let numerator = self.alpha.ln();
                        log_likelihood +=
                            (word_count_in_doc as f64) * (numerator - denominator);
                    }
                }
            }

            scores.push((class_id, log_prior + log_likelihood));
        }

        scores
    }

    /// Returns the single best class (highest posterior).
    pub fn predict_top1(&self, doc: &SparseVector) -> Option<u8> {
        self.score_all(doc)
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(class, _)| class)
    }

    /// Returns all classes whose log-posterior is within `self.threshold`
    /// nats of the best class — multi-label output.
    pub fn predict(&self, doc: &SparseVector) -> Vec<u8> {
        let scores = self.score_all(doc);
        if scores.is_empty() {
            return vec![];
        }

        let best = scores
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut result: Vec<u8> = scores
            .into_iter()
            .filter(|(_, s)| best - *s <= self.threshold)
            .map(|(c, _)| c)
            .collect();

        result.sort_unstable();
        result
    }
}
