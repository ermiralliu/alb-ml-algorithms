// use std::collections::HashMap;
//
// use rustc_hash::FxBuildHasher;
//
// use super::knn_hashmap::SparseVector;
//
// pub struct MultiLabelNB {
//     // Label -> Model for that specific label
//     // We treat each label as a independent Binary Naive Bayes classifier
//     models: HashMap<u16, BinaryNB, FxBuildHasher>,
//     total_docs: usize,
//     alpha: f64,
// }
//
// // Internal helper for a single label (Is it present or not?)
// struct BinaryNB {
//     docs_with_label: usize,
//     // WordID -> Count of times seen in docs WITH this label
//     feature_counts: HashMap<u16, u64, FxBuildHasher>,
//     total_word_count: u64,
// }
//
// impl MultiLabelNB {
//     pub fn new(alpha: f64) -> Self {
//         MultiLabelNB {
//             models: HashMap::default(),
//             total_docs: 0,
//             alpha
//         }
//     }
//     pub fn train(&mut self, docs: &[SparseVector], labels: &[Vec<u16>]) {
//         self.total_docs += docs.len();
//
//         for (doc, doc_labels) in docs.iter().zip(labels.iter()) {
//             for &label in doc_labels {
//                 let model = self.models.entry(label).or_insert_with(|| BinaryNB {
//                     docs_with_label: 0,
//                     feature_counts: HashMap::with_hasher(FxBuildHasher::default()),
//                     total_word_count: 0,
//                 });
//
//                 model.docs_with_label += 1;
//                 for (&word_id, &count) in doc {
//                     *model.feature_counts.entry(word_id).or_insert(0) += count as u64;
//                     model.total_word_count += count as u64;
//                 }
//             }
//         }
//     }
//
//     // pub fn predict(&self, doc: &SparseVector, threshold: f64) -> Vec<u16> {
//     //     let mut predicted_labels = Vec::new();
//     //
//     //     for (&label_id, model) in &self.models {
//     //         // Calculate probability that label exists vs doesn't exist
//     //         // For simplicity in multi-label, we often just check if
//     //         // the score passes a threshold or is higher than the "Not-Label" score.
//     //         if self.score_label(doc, model) > threshold {
//     //             predicted_labels.push(label_id);
//     //         }
//     //     }
//     //     predicted_labels
//     // }
//     //
//     // fn score_label(&self, doc: &SparseVector, model: &BinaryNB) -> f64 {
//     //     // Log-likelihood calculation similar to previous implementation
//     //     // but specific to this label's model
//     //     // ... (Logic follows the Multinomial math from before)
//     //     0.0 // Placeholder
//     // }
//     fn score_label(&self, doc: &SparseVector, model: &BinaryNB) -> f64 {
//         let vocab_size = self.vocab.len() as f64;
//
//         // 1. Calculate P(Label) and P(Not Label)
//         // Prior probability of the label existing
//         let prior_pos = (model.docs_with_label as f64 + self.alpha) /
//                         (self.total_docs as f64 + 2.0 * self.alpha);
//
//         // Prior probability of the label NOT existing
//         let prior_neg = 1.0 - prior_pos;
//
//         let mut log_prob_pos = prior_pos.ln();
//         let mut log_prob_neg = prior_neg.ln();
//
//         // 2. Calculate Likelihood: P(word | Label) vs P(word | Not Label)
//         // Note: For a true Binary NB, we need the counts for when the label is MISSING.
//         // To save memory, we calculate "Not Label" counts as (Global Count - Label Count).
//
//         let denom_pos = (model.total_word_count as f64 + self.alpha * vocab_size).ln();
//
//         // Total words across all docs minus words in this specific label
//         let total_words_not_in_label = self.total_word_count - model.total_word_count;
//         let denom_neg = (total_words_not_in_label as f64 + self.alpha * vocab_size).ln();
//
//         for (&word_id, &word_count_in_doc) in doc {
//             let count_pos = *model.feature_counts.get(&word_id).unwrap_or(&0) as f64;
//
//             // Get global count for this word from a pre-calculated map
//             let global_word_count = *self.feature_counts.get(&word_id).unwrap_or(&0) as f64;
//             let count_neg = global_word_count - count_pos;
//
//             // P(word | Label)
//             log_prob_pos += (word_count_in_doc as f64) * ((count_pos + self.alpha).ln() - denom_pos);
//
//             // P(word | Not Label)
//             log_prob_neg += (word_count_in_doc as f64) * ((count_neg + self.alpha).ln() - denom_neg);
//         }
//
//         // Return the difference.
//         // Positive = "Label applies", Negative = "Label does not apply"
//         log_prob_pos - log_prob_neg
//     }
//
//     pub fn predict(&self, doc: &SparseVector) -> Vec<u16> {
//         let mut results = Vec::new();
//         for (&label_id, model) in &self.models {
//             // Threshold of 0.0 means P(Label) > P(Not Label)
//             if self.score_label(doc, model) > 0.0 {
//                 results.push(label_id);
//             }
//         }
//         results
//     }
// }
use std::collections::HashMap;

use rustc_hash::FxBuildHasher;

use super::knn_hashmap::SparseVector;

pub struct MultiLabelNB {
  // Label -> Model for that specific label
  // We treat each label as a independent Binary Naive Bayes classifier
  models: HashMap<u16, BinaryNB, FxBuildHasher>,
  total_docs: usize,
  alpha: f64,
  // Global vocabulary - all unique word IDs seen across all documents
  vocab: HashMap<u16, (), FxBuildHasher>,
  // Global word counts across all documents (for calculating "Not Label" counts)
  feature_counts: HashMap<u16, u64, FxBuildHasher>,
  total_word_count: u64,
}

// Internal helper for a single label (Is it present or not?)
struct BinaryNB {
  docs_with_label: usize,
  // WordID -> Count of times seen in docs WITH this label
  feature_counts: HashMap<u16, u64, FxBuildHasher>,
  total_word_count: u64,
}

impl MultiLabelNB {
  pub fn new(alpha: f64) -> Self {
    MultiLabelNB {
      models: HashMap::default(),
      total_docs: 0,
      alpha,
      vocab: HashMap::default(),
      feature_counts: HashMap::default(),
      total_word_count: 0,
    }
  }

  pub fn train(&mut self, docs: &[SparseVector], labels: &[Vec<u16>]) {
    self.total_docs += docs.len();

    // First pass: build global vocabulary and feature counts
    for doc in docs.iter() {
      for (&word_id, &count) in doc {
        self.vocab.entry(word_id).or_insert(());
        *self.feature_counts.entry(word_id).or_insert(0) += count as u64;
        self.total_word_count += count as u64;
      }
    }

    // Second pass: train individual binary classifiers for each label
    for (doc, doc_labels) in docs.iter().zip(labels.iter()) {
      for &label in doc_labels {
        let model = self.models.entry(label).or_insert_with(|| BinaryNB {
          docs_with_label: 0,
          feature_counts: HashMap::with_hasher(FxBuildHasher::default()),
          total_word_count: 0,
        });

        model.docs_with_label += 1;
        for (&word_id, &count) in doc {
          *model.feature_counts.entry(word_id).or_insert(0) += count as u64;
          model.total_word_count += count as u64;
        }
      }
    }
  }

  fn score_label(&self, doc: &SparseVector, model: &BinaryNB) -> f64 {
    let vocab_size = self.vocab.len() as f64;

    // 1. Calculate P(Label) and P(Not Label)
    let prior_pos =
      (model.docs_with_label as f64 + self.alpha) / (self.total_docs as f64 + 2.0 * self.alpha);

    let prior_neg = 1.0 - prior_pos;

    let mut log_prob_pos = prior_pos.ln();
    let mut log_prob_neg = prior_neg.ln();

    // 2. Calculate Likelihood
    let denom_pos = (model.total_word_count as f64 + self.alpha * vocab_size).ln();

    let total_words_not_in_label = self.total_word_count - model.total_word_count;
    let denom_neg = (total_words_not_in_label as f64 + self.alpha * vocab_size).ln();

    for (&word_id, &word_count_in_doc) in doc {
      let count_pos = *model.feature_counts.get(&word_id).unwrap_or(&0) as f64;
      let global_word_count = *self.feature_counts.get(&word_id).unwrap_or(&0) as f64;
      let count_neg = global_word_count - count_pos;

      log_prob_pos += (word_count_in_doc as f64) * ((count_pos + self.alpha).ln() - denom_pos);
      log_prob_neg += (word_count_in_doc as f64) * ((count_neg + self.alpha).ln() - denom_neg);
    }

    log_prob_pos - log_prob_neg
  }

  pub fn predict(&self, doc: &SparseVector) -> Vec<u16> {
    let mut scores: Vec<(u16, f64)> = self
      .models
      .iter()
      .map(|(&label_id, model)| (label_id, self.score_label(doc, model)))
      .collect();

    // Sort by score descending
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Strategy 1: Use a higher threshold
    // scores.into_iter()
    //     .filter(|(_, score)| *score > 1.0)  // Increase from 0.0 to require stronger evidence
    //     .map(|(label, _)| label)
    //     .collect()

    // Strategy 2: Take top-k labels only
    // scores.into_iter()
    //     .take(5)  // Only return top 5 labels
    //     .filter(|(_, score)| *score > 0.0)
    //     .map(|(label, _)| label)
    //     .collect()

    // Strategy 3: Adaptive threshold based on score distribution
    if scores.is_empty() {
      return Vec::new();
    }

    let max_score = scores[0].1;
    let threshold = max_score - 3.0; // Only include labels within 2.0 log-prob of best

    scores
      .into_iter()
      .filter(|(_, score)| *score > 0.0 && *score > threshold)
      .map(|(label, _)| label)
      .collect()
  }
}
