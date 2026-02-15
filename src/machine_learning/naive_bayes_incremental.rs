use super::MachineLearningModel;

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// A custom struct to store a class and its calculated score.
// We implement `PartialOrd` and `Ord` to allow `BinaryHeap` to order
// elements from highest score to lowest.
#[derive(Debug, PartialEq)]
struct ScoredClass {
  score: f64,
  class_id: u16,
}

impl Eq for ScoredClass {}

impl PartialOrd for ScoredClass {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    self.score.partial_cmp(&other.score)
  }
}

impl Ord for ScoredClass {
  fn cmp(&self, other: &Self) -> Ordering {
    self.score.partial_cmp(&other.score).unwrap_or_else(|| {
      // Handle NaN cases explicitly
      match (self.score.is_nan(), other.score.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => Ordering::Equal,
      }
    })
  }
}

// The NaiveBayes model struct. It holds the learned parameters after training.

#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
pub struct NaiveBayesIncremental {
  // Stores the log probability of each class occurring.
  // Example: ln(P(class="Sports"))
  class_log_priors: std::collections::HashMap<u16, f64>,

  // Stores the log probability of a word occurring given a class.
  // The outer HashMap is keyed by class ID. The inner Vec is indexed
  // by the word's position in the vocabulary.
  // Example: ln(P(word="ball" | class="Sports"))
  class_log_likelihoods: std::collections::HashMap<u16, Vec<f64>>,

  // Total number of documents in each class.
  doc_counts: Vec<u64>,

  // Total number of words in each class.
  word_counts: Vec<u64>,

  // The size of the vocabulary.
  vocabulary_size: usize,

  // Internal state for incremental training.
  class_word_frequencies: Vec<Vec<u64>>,
  total_docs_processed: u64,
  num_classes: usize,
}

// configurations
const ALPHA: f64 = 1.0; // Laplace Smoothing
const MAX_PREDICTIONS: usize = 5;

// We implement the MachineLearningModel trait for our NaiveBayes struct.
// The input is a vector of word counts (features), and the output is a class ID (u32).
impl MachineLearningModel<u16, u16> for NaiveBayesIncremental {
  /// Creates a new, uninitialized instance of the model.
  fn new() -> Self {
    NaiveBayesIncremental {
      class_log_priors: std::collections::HashMap::new(),
      class_log_likelihoods: std::collections::HashMap::new(),
      doc_counts: Vec::new(),
      word_counts: Vec::new(),
      vocabulary_size: 0,
      class_word_frequencies: Vec::new(),
      total_docs_processed: 0,
      num_classes: 0,
    }
  }

  /// The `fit` method for training the model on text data.
  /// It now correctly accepts a slice of `u32` labels.
  /// The `train` method for training the model on text data.
  /// This is an incremental fit, so you can call it multiple times.
  fn train(&mut self, data: &[Vec<u16>], labels: &[Vec<u16>]) -> Result<(), String> {
    if data.len() != labels.len() {
      return Err("Data and labels must have same length".to_string());
    }
    if data.is_empty() {
      return Err("Cannot train on empty data".to_string());
    }

    // Initialize vocabulary size if it hasn't been set.
    if self.vocabulary_size == 0 {
      self.vocabulary_size = data[0].len();
    }

    // Find all unique classes from the new labels and update `num_classes`.
    let mut all_new_classes = std::collections::HashSet::new();
    for doc_labels in labels.iter() {
      for &label in doc_labels {
        all_new_classes.insert(label);
      }
    }
    let max_new_label = all_new_classes.iter().max().unwrap_or(&0) + 1;
    if max_new_label as usize > self.num_classes {
      self.num_classes = max_new_label as usize;
      self.doc_counts.resize(self.num_classes, 0);
      self.word_counts.resize(self.num_classes, 0);
      self
        .class_word_frequencies
        .resize(self.num_classes, vec![0; self.vocabulary_size]);
    }

    // Accumulate word and document occurrences for each class.
    self.total_docs_processed += data.len() as u64;
    for (i, doc_counts_row) in data.iter().enumerate() {
      let doc_labels = &labels[i];

      // For multi-label, we iterate over all labels for the current document.
      for &label in doc_labels {
        let class_index = label as usize;

        // Increment document count for the class.
        self.doc_counts[class_index] += 1;

        // Increment word counts for the class.
        for (word_index, &count) in doc_counts_row.iter().enumerate() {
          self.class_word_frequencies[class_index][word_index] += count as u64;
          self.word_counts[class_index] += count as u64;
        }
      }
    }

    // Finalize training by calculating the log probabilities after all data is trained
    self.finalize();

    Ok(())
  }

  /// The `predict` method for making a prediction on a single document.
  /// It now returns a vector of class IDs that meet a certain accuracy threshold.
  // fn predict(&self, data_point: &Vec<u16>) -> Vec<u16> {
  //     let mut best_class = 0_u16;
  //     let mut max_posterior = f64::NEG_INFINITY;

  //     // Iterate through each class to calculate the posterior probability
  //     for (&class, &log_prior) in &self.class_log_priors {
  //         let mut log_posterior = log_prior;

  //         // Calculate the log likelihood for the document's words
  //         if let Some(log_likelihoods) = self.class_log_likelihoods.get(&class) {
  //             for (i, &word_count) in data_point.iter().enumerate() {
  //                 // We add the log probability for each occurrence of the word.
  //                 log_posterior += word_count as f64 * log_likelihoods[i];
  //             }
  //         }

  //         // Check if this class has the highest posterior probability so far.
  //         if log_posterior > max_posterior {
  //             max_posterior = log_posterior;
  //             best_class = class;
  //         }
  //     }

  //     best_class
  // }

  // This new function returns multiple categories based on a relative accuracy threshold.
  fn predict(&self, data_point: &Vec<u16>) -> Vec<u16> {
    // Use a BinaryHeap to efficiently find the top-scoring classes.
    // It's a min-heap, so we will store tuples to sort by score.
    let mut top_classes = BinaryHeap::new();

    // Calculate the log posterior for every class.
    for (&class, &log_prior) in &self.class_log_priors {
      let mut log_posterior = log_prior;

      if let Some(log_likelihoods) = self.class_log_likelihoods.get(&class) {
        for (i, &word_count) in data_point.iter().enumerate() {
          log_posterior += word_count as f64 * log_likelihoods[i];
        }
      }

      // Push the current class's score into the heap.
      top_classes.push(ScoredClass {
        score: log_posterior,
        class_id: class,
      });
    }

    // The heap is a max-heap, so we can pop the top elements.
    let mut results: Vec<u16> = Vec::new();
    // Take the top 5 results.
    for _ in 0..MAX_PREDICTIONS {
      if let Some(scored_class) = top_classes.pop() {
        results.push(scored_class.class_id);
      } else {
        break;
      }
    }

    results
  }
}

impl NaiveBayesIncremental {
  // This public method finalizes the training by calculating the log probabilities
  // after all data has been incrementally passed to the `fit` method.
  // this doesn't remove the training data, so you can easily train it more later, and make it even stronger.
  fn finalize(&mut self) {
    let alpha = ALPHA; // Laplace smoothing alpha value
    for class in 0..self.num_classes {
      // Check if this class has data
      if self.doc_counts[class] == 0 {
        continue;
      }

      // Calculate log prior: ln(P(class))
      let doc_count = self.doc_counts[class];
      let log_prior = (doc_count as f64 / self.total_docs_processed as f64).ln();
      self.class_log_priors.insert(class as u16, log_prior);

      // Calculate word likelihoods: P(word | class)
      let total_words_in_class = self.word_counts[class] as f64;
      let mut log_likelihoods_vec = vec![0.0; self.vocabulary_size];

      let class_word_freqs = &self.class_word_frequencies[class];

      for i in 0..self.vocabulary_size {
        // Apply Laplace smoothing to prevent zero probabilities.
        let word_count = class_word_freqs[i] as f64;
        let numerator = word_count + alpha;
        let denominator = total_words_in_class + alpha * (self.vocabulary_size as f64);

        log_likelihoods_vec[i] = (numerator / denominator).ln();
      }
      self
        .class_log_likelihoods
        .insert(class as u16, log_likelihoods_vec);
    }
  }
}
