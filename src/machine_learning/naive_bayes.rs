use super::MachineLearningModel;

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;

use std::io::{Write, Read};
// A custom struct to store a class and its calculated score.
// We implement `PartialOrd` and `Ord` to allow `BinaryHeap` to order
// elements from highest score to lowest.
#[derive(Debug, PartialEq)]
struct ScoredClass {
    score: f64,
    class_id: u32,
}

impl Eq for ScoredClass {}

impl PartialOrd for ScoredClass {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for ScoredClass {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score)
            .unwrap_or_else(|| {
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
pub struct NaiveBayes {
    // Stores the log probability of each class occurring.
    // Example: ln(P(class="Sports"))
    class_log_priors: std::collections::HashMap<u32, f64>,

    // Stores the log probability of a word occurring given a class.
    // The outer HashMap is keyed by class ID. The inner Vec is indexed
    // by the word's position in the vocabulary.
    // Example: ln(P(word="ball" | class="Sports"))
    class_log_likelihoods: std::collections::HashMap<u32, Vec<f64>>,

    // Total number of documents in each class.
    doc_counts: std::collections::HashMap<u32, u64>,

    // Total number of words in each class.
    word_counts: std::collections::HashMap<u32, u64>,

    // The size of the vocabulary.
    vocabulary_size: usize,

    // Internal state for incremental training.
    class_word_frequencies: std::collections::HashMap<u32, Vec<u64>>,
    total_docs_processed: u64,
}

// configurations
const ALPHA: f64 = 1.0; // Laplace Smoothing
const MAX_PREDICTIONS: usize = 5;

// We implement the MachineLearningModel trait for our NaiveBayes struct.
// The input is a vector of word counts (features), and the output is a class ID (u32).
impl MachineLearningModel<Vec<u16>, u32> for NaiveBayes {
    /// Creates a new, uninitialized instance of the model.
    fn new() -> Self {
        NaiveBayes {
            class_log_priors: std::collections::HashMap::new(),
            class_log_likelihoods: std::collections::HashMap::new(),
            doc_counts: std::collections::HashMap::new(),
            word_counts: std::collections::HashMap::new(),
            vocabulary_size: 0,
            class_word_frequencies: std::collections::HashMap::new(),
            total_docs_processed: 0,
        }
    }

    /// The `fit` method for training the model on text data.
    /// It now correctly accepts a slice of `u32` labels.
    fn fit(&mut self, data: &[Vec<u16>], labels: &[u32]) -> Result< (), String> {
        if data.len() != labels.len() {
            return Err("Data and labels must have same length".to_string());
        }
        if data.is_empty() {
            return Err("Cannot train on empty data".to_string());
        }

        // Find the number of features (vocabulary size).
        if let Some(first_doc) = data.get(0) {
            self.vocabulary_size = first_doc.len();
        } else {
            // Handle empty data case.
            return Ok(());
        }

        // 1. Count word and document occurrences for each class.
        self.total_docs_processed += data.len() as u64;

        for (i, doc_counts) in data.iter().enumerate() {
            let label = labels[i];

            // Increment document count for the class.
            *self.doc_counts.entry(label).or_insert(0) += 1;

            // Increment word counts for the class.
            let word_freqs = self
                .class_word_frequencies
                .entry(label)
                .or_insert_with(|| vec![0; self.vocabulary_size]);
            for (word_index, &count) in doc_counts.iter().enumerate() {
                word_freqs[word_index] += count as u64;
                *self.word_counts.entry(label).or_insert(0) += count as u64;
            }
        }
        Ok(())
    }

    /// The `predict` method for making a prediction on a single document.
    /// It now returns a vector of class IDs that meet a certain accuracy threshold.
    fn predict(&self, data_point: &Vec<u16>) -> u32 {
        let mut best_class = 0;
        let mut max_posterior = f64::NEG_INFINITY;

        // Iterate through each class to calculate the posterior probability
        for (&class, &log_prior) in &self.class_log_priors {
            let mut log_posterior = log_prior;

            // Calculate the log likelihood for the document's words
            if let Some(log_likelihoods) = self.class_log_likelihoods.get(&class) {
                for (i, &word_count) in data_point.iter().enumerate() {
                    // We add the log probability for each occurrence of the word.
                    log_posterior += word_count as f64 * log_likelihoods[i];
                }
            }

            // Check if this class has the highest posterior probability so far.
            if log_posterior > max_posterior {
                max_posterior = log_posterior;
                best_class = class;
            }
        }

        best_class
    }

    /// This new function returns multiple categories based on a relative accuracy threshold.
    fn predict_multi(&self, data_point: &Vec<u16>) -> Vec<u32> {
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
        let mut results: Vec<u32> = Vec::new();
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

    // This public method finalizes the training by calculating the log probabilities
    // after all data has been incrementally passed to the `fit` method.
    fn finalize_training(&mut self) {
        let alpha = ALPHA; // Laplace smoothing alpha value
        for (&class, &doc_count) in &self.doc_counts {
            // Calculate log prior: ln(P(class))
            let log_prior = (doc_count as f64 / self.total_docs_processed as f64).ln();
            self.class_log_priors.insert(class, log_prior);

            // Calculate word likelihoods: P(word | class)
            let total_words_in_class = *self.word_counts.get(&class).unwrap_or(&0) as f64;
            let mut log_likelihoods_vec = vec![0.0; self.vocabulary_size];

            let class_word_freqs = self.class_word_frequencies.get(&class).unwrap();

            for i in 0..self.vocabulary_size {
                // Apply Laplace smoothing to prevent zero probabilities.
                let word_count = class_word_freqs[i] as f64;
                let numerator = word_count + alpha;
                let denominator = total_words_in_class + alpha * (self.vocabulary_size as f64);

                log_likelihoods_vec[i] = (numerator / denominator).ln();
            }
            self.class_log_likelihoods
                .insert(class, log_likelihoods_vec);
        }
    }
}


impl NaiveBayes {
    /// Saves the trained model to a file using bincode 2.0 for serialization.
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create the bincode configuration
        let config = bincode::config::standard();
        
        // Serialize the model
        let encoded = bincode::encode_to_vec(self, config)?;
        
        // Write to file
        let mut file = File::create(path)?;
        file.write_all(&encoded)?;
        file.flush()?; // Ensure data is written to disk
        
        Ok(())
    }

    /// Loads a trained model from a file using bincode 2.0 for deserialization.
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Read file contents
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        // Create the bincode configuration
        let config = bincode::config::standard();
        
        // Deserialize the model
        let (model, _len): (Self, usize) = bincode::decode_from_slice(&buffer, config)?;
        
        Ok(model)
    }
}

// Alternative implementation using references for better memory efficiency
impl NaiveBayes {
    /// Alternative save method that works with borrowed data
    pub fn save_to_file_alt(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        let config = bincode::config::standard();
        
        // Encode directly to the writer for better memory efficiency
        bincode::encode_into_std_write(self, &mut file, config)?;
        file.flush()?;
        
        Ok(())
    }

    /// Alternative load method using a reader
    pub fn load_from_file_alt(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let config = bincode::config::standard();
        
        // Decode directly from the reader
        let model: Self = bincode::decode_from_std_read(&mut file, config)?;
        
        Ok(model)
    }
}