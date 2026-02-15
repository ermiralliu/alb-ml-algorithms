use std::{collections::HashMap, fs::File, io::Write};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use super::MachineLearningModel;

// This trait bound is necessary to allow for calculating the distance between data points.
// We'll implement it for Vec<f64> to handle numerical data.

// This trait bound is necessary to allow for calculating the distance between data points.
// We'll implement it for Vec<f64> to handle numerical data.
pub trait EuclideanDistance<Other> {
  fn euclidean_distance(&self, other: &Other) -> f64;
}

// Implement the EuclideanDistance trait for Vec<f64>
// Implement the EuclideanDistance trait for Vec<f64>
impl EuclideanDistance<Vec<f64>> for Vec<f64> {
  fn euclidean_distance(&self, other: &Vec<f64>) -> f64 {
    self
      .iter()
      .zip(other.iter())
      .map(|(a, b)| (a - b).powi(2))
      .sum::<f64>()
      .sqrt()
  }
}
impl EuclideanDistance<Vec<u16>> for Vec<u16> {
    fn euclidean_distance(&self, other: &Vec<u16>) -> f64 {
        self.iter()
            .zip(other.iter())
            .map(|(&a, &b)| {
                let diff = (a as f64) - (b as f64);
                diff * diff
            })
            .sum::<f64>()
            .sqrt()
    }
}

// Define the K-NN struct. It is generic over the input and output types.
#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
pub struct KNearestNeighbors<I, O> {
  k: usize,
  training_data: Vec<Vec<I>>,
  training_labels: Vec<Vec<O>>,
}

// This is the custom constructor for KNN, as the `new()` method in the trait
// does not allow for passing the `k` parameter, which is essential for K-NN.
impl<I, O> KNearestNeighbors<I, O> {
  pub fn new_with_k(k: usize) -> Self {
    Self {
      k,
      training_data: Vec::new(),
      training_labels: Vec::new(),
    }
  }
}

// The implementation of the MachineLearningModel trait for KNearestNeighbors.
impl<I, O> MachineLearningModel<I, O> for KNearestNeighbors<I, O>
where
  // The change here is to require Vec<I> to implement EuclideanDistance<Vec<I>>,
  // which aligns with the types being passed to the `euclidean_distance` method.
  Vec<I>: EuclideanDistance<Vec<I>>,
  I: Clone,
  O: PartialEq + Eq + std::hash::Hash + Clone,
{
  // The `new` method required by the trait. It uses a default k of 3.
  // For a real-world application, it's better to use `new_with_k` to set this value.
  fn new() -> Self {
    Self::new_with_k(3)
  }
  /// The `train` method for K-NN simply stores the training data and labels.
  fn train(&mut self, data: &[Vec<I>], labels: &[Vec<O>]) -> Result<(), String> {
    if data.len() != labels.len() {
      return Err("Data and labels must have the same length.".to_string());
    }
    self.training_data = data.to_vec();
    self.training_labels = labels.to_vec();
    Ok(())
  }

  /// The `predict` method finds the k-nearest neighbors and returns the
  /// unique classes among them.
  fn predict(&self, data_point: &Vec<I>) -> Vec<O> {
    // Calculate the distance to every training data point.
    let mut distances: Vec<(f64, &Vec<O>)> = self
      .training_data
      .iter()
      .zip(self.training_labels.iter())
      .map(|(d, l)| (d.euclidean_distance(data_point), l))
      .collect();

    // Sort the neighbors by distance.
    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Get the labels of the k-nearest neighbors and flatten them into a single vector.
    let k_neighbors_labels = distances
      .iter()
      .take(self.k)
      .flat_map(|(_, labels_vec)| labels_vec.iter())
      .cloned()
      .collect::<Vec<_>>();

    // Count the frequency of each label.
    let mut counts = HashMap::new();
    for label in &k_neighbors_labels {
      *counts.entry(label.clone()).or_insert(0) += 1;
    }

    // Collect all labels that appeared in the k-nearest neighbors.
    let unique_labels = counts.keys().cloned().collect::<Vec<_>>();

    unique_labels
  }
}

impl super::SaveAndLoad<KNearestNeighbors<u16, u16>> for KNearestNeighbors<u16, u16> {
  /// Alternative save method that works with borrowed data
  fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    let config = bincode::config::standard();

    // Encode directly to the writer for better memory efficiency
    bincode::encode_into_std_write(self, &mut file, config)?;
    file.flush()?;

    Ok(())
  }

  /// Alternative load method using a reader
  fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let config = bincode::config::standard();

    // Decode directly from the reader
    let model: Self = bincode::decode_from_std_read(&mut file, config)?;

    Ok(model)
  }
}

pub type KNearestNeighborsAlbTextClassification = KNearestNeighbors<u16, u16>;
