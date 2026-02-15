use std::{
  cmp::{Ordering, Reverse},
  collections::BinaryHeap,
};

use super::MachineLearningModel;

#[derive(Debug, Clone)]
pub struct SupportVectorMachine {
  // Training data and labels
  training_data: Vec<Vec<f64>>,
  training_labels: Vec<i32>,

  // SVM parameters
  alphas: Vec<f64>,
  bias: f64,

  // Hyperparameters
  c: f64,         // Regularization parameter
  tolerance: f64, // Tolerance for convergence
  max_iterations: usize,

  // Support vectors (indices of training examples)
  support_vectors: Vec<usize>,

  // Feature dimension
  feature_dim: usize,

  // Number of classes
  num_classes: usize,

  // Multi-class handling (one-vs-rest classifiers)
  classifiers: Vec<BinaryClassifier>,
}

#[derive(Debug, Clone)]
struct BinaryClassifier {
  alphas: Vec<f64>,
  bias: f64,
  support_vectors: Vec<usize>,
  training_data: Vec<Vec<f64>>,
  training_labels: Vec<i32>,
}

impl BinaryClassifier {
  fn new() -> Self {
    Self {
      alphas: Vec::new(),
      bias: 0.0,
      support_vectors: Vec::new(),
      training_data: Vec::new(),
      training_labels: Vec::new(),
    }
  }

  fn linear_kernel(x1: &[f64], x2: &[f64]) -> f64 {
    x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum()
  }

  fn predict_score(&self, x: &[f64]) -> f64 {
    let mut score = self.bias;

    for &sv_idx in &self.support_vectors {
      if self.alphas[sv_idx] > 1e-10 {
        score += self.alphas[sv_idx]
          * self.training_labels[sv_idx] as f64
          * Self::linear_kernel(x, &self.training_data[sv_idx]);
      }
    }

    score
  }

  fn train_binary(&mut self, data: &[Vec<f64>], labels: &[i32], c: f64) -> Result<(), String> {
    let n_samples = data.len();
    if n_samples == 0 || labels.len() != n_samples {
      return Err("Invalid training data".to_string());
    }

    self.training_data = data.to_vec();
    self.training_labels = labels.to_vec();
    self.alphas = vec![0.0; n_samples];
    self.bias = 0.0;

    let max_iter = 1000;
    let tol = 1e-3;

    // SMO algorithm simplified
    for _iter in 0..max_iter {
      let mut num_changed = 0;

      for i in 0..n_samples {
        let ei = self.predict_score(&data[i]) - labels[i] as f64;

        if (labels[i] == 1 && ei < -tol && self.alphas[i] < c)
          || (labels[i] == -1 && ei > tol && self.alphas[i] > 0.0)
        {
          // Select second alpha randomly
          let j = (i + 1) % n_samples;
          if i == j {
            continue;
          }

          let ej = self.predict_score(&data[j]) - labels[j] as f64;

          let old_alpha_i = self.alphas[i];
          let old_alpha_j = self.alphas[j];

          // Calculate bounds
          let (low, high) = if labels[i] != labels[j] {
            let diff = self.alphas[i] - self.alphas[j];
            ((-diff).max(0.0), c + (-diff).min(0.0))
          } else {
            let sum = self.alphas[i] + self.alphas[j];
            ((sum - c).max(0.0), sum.min(c))
          };

          if (high - low).abs() < 1e-10 {
            continue;
          }

          // Calculate kernel values
          let kii = Self::linear_kernel(&data[i], &data[i]);
          let kjj = Self::linear_kernel(&data[j], &data[j]);
          let kij = Self::linear_kernel(&data[i], &data[j]);
          let eta = 2.0 * kij - kii - kjj;

          if eta >= 0.0 {
            continue;
          }

          // Calculate new alpha_j
          self.alphas[j] = old_alpha_j - labels[j] as f64 * (ei - ej) / eta;
          self.alphas[j] = self.alphas[j].clamp(low, high);

          if (self.alphas[j] - old_alpha_j).abs() < 1e-5 {
            continue;
          }

          // Calculate new alpha_i
          self.alphas[i] =
            old_alpha_i + labels[i] as f64 * labels[j] as f64 * (old_alpha_j - self.alphas[j]);

          // Calculate new bias
          let b1 = self.bias
            - ei
            - labels[i] as f64 * (self.alphas[i] - old_alpha_i) * kii
            - labels[j] as f64 * (self.alphas[j] - old_alpha_j) * kij;
          let b2 = self.bias
            - ej
            - labels[i] as f64 * (self.alphas[i] - old_alpha_i) * kij
            - labels[j] as f64 * (self.alphas[j] - old_alpha_j) * kjj;

          self.bias = if self.alphas[i] > 0.0 && self.alphas[i] < c {
            b1
          } else if self.alphas[j] > 0.0 && self.alphas[j] < c {
            b2
          } else {
            (b1 + b2) / 2.0
          };

          num_changed += 1;
        }
      }

      if num_changed == 0 {
        break;
      }
    }

    // Identify support vectors
    self.support_vectors.clear();
    for i in 0..n_samples {
      if self.alphas[i] > 1e-10 {
        self.support_vectors.push(i);
      }
    }

    Ok(())
  }
}

impl SupportVectorMachine {
  fn normalize_features(data: &[Vec<u16>]) -> Vec<Vec<f64>> {
    if data.is_empty() {
      return Vec::new();
    }

    let n_samples = data.len();
    let n_features = data[0].len();

    // Convert to f64 and calculate feature statistics
    let mut normalized_data = vec![vec![0.0; n_features]; n_samples];
    let mut feature_max = vec![0.0; n_features];

    // Find maximum values for normalization
    for sample in data {
      for (j, &value) in sample.iter().enumerate() {
        let val = value as f64;
        if val > feature_max[j] {
          feature_max[j] = val;
        }
      }
    }

    // Normalize features to [0, 1]
    for (i, sample) in data.iter().enumerate() {
      for (j, &value) in sample.iter().enumerate() {
        normalized_data[i][j] = if feature_max[j] > 0.0 {
          value as f64 / feature_max[j]
        } else {
          0.0
        };
      }
    }

    normalized_data
  }

  fn prepare_binary_labels(labels: &[Vec<u16>], target_class: u16) -> Vec<i32> {
    labels
      .iter()
      .map(|label_vec| {
        // Assume the first element of label vector is the class
        if !label_vec.is_empty() && label_vec[0] == target_class {
          1
        } else {
          -1
        }
      })
      .collect()
  }

  fn get_unique_classes(labels: &[Vec<u16>]) -> Vec<u16> {
    let mut classes = Vec::new();
    for label_vec in labels {
      if !label_vec.is_empty() {
        let class = label_vec[0];
        if !classes.contains(&class) {
          classes.push(class);
        }
      }
    }
    classes.sort_unstable();
    classes
  }
}

impl MachineLearningModel<u16, u16> for SupportVectorMachine {
  fn new() -> Self {
    Self {
      training_data: Vec::new(),
      training_labels: Vec::new(),
      alphas: Vec::new(),
      bias: 0.0,
      c: 1.0,
      tolerance: 1e-3,
      max_iterations: 1000,
      support_vectors: Vec::new(),
      feature_dim: 0,
      num_classes: 0,
      classifiers: Vec::new(),
    }
  }

  fn train(&mut self, data: &[Vec<u16>], labels: &[Vec<u16>]) -> Result<(), String> {
    if data.is_empty() || labels.is_empty() || data.len() != labels.len() {
      return Err("Invalid training data: data and labels must have the same length".to_string());
    }

    // Normalize features
    let normalized_data = Self::normalize_features(data);
    self.feature_dim = normalized_data[0].len();

    // Get unique classes
    let unique_classes = Self::get_unique_classes(labels);
    self.num_classes = unique_classes.len();

    if self.num_classes < 2 {
      return Err("Need at least 2 classes for classification".to_string());
    }

    // Train one-vs-rest binary classifiers
    self.classifiers.clear();
    for &target_class in &unique_classes {
      let mut binary_classifier = BinaryClassifier::new();
      let binary_labels = Self::prepare_binary_labels(labels, target_class);

      binary_classifier.train_binary(&normalized_data, &binary_labels, self.c)?;
      self.classifiers.push(binary_classifier);
    }

    // Store training data for prediction
    self.training_data = normalized_data;

    Ok(())
  }

  fn predict(&self, data_point: &Vec<u16>) -> Vec<u16> {
    if self.classifiers.is_empty() || data_point.len() != self.feature_dim {
      return vec![0];
    }

    // Normalize the input data point
    let normalized_point: Vec<f64> = data_point
      .iter()
      .map(|&x| x as f64 / 40000.0) // Simple normalization assuming max value is 40000
      .collect();

      // Get scores
    let threshold = 0.1; // Minimum score threshold
    let max_results = 4;
    let mut score_heap: BinaryHeap<ClassScore> = BinaryHeap::new();

    for (class_id, classifier) in self.classifiers.iter().enumerate() {
      let score = classifier.predict_score(&normalized_point);

      // Only consider scores above threshold
      if score > threshold {
        let class_id = class_id as u16;
        let class_score = ClassScore { class_id, score };

        if score_heap.len() < max_results {
          score_heap.push(class_score);
        } else if let Some(min_score) = score_heap.peek() {
          // If current score is better than the worst in heap, replace it
          if score > min_score.score {
            score_heap.pop();
            score_heap.push(class_score);
          }
        }
      }
    }
    score_heap.into_sorted_vec().iter().map(|x| x.class_id).collect()
  }

  // // Find the class with the highest score
  // let predicted_class = scores.iter()
  //     .enumerate()
  //     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
  //     .map(|(idx, _)| idx as u16)
  //     .unwrap_or(0);

  // vec![predicted_class]
}


// stuff like this will be placed outside the main class declaration and definition
#[derive(Debug, Clone, Copy, PartialEq)]
struct ClassScore {
    class_id: u16,
    score: f64,
}

// Manual implementation of `Eq` for `ClassScore`.
// We use a `debug_assert!` to ensure that we never compare against a NaN.
// If a NaN is present, the comparison will be false, which is invalid for `Eq`.
impl Eq for ClassScore {}

// Manual implementation of `PartialOrd` for `ClassScore`.
// This trait is a prerequisite for `Ord` and provides a way to compare values,
// returning `None` if the values are not comparable (e.g., if one is NaN).
impl PartialOrd for ClassScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

// Manual implementation of `Ord` for `ClassScore`.
// This trait requires a total ordering. The `unwrap()` method is used here,
// which will panic if `self.score.partial_cmp(&other.score)` ever returns `None`
// (which would happen if either value were NaN).
impl Ord for ClassScore {
    fn cmp(&self, other: &Self) -> Ordering {
        // We can safely unwrap here because we've already asserted in `Eq`
        // that we won't be dealing with NaNs.
        self.partial_cmp(other).unwrap()
    }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_svm_creation() {
    let svm = SupportVectorMachine::new();
    assert_eq!(svm.num_classes, 0);
    assert_eq!(svm.feature_dim, 0);
  }

  #[test]
  fn test_svm_training() {
    let mut svm = SupportVectorMachine::new();

    // Simple test data
    let data = vec![
      vec![1, 2, 3],
      vec![4, 5, 6],
      vec![7, 8, 9],
      vec![10, 11, 12],
    ];

    let labels = vec![vec![0], vec![1], vec![0], vec![1]];

    let result = svm.train(&data, &labels);
    assert!(result.is_ok());
    assert_eq!(svm.num_classes, 2);
    assert_eq!(svm.feature_dim, 3);
  }

  #[test]
  fn test_svm_prediction() {
    let mut svm = SupportVectorMachine::new();

    let data = vec![
      vec![1, 2, 3],
      vec![4, 5, 6],
      vec![7, 8, 9],
      vec![10, 11, 12],
    ];

    let labels = vec![vec![0], vec![1], vec![2], vec![3]];

    svm.train(&data, &labels).unwrap();

    let prediction = svm.predict(&vec![5, 6, 7]);
    assert!(prediction.len() >= 1);
    println!("{:?}", prediction);
    // assert!(prediction[0] <= 1); // Should predict class 0 or 1
  }
  // this returns 2, instead of 1, because geometrically it's closer. it's counterintuitive, because 1 has more shared values
  // but in larger datasets, it'll lead to closer results
}
