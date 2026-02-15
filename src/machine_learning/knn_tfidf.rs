use std::collections::{BinaryHeap, HashMap};

use crate::TfidfBag;

use ordered_float::OrderedFloat;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;

fn dot_product(b: &TfidfBag, a: &TfidfBag, a_magnitude: f32) -> f64 {
  let mut dot = 0;
  let b_magnitude: f64 = b.values().map(|val| (*val as f64) * (*val as f64)).sum();
  let b_magnitude = b_magnitude.sqrt();
  for (word_id, val_a) in a {
    if let Some(val_b) = b.get(word_id) {
      let a = *val_a as u64;
      let b = *val_b as u64;
      dot += a * b; // This is the entire calculation!
    }
  }
  (dot as f64) / (b_magnitude * a_magnitude as f64)
}

pub struct KNearestNeighbors {
  k: usize,
  // I is now fixed as FxHashMap<u16, u16>
  training_data: Vec<TfidfBag>,
  training_labels: Vec<Vec<u16>>,
  normalized_magnitudes: Vec<f32>,
  classes_weights: HashMap<u16, f64, FxBuildHasher>,
}
impl KNearestNeighbors {
  pub fn new_with_k(k: usize) -> Self {
    Self {
      k,
      training_data: Vec::new(),
      training_labels: Vec::new(),
      normalized_magnitudes: Vec::new(),
      classes_weights: HashMap::default(),
    }
  }

  pub fn train(&mut self, data: &[TfidfBag], labels: &[Vec<u16>]) -> Result<(), String> {
    if data.len() != labels.len() {
      return Err("Data and labels must have the same length.".to_string());
    }
    // self.training_data = data.to_vec();
    self.training_data = data.to_vec();
    self.normalized_magnitudes = data
      .iter()
      .map(|x| {
        let sum: u32 = x.values().map(|val| (*val as u32).pow(2)).sum();
        (sum as f32).sqrt()
      })
      .collect();
    self.training_labels = labels.to_vec();
    let mut class_counts: HashMap<u16, usize> = HashMap::default();
    for label_vec in labels {
      for &label in label_vec {
        *class_counts.entry(label).or_insert(0) += 1;
      }
    }

    let total_samples = labels.len() as f64;
    let class_weights: HashMap<u16, f64, FxBuildHasher> = class_counts
      .into_iter()
      .map(|(label, count)| (label, total_samples / count as f64))
      .collect();
    Ok(())
  }

  pub fn predict_new(&self, data_point: &TfidfBag) -> Vec<u16> {
    use std::cmp::Reverse;

    // Use a Min-Heap (via Reverse) to keep the K largest similarities
    // Or keep track of K smallest distances if using euclidean_distance
    let heap = self
      .training_data
      .par_iter()
      .enumerate()
      .fold(
        || BinaryHeap::with_capacity(self.k),
        |mut heap: BinaryHeap<(Reverse<OrderedFloat<f64>>, usize)>, (i, train_vec)| {
          // For cosine similarity (larger is better), we want to keep the K largest
          let similarity = dot_product(data_point, train_vec, self.normalized_magnitudes[i]);
          let score = Reverse(OrderedFloat(similarity));

          // Or for euclidean distance (smaller is better):
          // let distance = train_vec.euclidean_distance(data_point);
          // let score = OrderedFloat(distance);  // No Reverse needed

          if heap.len() < self.k {
            heap.push((score, i));
          } else if let Some(worst) = heap.peek() {
            // For similarity: if new similarity is LARGER than worst (smallest) in heap
            if score > worst.0 {
              heap.pop();
              heap.push((score, i));
            }
          }
          heap
        },
      )
      .reduce(
        || BinaryHeap::with_capacity(self.k),
        |mut heap1, heap2| {
          // Merge heap2 into heap1, keeping only K best
          for item in heap2 {
            if heap1.len() < self.k {
              heap1.push(item);
            } else if let Some(worst) = heap1.peek() {
              if item.0 > worst.0 {
                heap1.pop();
                heap1.push(item);
              }
            }
          }
          heap1
        },
      );

    // // Voting logic
    // let threshold = 0.4;
    // let final_neighbors_count = heap.len();
    // let mut counts: HashMap<u16, u16, FxBuildHasher> = HashMap::default();
    // for (_, idx) in heap {
    //   for label in &self.training_labels[idx] {
    //     *counts.entry(*label).or_insert(0) += 1;
    //   }
    // }

    // counts
    //   .into_iter()
    //   .filter_map(|(label, count)| {
    //     let confidence = count as f64 / final_neighbors_count as f64;
    //     if confidence >= threshold {
    //       Some(label)
    //     } else {
    //       None
    //     }
    //   })
    //   .collect()
    //impl 2 of voting below
    // counts.into_keys().collect()
    // }
    // let final_neighbors_count = heap.len();
    // if final_neighbors_count == 0 {
    //   return vec![];
    // }

    // // Use f64 for weights to maintain precision
    // let mut weighted_counts: HashMap<u16, f64, FxBuildHasher> = HashMap::default();
    // let mut total_weight = 0.0;
    // let epsilon = 1e-9;

    // for (Reverse(score), idx) in heap {
    //   let similarity = score.into_inner() + epsilon;

    //   for label in &self.training_labels[idx] {
    //     // Look up the pre-calculated weight for this specific class
    //     let class_penalty_weight = self.classes_weights.get(label).unwrap_or(&1.0);

    //     // Combine Similarity Weight AND Class Weight
    //     let final_weight = similarity /* * class_penalty_weight */;

    //     *weighted_counts.entry(*label).or_insert(0.0) += final_weight;
    //     total_weight += final_weight;
    //   }
    // }
    // let threshold = 0.0;

    // weighted_counts
    //   .into_iter()
    //   .filter_map(|(label, sum_weight)| {
    //     // Confidence is now: (Sum of weights for this label) / (Total weight of all K neighbors)
    //     let confidence = sum_weight / total_weight;

    //     if confidence >= threshold {
    //       Some(label)
    //     } else {
    //       None
    //     }
    //   })
    //   .collect()
    // 1. Calculate weighted counts
    let mut weighted_counts: HashMap<u16, f64, FxBuildHasher> = HashMap::default();
    let mut total_weight = 0.0;
    let epsilon = 1e-9;

    for (Reverse(score), idx) in heap {
      // With normalized TF-IDF, score is likely between 0.0 and 1.0
      let similarity = score.into_inner() + epsilon;

      for label in &self.training_labels[idx] {
        let class_penalty_weight = self.classes_weights.get(label).unwrap_or(&1.0);

        // Weighting the vote by the document similarity
        let final_weight = similarity * (*class_penalty_weight as f64);

        *weighted_counts.entry(*label).or_insert(0.0) += final_weight;
        total_weight += final_weight;
      }
    }

    if total_weight == 0.0 {
      return vec![];
    }

    // 2. Find the maximum confidence so we don't return an empty set
    let max_confidence = weighted_counts
      .values()
      .map(|&v| v / total_weight)
      .fold(0.0, f64::max);

    // 3. Filter with a "Safety Valve"
    let threshold = 0.05; // Example: 15% confidence
    let results: Vec<u16> = weighted_counts
      .into_iter()
      .filter_map(|(label, sum_weight)| {
        let confidence = sum_weight / total_weight;

        // Fix: Return the label if it meets threshold OR if it's the absolute best
        // match and we would otherwise return nothing.
        if confidence >= threshold || (confidence == max_confidence && max_confidence > 0.0) {
          Some(label)
        } else {
          None
        }
      })
      .collect();

    results
  }
}
