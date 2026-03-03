use std::collections::{BinaryHeap, HashMap};

use ordered_float::OrderedFloat;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;

// 1. Fixed Dot Product: No more integer truncation!
fn calculate_cosine_similarity(
  query: &HashMap<u16, f32, FxBuildHasher>,
  train_vec: &HashMap<u16, f32, FxBuildHasher>,
  query_mag: f64,
  train_mag: f64,
) -> f64 {
  if query_mag == 0.0 || train_mag == 0.0 {
    return 0.0;
  }

  let mut dot = 0.0;
  // Iterate over the smaller map for speed
  if query.len() < train_vec.len() {
    for (id, q_val) in query {
      if let Some(t_val) = train_vec.get(id) {
        dot += (*q_val as f64) * (*t_val as f64);
      }
    }
  } else {
    for (id, t_val) in train_vec {
      if let Some(q_val) = query.get(id) {
        dot += (*q_val as f64) * (*t_val as f64);
      }
    }
  }

  let res = dot / (query_mag * train_mag);
  // dbg!(res);
  res
}

pub struct KNearestNeighbors {
  k: usize,
  training_data: Vec<HashMap<u16, f32, FxBuildHasher>>,
  training_labels: Vec<Vec<u8>>,
  normalized_magnitudes: Vec<f64>,
  classes_weights: HashMap<u8, f64, FxBuildHasher>,
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
  pub fn train(
    &mut self,
    data: &[HashMap<u16, f32, FxBuildHasher>],
    labels: &[Vec<u8>],
  ) -> Result<(), String> {
    if data.len() != labels.len() {
      return Err("Data and labels must have the same length.".to_string());
    }

    self.training_data = data.to_vec();
    self.training_labels = labels.to_vec();

    // Fix 2: Proper float magnitude calculation
    self.normalized_magnitudes = data
      .iter()
      .map(|vec| {
        let sum_sq: f64 = vec.values().map(|&val| (val as f64).powi(2)).sum();
        sum_sq.sqrt()
      })
      .collect();

    let mut class_counts: HashMap<u8, usize, FxBuildHasher> = HashMap::default();
    for label_vec in labels {
      for &label in label_vec {
        *class_counts.entry(label).or_insert(0) += 1;
      }
    }

    let total_samples = labels.len() as f64;
    // Fix 3: Actually assign the weights to the struct field
    // self.classes_weights = class_counts
    //   .into_iter()
    //   .map(|(label, count)| (label, total_samples / count as f64))
    //   .collect();
    self.classes_weights = class_counts
      .into_iter()
      .map(|(label, count)| {
        let weight = (total_samples / count as f64).ln() + 1.0;
        (label, weight)
      })
      .collect();

    Ok(())
  }

  pub fn predict_new(&self, data_point: &HashMap<u16, f32, FxBuildHasher>) -> Vec<u8> {
    use std::cmp::Reverse;

    // Fix 4: Pre-calculate query magnitude once
    let query_mag: f64 = data_point
      .values()
      .map(|&v| (v as f64).powi(2))
      .sum::<f64>()
      .sqrt();
    if query_mag == 0.0 {
      return vec![];
    }

    let heap = self
      .training_data
      .par_iter()
      .enumerate()
      .fold(
        || BinaryHeap::with_capacity(self.k),
        |mut heap, (i, train_vec)| {
          let sim = calculate_cosine_similarity(
            data_point,
            train_vec,
            query_mag,
            self.normalized_magnitudes[i],
          );
          let score = Reverse(OrderedFloat(sim));

          if heap.len() < self.k {
            heap.push((score, i));
          } else if let Some(worst) = heap.peek() {
            if score < worst.0 {
              let _popped = heap.pop();
              // dbg!(popped);
              // dbg!(score);
              heap.push((score, i));
            }
            // dbg!(&heap);
            // println!("{:?}", heap);
          }
          heap
        },
      )
      .reduce(
        || BinaryHeap::with_capacity(self.k),
        |mut h1, h2| {
          // println!("{:?}", h1);
          for item in h2 {
            if h1.len() < self.k {
              h1.push(item);
            } else if item.0 < h1.peek().unwrap().0 {
              h1.pop();
              h1.push(item);
            }
          }
          h1
        },
      );
    // println!("Final heap: {:?}", heap);

    // Voting logic
    let mut weighted_counts: HashMap<u8, f64, FxBuildHasher> = HashMap::default();

    let mut dot_total_size = 0.0;
    for (Reverse(score), idx) in heap {
      dot_total_size += score.into_inner();
      for label in &self.training_labels[idx] {
        // let class_weight = self.classes_weights.get(label).unwrap_or(&1.0);
        *weighted_counts.entry(*label).or_insert(0.0) += score.into_inner();
      }
    }

    if dot_total_size == 0.0 {
      return vec![];
    }

    // let threshold = 0.9 / self.k as f64; // this needs better logic
    let max_conf = weighted_counts.values().cloned().fold(0.0_f64, f64::max);
    let threshold = max_conf * 0.4; // only keep labels within 50% of the top
                                    
    weighted_counts
      .into_iter()
      .filter_map(|(label, sum_weight)| {
        // let conf = sum_weight / dot_total_size;
        let conf = sum_weight;
        // dbg!(&label, conf);
        if conf >= threshold { Some(label) } else { None }
      })
      .collect()
  }
}
