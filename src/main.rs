pub mod machine_learning;
use rkyv::{Archive, Deserialize, Serialize};
use rustc_hash::FxBuildHasher;

use crate::machine_learning::{
  knn_hashmap::{KNearestNeighbors, SparseVector},
};

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq)]
#[rkyv(
    // This will generate a PartialEq impl between archived and normal types
    compare(PartialEq),
    // bytecheck can be used to validate your data if you want
    derive(Debug),
)]

struct Data {
  matrix: Vec<Vec<u16>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  println!("Hello, world!");
  let file_bytes_categories = std::fs::read("out/albanian_test_categories.rkyv")?;
  let categories_archived =
    rkyv::access::<ArchivedData, rkyv::rancor::Error>(&file_bytes_categories)?;
  let map_of_word_usages: Vec<SparseVector> = {
    let file_bytes_articles = std::fs::read("out/albanian_test.rkyv")?;
    let articles_archived =
      rkyv::access::<ArchivedData, rkyv::rancor::Error>(&file_bytes_articles)?;
    let article_matrix = rkyv::deserialize::<Data, rkyv::rancor::Error>(articles_archived)?;
    (&article_matrix.matrix)
      .iter()
      .map(|x| vectorize_sparse(x))
      .collect()
  };
  let categories_matrix: Data =
    rkyv::deserialize::<Data, rkyv::rancor::Error>(categories_archived)?;
  let mut knn_test = KNearestNeighbors::new_with_k(3);
  knn_test.train(&map_of_word_usages, &categories_matrix.matrix)?;
  // let data_point = &article_matrix.matrix[0];
  // knn_test.predict(data_point);
  // 1. Define your test set (using a slice of your data)
  let test_data = &map_of_word_usages;
  let test_labels = &categories_matrix.matrix;

  let mut correct_predictions = 0;
  let total_points = test_data.len();

  for i in 0..50 {
    let actual_label = &test_labels[i];
    let prediction = knn_test.predict_new(&test_data[i]);
    if prediction == *actual_label {
      correct_predictions += 1;
    }
    let accuracy = (correct_predictions as f64 / (i+1) as f64) * 100.0;
    println!("Prediction {i} finished, accuracy: {accuracy}");

    // Since predict returns a Vec<O>, check if it matches the actual label
    // If your labels are single-element vectors, compare directly:
  }

  let accuracy = (correct_predictions as f64 / total_points as f64) * 100.0;
  println!(
    "Model Accuracy: {}% ({} out of {})",
    accuracy, correct_predictions, total_points
  );
  Ok(())
}
fn vectorize(sequence: &[u16], vocab_size: usize) -> Vec<u16> {
  let mut vector = vec![0u16; vocab_size];
  for &word_id in sequence {
    if (word_id as usize) < vocab_size {
      // Increment the count at the index of this word
      vector[word_id as usize] += 1;
    }
  }
  vector
}

fn vectorize_sparse(sequence: &[u16]) -> SparseVector {
  // Initialize with the fast hasher
  let mut sparse_vec = SparseVector::with_hasher(FxBuildHasher::default());

  for &word_id in sequence {
    // .entry() handles checking if the key exists and updating it in one go
    *sparse_vec.entry(word_id).or_insert(0) += 1;
  }

  sparse_vec
}
