/// Linear SVM with Pegasos SGD training.
/// Multi-class via One-vs-Rest (OvR).
/// Input: sparse BoW / TF-IDF vectors (HashMap<u16, f32>).
/// Output: Vec<u8> of predicted class labels (multi-label via score threshold).
use rkyv::{Archive, Deserialize, Serialize};
use rustc_hash::FxBuildHasher;
use std::collections::HashMap;

pub type SparseVec = HashMap<u16, f32, FxBuildHasher>;

// ─── single binary (OvR) classifier ────────────────────────────────────────

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct LinearBinaryClassifier {
    /// Sparse weight vector: word_id → weight
    pub weights: HashMap<u16, f32>,
    pub bias: f32,
    pub class_id: u8,
}

impl LinearBinaryClassifier {
    fn new(class_id: u8) -> Self {
        Self {
            weights: HashMap::new(),
            bias: 0.0,
            class_id,
        }
    }

    /// Dot product of sparse weight vector with sparse input.
    #[inline]
    fn decision_score(&self, x: &SparseVec) -> f32 {
        let mut dot = self.bias;
        // Iterate over the smaller map
        if self.weights.len() < x.len() {
            for (id, w) in &self.weights {
                if let Some(v) = x.get(id) {
                    dot += w * v;
                }
            }
        } else {
            for (id, v) in x {
                if let Some(w) = self.weights.get(id) {
                    dot += w * v;
                }
            }
        }
        dot
    }

    /// Pegasos SGD training.
    ///
    /// `data`         – TF-IDF sparse vectors
    /// `binary_labels`– +1 for the target class, -1 for all others
    /// `lambda`       – regularisation strength (typical: 1e-4 … 1e-2)
    /// `epochs`       – how many passes over the data
    pub fn train(
        &mut self,
        data: &[SparseVec],
        binary_labels: &[i8],
        lambda: f32,
        epochs: usize,
    ) {
        let n = data.len();
        if n == 0 {
            return;
        }

        // Start t at ceil(1/lambda) so the initial learning rate eta = 1/(lambda*t)
        // equals 1.0, regardless of lambda. Starting at t=1 gives eta_initial = 1/lambda,
        // which is 100 for lambda=0.01 or 10,000 for lambda=1e-4 — a single step
        // then drives the bias to ±(1/lambda), and subsequent shrinkage can never recover.
        let mut t: usize = (1.0_f32 / lambda).ceil() as usize;

        for _epoch in 0..epochs {
            for (x, &y) in data.iter().zip(binary_labels.iter()) {
                let eta = 1.0 / (lambda * t as f32); // step size
                let score = (y as f32) * self.decision_score(x);

                // Pegasos update
                if score < 1.0 {
                    // Hinge loss gradient: scale weights + add x*y*eta
                    let scale = 1.0 - eta * lambda;
                    // Scale existing weights
                    for w in self.weights.values_mut() {
                        *w *= scale;
                    }
                    // Add gradient contribution from this sample
                    let grad_coeff = eta * (y as f32);
                    for (id, &v) in x {
                        *self.weights.entry(*id).or_insert(0.0) += grad_coeff * v;
                    }
                    self.bias += eta * (y as f32);
                } else {
                    // No loss: only apply L2 regularisation to weights
                    let scale = 1.0 - eta * lambda;
                    for w in self.weights.values_mut() {
                        *w *= scale;
                    }
                }

                t += 1;
            }
        }

        // Project onto L2 ball ||w|| ≤ 1/sqrt(lambda)  (Pegasos guarantee)
        let norm_sq: f32 = self.weights.values().map(|w| w * w).sum();
        let max_norm_sq = 1.0 / lambda;
        if norm_sq > max_norm_sq {
            let scale = (max_norm_sq / norm_sq).sqrt();
            for w in self.weights.values_mut() {
                *w *= scale;
            }
        }

        // Prune near-zero weights to keep memory down
        self.weights.retain(|_, w| w.abs() > 1e-6);
    }
}

// ─── Multi-class SVM (OvR) ──────────────────────────────────────────────────

#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct SupportVectorMachine {
    /// One binary classifier per class
    pub classifiers: Vec<LinearBinaryClassifier>,

    // Hyper-parameters stored for reference / serialisation
    pub lambda: f32,
    pub epochs: usize,

    /// A class is included in multi-label output when its decision score is
    /// ≥ `score_threshold`.  Tune between 0.0 and 1.0.
    pub score_threshold: f32,

    /// Maximum number of labels to return per prediction.
    pub max_labels: usize,
}

impl SupportVectorMachine {
    /// `lambda`          – Pegasos regularisation (1e-4 is a safe default)
    /// `epochs`          – training passes (3–10 usually enough)
    /// `score_threshold` – min decision score to include a class in output
    /// `max_labels`      – cap on how many labels to return
    pub fn new(lambda: f32, epochs: usize, score_threshold: f32, max_labels: usize) -> Self {
        Self {
            classifiers: Vec::new(),
            lambda,
            epochs,
            score_threshold,
            max_labels,
        }
    }

    /// Build binary label vector: +1 for `target`, -1 for everything else.
    fn make_binary_labels(labels: &[Vec<u8>], target: u8) -> Vec<i8> {
        labels
            .iter()
            .map(|lv| if lv.contains(&target) { 1 } else { -1 })
            .collect()
    }

    /// Collect all unique class IDs from a label set.
    fn unique_classes(labels: &[Vec<u8>]) -> Vec<u8> {
        let mut set: std::collections::HashSet<u8> = std::collections::HashSet::new();
        for lv in labels {
            for &l in lv {
                set.insert(l);
            }
        }
        let mut v: Vec<u8> = set.into_iter().collect();
        v.sort_unstable();
        v
    }

    // ── public API ──────────────────────────────────────────────────────────

    /// Train one OvR binary classifier per class.
    ///
    /// `data`   – TF-IDF sparse vectors (same format as KNN input)
    /// `labels` – multi-label: each entry is a `Vec<u8>` of category IDs
    pub fn train(&mut self, data: &[SparseVec], labels: &[Vec<u8>]) -> Result<(), String> {
        if data.is_empty() || data.len() != labels.len() {
            return Err("data and labels must be non-empty and equal length".into());
        }

        let classes = Self::unique_classes(labels);
        if classes.len() < 2 {
            return Err("need at least 2 classes".into());
        }

        self.classifiers.clear();

        for class_id in classes {
            let binary_labels = Self::make_binary_labels(labels, class_id);
            let mut clf = LinearBinaryClassifier::new(class_id);
            clf.train(data, &binary_labels, self.lambda, self.epochs);
            self.classifiers.push(clf);
        }

        Ok(())
    }

    /// Return decision scores for all classes, sorted descending.
    pub fn score_all(&self, x: &SparseVec) -> Vec<(u8, f32)> {
        let mut scores: Vec<(u8, f32)> = self
            .classifiers
            .iter()
            .map(|clf| (clf.class_id, clf.decision_score(x)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores
    }

    /// Predict: returns all classes with score ≥ threshold,
    /// capped at `max_labels`.  Falls back to the top-1 class if nothing
    /// clears the threshold.
    pub fn predict(&self, x: &SparseVec) -> Vec<u8> {
        if self.classifiers.is_empty() {
            return vec![];
        }

        let scores = self.score_all(x);

        let mut result: Vec<u8> = scores
            .iter()
            .take(self.max_labels)
            .filter(|(_, s)| *s >= self.score_threshold)
            .map(|(c, _)| *c)
            .collect();

        // Always return at least the top-1 prediction
        if result.is_empty() {
            result.push(scores[0].0);
        }

        result
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxBuildHasher;

    fn make_sparse(pairs: &[(u16, f32)]) -> SparseVec {
        let mut m = HashMap::with_hasher(FxBuildHasher::default());
        for &(k, v) in pairs {
            m.insert(k, v);
        }
        m
    }

    #[test]
    fn test_binary_separable() {
        // Class 0: high word 0;  Class 1: high word 1
        let data = vec![
            make_sparse(&[(0, 5.0)]),
            make_sparse(&[(0, 4.0)]),
            make_sparse(&[(1, 5.0)]),
            make_sparse(&[(1, 4.0)]),
        ];
        let labels = vec![vec![0u8], vec![0u8], vec![1u8], vec![1u8]];

        let mut svm = SupportVectorMachine::new(1e-3, 10, 0.0, 4);
        svm.train(&data, &labels).unwrap();

        let p0 = svm.predict(&make_sparse(&[(0, 5.0)]));
        let p1 = svm.predict(&make_sparse(&[(1, 5.0)]));
        println!("p0={p0:?}  p1={p1:?}");
        assert_eq!(p0[0], 0);
        assert_eq!(p1[0], 1);
    }

    #[test]
    fn test_multi_class() {
        let data: Vec<SparseVec> = (0u16..6)
            .map(|i| make_sparse(&[(i % 3, (i + 1) as f32)]))
            .collect();
        let labels = vec![
            vec![0u8], vec![1u8], vec![2u8],
            vec![0u8], vec![1u8], vec![2u8],
        ];

        let mut svm = SupportVectorMachine::new(1e-3, 20, 0.0, 3);
        svm.train(&data, &labels).unwrap();
        assert_eq!(svm.classifiers.len(), 3);
    }
}
