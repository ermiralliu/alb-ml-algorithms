/// metrics.rs
/// Multi-label classification metrics.
///
/// For each class we compute:
///   TP – predicted AND in ground truth
///   FP – predicted but NOT in ground truth
///   FN – in ground truth but NOT predicted
///
/// Then:
///   Precision = TP / (TP + FP)
///   Recall    = TP / (TP + FN)
///   F1        = 2 * P * R / (P + R)
///
/// Macro averages are the unweighted mean across all classes.
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct ClassMetrics {
    pub class_id: u8,
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize, // number of ground-truth positives
}

#[derive(Debug)]
pub struct MetricsReport {
    pub per_class: Vec<ClassMetrics>,
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub jaccard_avg: f64,
    pub n_samples: usize,
}

/// Accumulates per-class TP/FP/FN counts across a batch of predictions.
pub struct MetricsAccumulator {
    tp: HashMap<u8, usize>,
    fp: HashMap<u8, usize>,
    fn_: HashMap<u8, usize>,
    jaccard_sum: f64,
    n_samples: usize,
}

impl MetricsAccumulator {
    pub fn new() -> Self {
        Self {
            tp: HashMap::new(),
            fp: HashMap::new(),
            fn_: HashMap::new(),
            jaccard_sum: 0.0,
            n_samples: 0,
        }
    }

    /// Record one sample: `predicted` and `actual` are sets of class IDs.
    pub fn update(&mut self, predicted: &[u8], actual: &[u8]) {
        let pred_set: HashSet<u8> = predicted.iter().copied().collect();
        let true_set: HashSet<u8> = actual.iter().copied().collect();

        // TP
        for &c in pred_set.intersection(&true_set) {
            *self.tp.entry(c).or_insert(0) += 1;
        }
        // FP
        for &c in pred_set.difference(&true_set) {
            *self.fp.entry(c).or_insert(0) += 1;
        }
        // FN
        for &c in true_set.difference(&pred_set) {
            *self.fn_.entry(c).or_insert(0) += 1;
        }

        // Jaccard similarity for this sample
        let intersection = pred_set.intersection(&true_set).count();
        let union = pred_set.union(&true_set).count();
        self.jaccard_sum += if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        };
        self.n_samples += 1;
    }

    /// Compute the final report.
    pub fn report(&self) -> MetricsReport {
        // Collect all class ids
        let mut all_classes: HashSet<u8> = HashSet::new();
        all_classes.extend(self.tp.keys());
        all_classes.extend(self.fp.keys());
        all_classes.extend(self.fn_.keys());

        let mut per_class: Vec<ClassMetrics> = all_classes
            .into_iter()
            .map(|c| {
                let tp = *self.tp.get(&c).unwrap_or(&0);
                let fp = *self.fp.get(&c).unwrap_or(&0);
                let fn_ = *self.fn_.get(&c).unwrap_or(&0);
                let support = tp + fn_;
                let precision = if tp + fp == 0 {
                    0.0
                } else {
                    tp as f64 / (tp + fp) as f64
                };
                let recall = if tp + fn_ == 0 {
                    0.0
                } else {
                    tp as f64 / (tp + fn_) as f64
                };
                let f1 = if precision + recall == 0.0 {
                    0.0
                } else {
                    2.0 * precision * recall / (precision + recall)
                };
                ClassMetrics {
                    class_id: c,
                    tp,
                    fp,
                    fn_,
                    precision,
                    recall,
                    f1,
                    support,
                }
            })
            .collect();

        per_class.sort_by_key(|m| m.class_id);

        let n = per_class.len() as f64;
        let macro_precision = per_class.iter().map(|m| m.precision).sum::<f64>() / n;
        let macro_recall = per_class.iter().map(|m| m.recall).sum::<f64>() / n;
        let macro_f1 = per_class.iter().map(|m| m.f1).sum::<f64>() / n;
        let jaccard_avg = if self.n_samples == 0 {
            0.0
        } else {
            self.jaccard_sum / self.n_samples as f64
        };

        MetricsReport {
            per_class,
            macro_precision,
            macro_recall,
            macro_f1,
            jaccard_avg,
            n_samples: self.n_samples,
        }
    }
}

impl MetricsReport {
    /// Pretty-print a table to stdout.
    pub fn print(&self, label_name: &dyn Fn(u8) -> &'static str) {
        println!(
            "\n{:<25} {:>8} {:>8} {:>8} {:>8}",
            "Class", "Prec", "Recall", "F1", "Support"
        );
        println!("{}", "-".repeat(61));

        for m in &self.per_class {
            println!(
                "{:<25} {:>8.4} {:>8.4} {:>8.4} {:>8}",
                label_name(m.class_id),
                m.precision,
                m.recall,
                m.f1,
                m.support
            );
        }

        println!("{}", "-".repeat(61));
        println!(
            "{:<25} {:>8.4} {:>8.4} {:>8.4}",
            "macro avg",
            self.macro_precision,
            self.macro_recall,
            self.macro_f1
        );
        println!("Jaccard avg (samples):  {:.4}", self.jaccard_avg);
        println!("Evaluated on {} samples", self.n_samples);
    }

    /// Print a condensed confusion-style matrix showing TP/FP/FN per class.
    pub fn print_confusion_matrix(&self, label_name: &dyn Fn(u8) -> &'static str) {
        println!(
            "\n{:<25} {:>6} {:>6} {:>6}",
            "Class", "TP", "FP", "FN"
        );
        println!("{}", "-".repeat(45));
        for m in &self.per_class {
            println!(
                "{:<25} {:>6} {:>6} {:>6}",
                label_name(m.class_id),
                m.tp,
                m.fp,
                m.fn_
            );
        }
    }
}
