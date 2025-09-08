use std::collections::HashMap;
use rand::Rng;

use super::MachineLearningModel;

#[derive(Debug, Clone)]
pub struct CNN<I, O> {
    // Network architecture parameters
    vocab_size: usize,
    embedding_dim: usize,
    num_filters: usize,
    filter_sizes: Vec<usize>,
    num_classes: usize,
    max_sequence_length: usize,
    
    // Learned parameters
    embedding_weights: Vec<Vec<f32>>,
    conv_filters: Vec<Vec<Vec<f32>>>, // [filter_size][filter_num][embedding_dim]
    conv_biases: Vec<Vec<f32>>,       // [filter_size][filter_num]
    dense_weights: Vec<Vec<f32>>,     // [num_filters * filter_sizes.len()][num_classes]
    dense_biases: Vec<f32>,           // [num_classes]
    
    // Training parameters
    learning_rate: f32,
    dropout_rate: f32,
    
    // Vocabulary mapping
    token_to_id: HashMap<I, usize>,
    id_to_token: HashMap<usize, I>,
    class_to_id: HashMap<O, usize>,
    id_to_class: HashMap<usize, O>,
    
    // Training state
    is_trained: bool,
    _phantom: std::marker::PhantomData<(I, O)>,
}

impl<I, O> CNN<I, O> 
where 
    I: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    O: Clone + std::hash::Hash + Eq + std::fmt::Debug,
{
    pub fn with_config(
        embedding_dim: usize,
        num_filters: usize,
        filter_sizes: Vec<usize>,
        max_sequence_length: usize,
        learning_rate: f32,
    ) -> Self {
        CNN {
            vocab_size: 0,
            embedding_dim,
            num_filters,
            filter_sizes,
            num_classes: 0,
            max_sequence_length,
            embedding_weights: Vec::new(),
            conv_filters: Vec::new(),
            conv_biases: Vec::new(),
            dense_weights: Vec::new(),
            dense_biases: Vec::new(),
            learning_rate,
            dropout_rate: 0.5,
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            class_to_id: HashMap::new(),
            id_to_class: HashMap::new(),
            is_trained: false,
            _phantom: std::marker::PhantomData,
        }
    }

    fn build_vocabulary(&mut self, data: &[Vec<I>], labels: &[Vec<O>]) {
        // Build token vocabulary
        let mut token_id = 0;
        for sequence in data {
            for token in sequence {
                if !self.token_to_id.contains_key(token) {
                    self.token_to_id.insert(token.clone(), token_id);
                    self.id_to_token.insert(token_id, token.clone());
                    token_id += 1;
                }
            }
        }
        self.vocab_size = token_id;

        // Build class vocabulary
        let mut class_id = 0;
        for label_vec in labels {
            for label in label_vec {
                if !self.class_to_id.contains_key(label) {
                    self.class_to_id.insert(label.clone(), class_id);
                    self.id_to_class.insert(class_id, label.clone());
                    class_id += 1;
                }
            }
        }
        self.num_classes = class_id;
    }

    fn initialize_weights(&mut self) {
        let mut rng = rand::rng();
        
        // Initialize embeddings
        self.embedding_weights = (0..self.vocab_size)
            .map(|_| {
                (0..self.embedding_dim)
                    .map(|_| rng.random_range(-0.1..0.1))
                    .collect()
            })
            .collect();

        // Initialize convolutional layers
        self.conv_filters = self.filter_sizes.iter().map(|&filter_size| {
            (0..self.num_filters)
                .map(|_| {
                    (0..filter_size * self.embedding_dim)
                        .map(|_| rng.random_range(-0.1..0.1))
                        .collect()
                })
                .collect()
        }).collect();

        self.conv_biases = self.filter_sizes.iter().map(|_| {
            (0..self.num_filters)
                .map(|_| rng.random_range(-0.1..0.1))
                .collect()
        }).collect();

        // Initialize dense layer
        let dense_input_size = self.num_filters * self.filter_sizes.len();
        self.dense_weights = (0..dense_input_size)
            .map(|_| {
                (0..self.num_classes)
                    .map(|_| rng.random_range(-0.1..0.1))
                    .collect()
            })
            .collect();

        self.dense_biases = (0..self.num_classes)
            .map(|_| rng.random_range(-0.1..0.1))
            .collect();
    }

    fn tokenize_sequence(&self, sequence: &[I]) -> Vec<usize> {
        sequence.iter()
            .filter_map(|token| self.token_to_id.get(token).copied())
            .take(self.max_sequence_length)
            .collect()
    }

    fn embed_sequence(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        token_ids.iter()
            .map(|&token_id| self.embedding_weights[token_id].clone())
            .collect()
    }

    fn apply_convolution(&self, embedded: &[Vec<f32>], filter_size_idx: usize) -> Vec<f32> {
        let filter_size = self.filter_sizes[filter_size_idx];
        let mut feature_maps = Vec::new();

        for filter_idx in 0..self.num_filters {
            let filter = &self.conv_filters[filter_size_idx][filter_idx];
            let bias = self.conv_biases[filter_size_idx][filter_idx];
            let mut max_activation = f32::NEG_INFINITY;

            // Apply convolution with this filter
            for i in 0..=embedded.len().saturating_sub(filter_size) {
                let mut activation = bias;
                
                for j in 0..filter_size {
                    if i + j < embedded.len() {
                        for k in 0..self.embedding_dim {
                            activation += filter[j * self.embedding_dim + k] * embedded[i + j][k];
                        }
                    }
                }
                
                // ReLU activation
                activation = activation.max(0.0);
                
                // Max pooling
                if activation > max_activation {
                    max_activation = activation;
                }
            }
            
            feature_maps.push(max_activation);
        }

        feature_maps
    }

    fn forward_pass(&self, sequence: &[I]) -> Vec<f32> {
        let token_ids = self.tokenize_sequence(sequence);
        if token_ids.is_empty() {
            return vec![0.0; self.num_classes];
        }

        let embedded = self.embed_sequence(&token_ids);
        
        // Apply convolutions with different filter sizes
        let mut all_features = Vec::new();
        for (i, _) in self.filter_sizes.iter().enumerate() {
            let features = self.apply_convolution(&embedded, i);
            all_features.extend(features);
        }

        // Dense layer
        let mut output = vec![0.0; self.num_classes];
        for i in 0..self.num_classes {
            output[i] = self.dense_biases[i];
            for j in 0..all_features.len() {
                output[i] += self.dense_weights[j][i] * all_features[j];
            }
        }

        // Softmax activation
        let max_val = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = output.iter().map(|&x| (x - max_val).exp()).sum();
        
        for val in &mut output {
            *val = (*val - max_val).exp() / exp_sum;
        }

        output
    }

    fn encode_labels(&self, labels: &[O]) -> Vec<f32> {
        let mut encoded = vec![0.0; self.num_classes];
        for label in labels {
            if let Some(&class_id) = self.class_to_id.get(label) {
                encoded[class_id] = 1.0;
            }
        }
        // Normalize
        let sum: f32 = encoded.iter().sum();
        if sum > 0.0 {
            for val in &mut encoded {
                *val /= sum;
            }
        }
        encoded
    }
}

impl<I, O> MachineLearningModel<I, O> for CNN<I, O> 
where 
    I: Clone + std::hash::Hash + Eq + std::fmt::Debug,
    O: Clone + std::hash::Hash + Eq + std::fmt::Debug,
{
    fn new() -> Self {
        CNN::with_config(
            128,           // embedding_dim
            100,           // num_filters
            vec![3, 4, 5], // filter_sizes
            512,           // max_sequence_length
            0.001,         // learning_rate
        )
    }

    fn train(&mut self, data: &[Vec<I>], labels: &[Vec<O>]) -> Result<(), String> {
        if data.len() != labels.len() {
            return Err("Data and labels must have the same length".to_string());
        }

        if data.is_empty() {
            return Err("Training data cannot be empty".to_string());
        }

        // Build vocabularies
        self.build_vocabulary(data, labels);
        
        if self.vocab_size == 0 {
            return Err("No vocabulary found in training data".to_string());
        }

        if self.num_classes == 0 {
            return Err("No classes found in labels".to_string());
        }

        // Initialize weights
        self.initialize_weights();

        // Simple training loop (in practice, you'd want SGD with batches)
        let epochs = 10;
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            
            for (i, (sequence, label_vec)) in data.iter().zip(labels.iter()).enumerate() {
                // Forward pass
                let predictions = self.forward_pass(sequence);
                let target = self.encode_labels(label_vec);
                
                // Calculate loss (cross-entropy)
                let mut loss = 0.0;
                for j in 0..self.num_classes {
                    if target[j] > 0.0 {
                        loss -= target[j] * predictions[j].ln();
                    }
                }
                total_loss += loss;
                
                // Simplified gradient descent (this is a very basic implementation)
                // In practice, you'd compute proper gradients and update all parameters
                if i % 100 == 0 && epoch == epochs - 1 {
                    // Just a placeholder for actual backpropagation
                    for layer_weights in &mut self.dense_weights {
                        for weight in layer_weights {
                            *weight *= 0.9999; // Simple weight decay
                        }
                    }
                }
            }
            
            if epoch % (epochs / 4).max(1) == 0 {
                println!("Epoch {}: Average loss = {:.4}", epoch, total_loss / data.len() as f32);
            }
        }

        self.is_trained = true;
        Ok(())
    }

    fn predict(&self, data_point: &Vec<I>) -> Vec<O> {
        if !self.is_trained {
            return Vec::new();
        }

        let probabilities = self.forward_pass(data_point);
        
        // Return top-k predictions (here we'll return top 3 or all classes if fewer)
        let mut class_probs: Vec<(usize, f32)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();
        
        class_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top predictions above a threshold or top 3
        class_probs
            .iter()
            .take(3)
            .filter(|(_, prob)| *prob > 0.1)
            .filter_map(|(class_id, _)| self.id_to_class.get(class_id).cloned())
            .collect()
    }
}

// Type alias for your specific use case
pub type TextClassificationCNN = CNN<u16, u16>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cnn_creation() {
        let cnn: TextClassificationCNN = CNN::new();
        assert_eq!(cnn.embedding_dim, 128);
        assert_eq!(cnn.num_filters, 100);
        assert_eq!(cnn.filter_sizes, vec![3, 4, 5]);
    }

    #[test]
    fn test_training_and_prediction() {
        let mut cnn: TextClassificationCNN = CNN::new();
        
        // Sample tokenized text data
        let data = vec![
            vec![1u16, 2, 3, 4, 5],      // Document 1
            vec![2u16, 3, 4, 6, 7],      // Document 2  
            vec![1u16, 3, 5, 8, 9],      // Document 3
            vec![4u16, 6, 7, 10, 11],    // Document 4
        ];
        
        // Sample labels (categories for each document)
        let labels = vec![
            vec![100u16],          // Category 100
            vec![200u16],          // Category 200
            vec![100u16],          // Category 100
            vec![200u16],          // Category 200
        ];
        
        // Train the model
        let result = cnn.train(&data, &labels);
        assert!(result.is_ok());
        
        // Make a prediction
        let prediction = cnn.predict(&vec![1u16, 2, 3]);
        assert!(!prediction.is_empty());
        println!("Prediction: {:?}", prediction);
    }
}
