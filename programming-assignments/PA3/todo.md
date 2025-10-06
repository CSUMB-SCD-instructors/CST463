# PA3 Development Todo

## Assignment Overview
PA3: Backpropagation & Neural Networks - focusing on understanding gradient flow and chain rule application through simple neural networks.

## Development Phase Plan

### Phase 1: Function Skeletons & Architecture (student_code.py)

#### Core Mathematical Functions
- [x] `linear_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray`
  - Basic linear transformation: z = X @ W + b
- [x] `sigmoid_forward(z: np.ndarray) -> np.ndarray`
  - Sigmoid activation function: σ(z) = 1 / (1 + exp(-z))
- [x] `relu_forward(z: np.ndarray) -> np.ndarray`
  - ReLU activation function: max(0, z)
- [x] `softmax_forward(z: np.ndarray) -> np.ndarray`
  - Softmax for multi-class: exp(z_i) / sum(exp(z_j))

#### Loss Functions & Derivatives
- [x] `mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float`
  - Mean squared error for regression
- [x] `cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float`
  - Cross-entropy for classification
- [x] `mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray`
  - Derivative of MSE w.r.t. predictions
- [x] `sigmoid_derivative(z: np.ndarray) -> np.ndarray`
  - Derivative of sigmoid: σ(z) * (1 - σ(z))
- [x] `relu_derivative(z: np.ndarray) -> np.ndarray`
  - Derivative of ReLU: 1 if z > 0, else 0

#### Backpropagation Core Functions
- [x] `linear_backward(dL_dz: np.ndarray, X: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
  - Compute gradients through linear layer: dL_dW, dL_db, dL_dX
- [x] `single_layer_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str) -> Tuple[np.ndarray, np.ndarray]`
  - Forward pass through single layer with activation
  - Return both pre-activation (z) and post-activation (a) for backprop
- [x] `single_layer_backward(dL_da: np.ndarray, z: np.ndarray, X: np.ndarray, W: np.ndarray, activation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
  - Backward pass through single layer with activation

#### Two-Layer Network Functions
- [x] `two_layer_forward(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, hidden_activation: str, output_activation: str) -> Dict`
  - Full forward pass through 2-layer network
  - Return dict with all intermediate values for backprop
- [x] `two_layer_backward(forward_cache: Dict, y_true: np.ndarray, loss_type: str) -> Dict`
  - Full backward pass through 2-layer network
  - Return dict with all gradients

#### Training & Optimization
- [x] `initialize_network_weights(layer_sizes: List[int], method: str = 'xavier') -> List[Tuple[np.ndarray, np.ndarray]]`
  - Initialize weights and biases for multi-layer network
  - Support 'zeros', 'random', 'xavier', 'he' methods
- [x] `gradient_descent_step_network(gradients: Dict, weights: Dict, learning_rate: float) -> Dict`
  - Update network weights using computed gradients
- [x] `train_single_layer(X: np.ndarray, y: np.ndarray, activation: str, loss_type: str, epochs: int, learning_rate: float) -> Tuple[Dict, List[float]]`
  - Train single layer network (logistic regression)
- [x] `train_two_layer_network(X: np.ndarray, y: np.ndarray, hidden_size: int, epochs: int, learning_rate: float) -> Tuple[Dict, List[float]]`
  - Train two-layer network end-to-end

#### Utility & Verification Functions
- [x] `numerical_gradient_check(X: np.ndarray, y: np.ndarray, weights: Dict, network_type: str, epsilon: float = 1e-7) -> Dict`
  - Gradient checking to verify analytical gradients
- [x] `predict_network(X: np.ndarray, weights: Dict, network_type: str) -> np.ndarray`
  - Make predictions with trained network
- [x] `generate_classification_data(n_samples: int, n_features: int, n_classes: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]`
  - Generate synthetic classification datasets
- [x] `generate_nonlinear_data(n_samples: int, noise: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]`
  - Generate synthetic data requiring nonlinear decision boundary

### Phase 2: Unit Tests (tests.py)

#### Forward Pass Tests
- [x] Test `linear_forward` with known inputs/outputs
- [x] Test activation functions (`sigmoid_forward`, `relu_forward`, `softmax_forward`)
- [x] Test loss functions with expected values
- [x] Test `single_layer_forward` end-to-end
- [x] Test `two_layer_forward` with known propagation

#### Derivative Tests
- [x] Test activation derivatives against numerical derivatives
- [x] Test loss derivatives with simple cases
- [x] Verify derivative shapes match forward pass shapes

#### Backpropagation Tests
- [x] Test `linear_backward` with known gradients
- [x] Test `single_layer_backward` chain rule application
- [x] Test `two_layer_backward` full gradient computation
- [x] Gradient checking tests (analytical vs numerical)

#### Integration Tests
- [x] Test complete training loop doesn't crash
- [x] Test overfitting on tiny dataset (should get perfect fit)
- [x] Test that loss decreases over training
- [x] Test prediction function consistency

#### Edge Case Tests
- [x] Test with single sample
- [x] Test with various batch sizes
- [x] Test numerical stability (large inputs, gradients)
- [x] Test different network architectures

### Phase 3: Golden Solution Implementation

#### Implementation Order
- [ ] Implement forward pass functions (linear, activations, loss)
- [ ] Implement derivative functions with proper vectorization
- [ ] Implement backward pass functions (start with linear_backward)
- [ ] Implement single layer network (easier debugging)
- [ ] Implement two-layer network (build on single layer)
- [ ] Implement gradient checking for verification
- [ ] Implement training loops with proper weight updates
- [ ] Add data generation and utility functions

#### Verification Steps
- [ ] Run gradient checking on all implemented functions
- [ ] Verify training reduces loss on synthetic data
- [ ] Test overfitting capability (memorize small dataset)
- [ ] Compare single layer vs two layer on nonlinear data
- [ ] Ensure all tests pass

### Phase 4: Visualization Notebook (visualization_analysis.ipynb)

#### Core Analysis Components
- [ ] **Gradient Flow Visualization**
  - Plot gradient magnitudes during training
  - Show how gradients change across layers
  - Demonstrate vanishing gradient effect in deeper networks
- [ ] **Decision Boundary Evolution**
  - Animate decision boundary changes during training
  - Compare linear vs nonlinear boundaries
  - Show effect of hidden layer size
- [ ] **Gradient Checking Demonstration**
  - Visual comparison of analytical vs numerical gradients
  - Show when gradient checking fails and why
- [ ] **Network Internal Analysis**
  - Visualize hidden layer activations
  - Show feature transformations through layers
  - Interpret what hidden units learn

#### Comparative Studies
- [ ] **Activation Function Comparison**
  - Compare sigmoid, ReLU, tanh on same problem
  - Analyze gradient flow differences
  - Show dead neuron problem with ReLU
- [ ] **Network Architecture Effects**
  - Compare single layer vs two layer performance
  - Show underfitting vs overfitting patterns
  - Analyze when nonlinearity helps vs hurts
- [ ] **Initialization Method Analysis**
  - Compare different weight initialization strategies
  - Show impact on convergence speed and stability
  - Demonstrate symmetry breaking

#### Understanding-Focused Experiments
- [ ] **Chain Rule Walkthrough**
  - Step-by-step gradient computation for single example
  - Visual representation of gradient flow
  - Connect mathematical formulation to implementation
- [ ] **Loss Landscape Exploration**
  - 2D visualization of loss surface
  - Show optimization path for different learning rates
  - Demonstrate local minima and saddle points

### Phase 5: Supporting Files

#### Assignment Documentation
- [ ] `README.md` - Assignment instructions and learning objectives
- [ ] `scoring.yaml` - Grading configuration and rubric
- [ ] `peer_review_rubric.md` - Peer evaluation guidelines
- [ ] `canvas_rubric.csv` - Canvas-compatible grading rubric

#### Sample Data Files
- [ ] Generate and save sample datasets for consistent testing
- [ ] Include 2D classification problems for visualization
- [ ] Create regression datasets for MSE demonstrations

## Design Principles for PA3

### Progressive Difficulty
- Remove more hints compared to PA2
- Expect students to derive chain rule applications
- Minimal function docstrings - focus on type hints
- Students must reason through gradient shapes and computations

### Pedagogical Focus
- **Deep Understanding**: Emphasize gradient flow intuition over complex architectures
- **Visual Learning**: Heavy use of animations and interactive plots
- **Mathematical Foundation**: Connect implementation to mathematical derivation
- **Practical Insights**: When/why to use different activation functions and architectures

### Scope Management (Exam Period Considerations)
- Maximum 2 hidden layers to keep complexity manageable
- Focus on 2D problems for clear visualization
- Shorter training runs for faster experimentation
- Synthetic datasets for clean, interpretable results

### Building on Previous PAs
- Use gradient descent implementation from PA2
- Apply matrix operations understanding from PA1
- Connect numerical gradient concepts from PA2 to gradient checking
- Leverage visualization skills developed in PA1 and PA2

## Quality Assurance Checklist

### Before Student Release
- [ ] All tests pass with golden solution
- [ ] Gradient checking validates all backprop implementations
- [ ] Notebook runs without errors and produces meaningful visualizations
- [ ] Assignment timing is appropriate for exam period (target: 8-12 hours total)
- [ ] Peer review components are clear and focused
- [ ] README instructions are comprehensive but concise