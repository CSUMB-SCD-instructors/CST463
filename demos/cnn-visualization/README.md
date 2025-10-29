# CNN Explainability Demo

Interactive demonstration of three key techniques for understanding what Convolutional Neural Networks learn and how they make predictions.

## Overview

This demo uses a simple 2-layer CNN trained on MNIST or Fashion-MNIST to visualize:

1. **Filter Visualizations** - What patterns individual convolutional filters detect
2. **Activation Maximization** - Synthesize images that maximally activate specific filters
3. **Grad-CAM** - Highlight which input regions drive predictions

## Structure

```
demos/cnn-visualization/
├── cnn_explainability_demo.ipynb    # Main interactive notebook
├── saved_models/                    # Pre-trained model weights
│   ├── mnist_simple_cnn.h5         # (to be generated)
│   └── fashion_mnist_simple_cnn.h5 # (to be generated)
└── README.md                        # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow matplotlib numpy jupyter
```

### 2. Launch Notebook

```bash
cd demos/cnn-visualization
jupyter notebook cnn_explainability_demo.ipynb
```

### 3. Choose Dataset

In the first configuration cell, set:

```python
DATASET = 'mnist'  # or 'fashion_mnist'
```

### 4. Train or Load Model

- **First run**: Set `TRAIN_NEW_MODEL = True` to train a new model (5 epochs, ~2-3 minutes on GPU)
- **Subsequent runs**: Set `TRAIN_NEW_MODEL = False` to load the saved model

## Model Architecture

Intentionally simple for interpretability:

```
Input (1x28x28)
  ↓
Conv2d(1→8, kernel=3x3) + ReLU + MaxPool(2x2) → (8x14x14)
  ↓
Conv2d(8→8, kernel=3x3) + ReLU + MaxPool(2x2) → (8x7x7)
  ↓
Flatten + Linear(8*7*7 → 10)
  ↓
Output (10 classes)
```

**Why so simple?**
- Only 16 total filters makes individual filter analysis tractable
- Fast training (minutes, not hours)
- Runs on CPU if needed
- Still achieves ~98% accuracy on MNIST

## Visualization Techniques

### Method 1: Filter Visualizations

**Shows:** What patterns each convolutional filter learns

**Output:**
- Raw filter weights (3x3 matrices with float values)
- Heatmap visualizations (red = positive weights, blue = negative)
- Feature maps showing filter activations on real images

**What to look for:**
- Conv1 filters often detect edges, corners, and simple textures
- Conv2 filters detect more complex shape combinations
- Different filters activate on different image regions

### Method 2: Activation Maximization

**Shows:** Synthesized images that maximally activate specific filters

**How it works:**
1. Start with random noise
2. Use gradient ascent to modify the input
3. Maximize the mean activation of a target filter
4. Result: an image showing what that filter "wants to see"

**What to look for:**
- Conv1 patterns are usually simple (oriented edges)
- Conv2 patterns can be more abstract
- Some filters may not converge to clear patterns

### Method 3: Grad-CAM (Class Activation Mapping)

**Shows:** Which image regions are important for predictions

**How it works:**
1. Compute gradients of predicted class w.r.t. feature maps
2. Weight feature maps by their importance
3. Create heatmap showing influential regions

**What to look for:**
- Does the model focus on relevant features?
- For misclassifications, what is the model looking at?
- How do heatmaps differ for different target classes?

## Classroom Usage Tips

### During Live Demo

1. **Start with MNIST** - digit patterns are more intuitive
2. **Show filter evolution** - compare Conv1 (simple edges) to Conv2 (complex patterns)
3. **Pick interesting examples** - find cases where Grad-CAM reveals insights
4. **Switch to Fashion-MNIST** - show how filters adapt to different data

### For Students to Explore

Encourage students to:
- Change `sample_idx` to visualize different test images
- Modify `filter_idx` in activation maximization
- Compare Grad-CAM for correct vs incorrect predictions
- Train with different random seeds and compare learned filters

### Discussion Questions

1. Why do Conv1 filters look similar across different datasets?
2. What happens to Grad-CAM when the model is wrong?
3. Can you identify "dead" filters that don't learn useful patterns?
4. How would adding more layers change the visualization?

## Technical Details

### Dataset Options

**MNIST:**
- Handwritten digits (0-9)
- 60,000 training images
- Simple, clean patterns
- Fast training

**Fashion-MNIST:**
- Clothing items (10 categories)
- Same size/format as MNIST
- More complex textures
- Slightly harder classification

### Training Parameters

Default settings (adjustable in notebook):
```python
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
```

Expected performance:
- MNIST: ~98% test accuracy
- Fashion-MNIST: ~85-90% test accuracy

### Hardware Requirements

- **CPU**: Works fine, training takes ~5-10 minutes
- **GPU**: Recommended for live demos, training takes ~1-2 minutes
- **Memory**: ~2GB RAM sufficient

## Troubleshooting

**No pre-trained model found:**
- Set `TRAIN_NEW_MODEL = True` and run training cells

**Activation maximization produces unclear images:**
- Try adjusting `lr` (learning rate) and `iterations`
- Some filters may not converge to clear patterns (this is normal)

**Grad-CAM heatmaps look uniform:**
- Check model is actually trained (accuracy should be >90%)
- Some predictions may genuinely use the whole image

**Dataset download fails:**
- Check internet connection
- Datasets will be saved to `./data/` directory

**"AttributeError: The layer sequential has never been called and thus has no defined input":**
- This error occurs when trying to access `model.input` on a Sequential model that hasn't properly established its input graph
- **The notebook already handles this** - all visualization functions create explicit `Input` tensors rather than using `model.input`
- If you see this error, you may be using an older version of the notebook - make sure to run all cells from the start

## Extensions and Modifications

### Easy Modifications

1. **Change architecture:**
   ```python
   # In create_simple_cnn() function
   layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')  # 16 filters instead of 8
   ```

2. **Add more layers:**
   ```python
   # Add a third convolutional layer
   layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv3'),
   layers.MaxPooling2D((2, 2), name='pool3'),
   ```

3. **Different activation maximization targets:**
   ```python
   # Maximize final layer neurons instead of filters
   # Use model.output instead of intermediate layer
   class_score = predictions[:, target_class]
   ```

### Advanced Extensions

- **Adversarial examples**: Modify Grad-CAM to fool the classifier
- **Style transfer**: Use activation maximization with style loss
- **Feature inversion**: Reconstruct input from intermediate activations
- **Guided backprop**: Alternative to Grad-CAM with sharper visualizations

## References

- **Grad-CAM paper**: Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **Activation Maximization**: Erhan et al. (2009) - "Visualizing Higher-Layer Features of a Deep Network"
- **Understanding CNNs**: Zeiler & Fergus (2014) - "Visualizing and Understanding Convolutional Networks"

## License

This demo is part of CST463 course materials. Free to use for educational purposes.
