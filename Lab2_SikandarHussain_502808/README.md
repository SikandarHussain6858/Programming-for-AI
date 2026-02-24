# Lab2: Programming for AI

**Student:** Sikandar Hussain (502808)

## Overview

This lab implements two core tasks using NumPy arrays and matrix operations:

**Task 1:** Numerical pipeline for feature transformation and linear regression  
**Task 2:** Text vectorization and scoring using Bag-of-Words

## Running the Code

```bash
python main.py
```

Results saved to `results/timing.txt` and `results/text_scores.txt`

## Task 1: Numerical Pipeline

**Shape Flow:**
```
Raw Data (10000, 3)
  ↓ standardize
(10000, 3)
  ↓ polynomial_features(degree=2)
(10000, 9)
  ↓ add_bias
(10000, 10)
  ↓ linear regression
predictions
```

**Components:**
- Loop-based standardization (baseline)
- Vectorized standardization (~40x faster)
- Polynomial feature expansion
- Bias column injection
- Batch linear forward pass

## Task 2: Text Scoring

**Shape Flow:**
```
Texts: ["I love AI", "AI loves math", "I love math"]
  ↓ build_vocabulary
vocab = {"i": 0, "love": 1, "ai": 2, "math": 3}
  ↓ batch_text_to_bow
BoW matrix (3, 4)
  ↓ linear_text_scoring @ weights
scores (3,)
```

**Components:**
- Vocabulary construction from texts
- Text to token ID conversion
- Token IDs to Bag-of-Words vectors
- Batch vectorization
- Linear scoring via matrix multiplication

## Performance

**Vectorization Speedup:** ~40x (loop: 0.030s, vectorized: 0.0007s)

Vectorized operations using NumPy are significantly faster than Python loops for array operations.

## Reproducibility

- Random seed: `np.random.seed(42)`
- Dataset: `data/dataset.csv` (10000 samples, 3 features)
- All scripts are deterministic

## Files

- `main.py` - Runs both tasks end-to-end
- `utils.py` - Task 1 numerical functions
- `text_pipeline.py` - Task 2 text processing
- `data/dataset.csv` - Input data
- `results/timing.txt` - Performance metrics
- `results/text_scores.txt` - Text scoring output
