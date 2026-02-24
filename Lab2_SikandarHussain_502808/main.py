import numpy as np
import pandas as pd
import time
import os
from utils import standardize_loop, standardize_vectorized, polynomial_features, add_bias
from text_pipeline import build_vocabulary, text_to_ids, ids_to_bow, batch_text_to_bow, linear_text_scoring


def main():
    np.random.seed(42)
    os.makedirs("results", exist_ok=True)
    
    print("Lab 2")
    
    # task 1: numerical pipeline
    print("\n==>TASK 1: Numerical Pipeline ")
    
    df = pd.read_csv('data/dataset.csv')
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    X = df[feature_cols].values
    y = df['target'].values
    
    print(f"Dataset shape: {X.shape}")
    
    n_samples = X.shape[0]
    n_test = int(n_samples * 0.2)
    indices = np.random.permutation(n_samples)
    X_train, X_test = X[indices[n_test:]], X[indices[:n_test]]
    y_train, y_test = y[indices[n_test:]], y[indices[:n_test]]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # benchmark standardization
    print("\nBenchmarking standardization")
    
    loop_times = []
    vec_times = []
    for i in range(5):
        start = time.time()
        _ = standardize_loop(X_train)
        loop_times.append(time.time() - start)
        
        start = time.time()
        _ = standardize_vectorized(X_train)
        vec_times.append(time.time() - start)
    
    loop_avg = np.mean(loop_times)
    vec_avg = np.mean(vec_times)
    speedup = loop_avg / vec_avg
    
    print(f"Loop: {loop_avg:.6f}s, Vectorized: {vec_avg:.6f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # feature pipeline 
    X_train_std = standardize_vectorized(X_train)
    X_test_std = standardize_vectorized(X_test)
    
    X_train_poly = polynomial_features(X_train_std, degree=2)
    X_test_poly = polynomial_features(X_test_std, degree=2)
    
    X_train_final = add_bias(X_train_poly)
    X_test_final = add_bias(X_test_poly)
    
    print(f"Shape flow: {X_train.shape} -> {X_train_std.shape} -> {X_train_poly.shape} -> {X_train_final.shape}")
    
    # lirear regression
    W = np.linalg.lstsq(X_train_final, y_train, rcond=None)[0]
    y_pred = X_test_final @ W
    mse = np.mean((y_test - y_pred) ** 2)
    
    print(f"Test MSE: {mse:.6f}")
    
    # task 2: text scoring
    print("\n===> TASK 2: Text Scoring")
    
    texts = ["I love AI", "AI loves math", "I love math"]
    
    # vocab as per specification: only 4 words
    vocab = {"i": 0, "love": 1, "ai": 2, "math": 3}
    print(f"Vocabulary: {vocab}")
    print(f"Vocab size: {len(vocab)}")
    
    bow_matrix = batch_text_to_bow(texts, vocab)
    print(f"BoW matrix shape: {bow_matrix.shape}")
    
    # word importance weights as per specification
    weights = np.array([0.2, 0.5, 1.0, 0.8])
    scores = linear_text_scoring(bow_matrix, weights)
    
    print(f"Text scores: {scores}")
    
    # Save results
    with open('results/timing.txt', 'w') as f:
        f.write("TASK 1: Numerical Pipeline\n")
        f.write("\n")
        f.write(f"Loop-based standardization: {loop_avg:.6f}s\n")
        f.write(f"Vectorized standardization: {vec_avg:.6f}s\n")
        f.write(f"Speedup: {speedup:.2f}x\n\n")
        f.write(f"Shape flow: {X_train.shape} -> {X_train_final.shape}\n")
        f.write(f"Test MSE: {mse:.6f}\n")
    
    with open('results/text_scores.txt', 'w') as f:
        f.write("TASK 2: Text Scoring Results\n")
        f.write("\n")
        f.write(f"Vocabulary: {vocab}\n\n")
        for text, score in zip(texts, scores):
            f.write(f'"{text}": {score:.2f}\n')
    
    print("\n")
    print("Results saved to results")


if __name__ == "__main__":
    main()

