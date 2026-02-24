import numpy as np
import time
import tracemalloc


def generate_data(size: int) -> np.ndarray:
    """Generate random dataset"""
    return np.random.rand(size)


def loop_square(data: np.ndarray) -> np.ndarray:
    """Square values using Python loop"""
    result = []
    for i in range(len(data)):
        result.append(data[i] ** 2)
    return np.array(result)


def vectorized_square(data: np.ndarray) -> np.ndarray:
    """Square values using NumPy vectorization"""
    return data ** 2


def measure_time_and_memory(func, data: np.ndarray):
    """
    Measure execution time and peak memory usage
    Returns: (time_in_seconds, memory_in_MB)
    """
    tracemalloc.start()

    start_time = time.time()
    func(data)
    end_time = time.time()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_mb = peak / (1024 * 1024)
    time_taken = end_time - start_time

    return time_taken, memory_mb
