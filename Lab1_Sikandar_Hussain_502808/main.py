from utils import (
    generate_data,
    loop_square,
    vectorized_square,
    measure_time_and_memory
)

DATA_SIZE = 2_000_000  # Use 5_000_000 if system allows


def main():
    data = generate_data(DATA_SIZE)

    loop_time, loop_memory = measure_time_and_memory(loop_square, data)
    vector_time, vector_memory = measure_time_and_memory(vectorized_square, data)

    print(f"Loop-based time: {loop_time:.4f} seconds")
    print(f"Loop-based memory usage: {loop_memory:.2f} MB")

    print(f"Vectorized time: {vector_time:.4f} seconds")
    print(f"Vectorized memory usage: {vector_memory:.2f} MB")

    with open("results/timing.txt", "w") as file:
        file.write(f"Dataset size: {DATA_SIZE}\n\n")

        file.write("Loop-based computation:\n")
        file.write(f"Time: {loop_time:.4f} seconds\n")
        file.write(f"Memory usage: {loop_memory:.2f} MB\n\n")

        file.write("Vectorized computation:\n")
        file.write(f"Time: {vector_time:.4f} seconds\n")
        file.write(f"Memory usage: {vector_memory:.2f} MB\n")


if __name__ == "__main__":
    main()
