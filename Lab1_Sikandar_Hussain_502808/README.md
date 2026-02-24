Lab1 - NumPy Vectorization Performance Analysis
Name : Sikandar Hussain  
CMS ID : 502808  
Datev: January 27, 2026

Project Overview
This project demonstrates the significant performance difference between traditional Python loops and NumPy vectorized operations. It implements a comparative analysis by squaring values in a large dataset using both approaches and measuring their execution times.

Objectives
- Compare execution time between loop-based and vectorized operations
- Demonstrate the efficiency gains of NumPy vectorization
- Analyze performance on large datasets (5 million elements)
- Generate timing reports for performance analysis

Project Structure
```
Lab1_Sikandar_Hussain_502808/
├── main.py              # Main execution script
├── utils.py             # Utility functions for data generation and processing
├── requirements.txt     # Python package dependencies
├── README.md           # Project documentation
├── results/            # Output directory
│   └── timing.txt      # Performance timing results
├── ai_env/             # Virtual environment (not tracked in git)
└── __pycache__/        # Python cache files
```

Dependencies
The project requires the following Python packages:
numpy (2.4.1) - Numerical computing library for vectorized operations
pandas (3.0.0) - Data manipulation and analysis
matplotlib (3.10.8) - Plotting and visualization library

Installation & Setup

1. Create Virtual Environment
```powershell
# Navigate to project directory
cd "D:\Programming for AI\Lab1_Sikandar_Hussain_502808"

# Create virtual environment
python -m venv ai_env

2. Activate Virtual Environment
ai_env\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

# Ensure virtual environment is activated
python main.py
```

main.py
The main execution script that:
1. Generates a dataset of 5 million random numbers
2. Measures execution time for loop-based squaring
3. Measures execution time for vectorized squaring
4. Prints performance results to console
5. Saves results to `results/timing.txt`

Key Configuration:
- `DATA_SIZE = 5_000_000` - Size of the test dataset

utils.py
Contains utility functions:

`generate_data(size: int) -> np.ndarray`
Generates random dataset using NumPy's random module.
- **Parameters:** size - number of random values to generate
- **Returns:** NumPy array of random floats

`loop_square(data: np.ndarray) -> np.ndarray`
Squares each value using a traditional Python loop.
- **Parameters:** data - input NumPy array
- **Returns:** NumPy array with squared values
- **Time Complexity:** O(n) with Python-level iteration overhead

`vectorized_square(data: np.ndarray) -> np.ndarray`
Squares values using NumPy vectorization.
- **Parameters:** data - input NumPy array
- **Returns:** NumPy array with squared values
- **Time Complexity:** O(n) with C-level optimization

`measure_time(func, data: np.ndarray) -> float`
Measures execution time of a given function.
- **Parameters:** 
  - func - function to measure
  - data - input data for the function
- **Returns:** Execution time in seconds

Performance Results

### Latest Test Run (5 Million Elements)
- **Loop-based approach:** 1.8427 seconds
- **Vectorized approach:** 0.0165 seconds
- **Speed improvement:** ~111.6x faster

Performance Analysis
The vectorized approach is approximately **111 times faster** than the loop-based approach. This dramatic difference is due to:

1. **C-level optimization** - NumPy operations are implemented in C
2. **SIMD instructions** - Modern CPUs can process multiple data points simultaneously
3. **Reduced Python overhead** - No Python-level loop interpretation
4. **Memory locality** - Better cache utilization with contiguous memory


Best Practices Demonstrated:
- Use NumPy vectorization for numerical operations
- Avoid Python loops when working with large arrays
- Profile code to identify performance bottlenecks
- Measure and document performance improvements






Sikandar Hussain  
Student ID: 502808

---
*Last Updated: January 27, 2026*
