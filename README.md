# Medical Image Contrast Enhancement using Genetic Algorithms

This repository contains an implementation of genetic algorithms (GAs) for improving contrast in medical images through sigmoid transformation. The project explores two optimization models: one that maximizes entropy and another that maximizes standard deviation as objective functions.

## Project Overview

Medical image enhancement is crucial for accurate diagnosis and analysis. This project implements a GA-based approach to optimize sigmoid transformation parameters (alpha and delta) that enhance image contrast. The implementation supports:

- Two objective functions for comparison: entropy maximization and standard deviation maximization
- Simulated Binary Crossover (SBX) with boundary handling 
- Polynomial mutation with configurable distribution parameters
- Tournament selection for parent selection
- Elitism to preserve the best solution in each generation
- Statistical analysis of multiple runs with comprehensive reporting

## Theoretical Background

### Sigmoid Transformation

The sigmoid function is applied to transform pixel values and enhance image contrast:

```
sigmoid(x) = 1 / (1 + exp(-alpha * (x - delta)))
```

Where:
- `alpha` controls the slope of the transformation (contrast strength)
- `delta` controls the midpoint shift (brightness adjustment)

### Genetic Algorithm Components

1. **Chromosome Representation**: Each individual is represented by two values [alpha, delta]
2. **Population Initialization**: Random uniform initialization within parameter bounds
3. **Selection**: Tournament selection to choose parents for reproduction
4. **Crossover**: Simulated Binary Crossover (SBX) with boundary handling
5. **Mutation**: Polynomial mutation with configurable distribution parameters
6. **Elitism**: The best individual from each generation is preserved
7. **Fitness Functions**:
   - Entropy maximization: Measures information content in the transformed image
   - Standard deviation maximization: Measures overall contrast in the transformed image

## Repository Structure

```
├── AG.py                      # Main genetic algorithm implementation
├── AG_confs.py                # Configuration parameters for the algorithm
├── main.py                    # Main execution script
├── libs/
│   ├── auxiliaries_functions.py  # Helper functions for GA operations
│   ├── crossover.py           # SBX crossover implementation
│   ├── functions.py           # Image transformation and fitness functions
│   ├── mutation.py            # Polynomial mutation implementation
│   ├── plot.py                # Visualization functions
│   └── selection.py           # Tournament selection implementation
├── outputs/                   # Generated output directory (created at runtime)
│   ├── Entropy/               # Results using entropy as fitness
│   │   ├── historiales/       # History of each run
│   │   └── resumenes/         # Summary statistics
│   └── Std._Deviation/        # Results using standard deviation as fitness
│       ├── historiales/       # History of each run
│       └── resumenes/         # Summary statistics
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- SciPy

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/medical-image-contrast-enhancement.git
cd medical-image-contrast-enhancement
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib pillow scipy
```

## Usage

### Basic Usage

Run the main script to execute the genetic algorithm:

```bash
python main.py
```

### Configuration

Modify `AG_confs.py` to adjust algorithm parameters:

```python
# Population and generation settings
POP_SIZE = 100            # Number of individuals in population
NUM_GENERATIONS = 50      # Number of generations
NUM_RUNS = 5              # Number of complete execution cycles

# Tournament selection settings
TOURNAMENT_SIZE = 3       # Number of individuals in each tournament

# SBX crossover settings
CROSSOVER_PROB = 0.9      # Probability of applying crossover
ETA_C = 0.5               # Distribution index for SBX

# Parameter bounds
LB = np.array([0.0, 0.0]) # Lower bounds [alpha, delta]
UB = np.array([10.0, 1.0]) # Upper bounds [alpha, delta]

# Polynomial mutation settings
MUTATION_PROB = 10.0 / 2  # Probability of mutating each gene
ETA_MUT = 0.5             # Distribution index for polynomial mutation

# Image path
IMG_PATH = 'kodim23.png'  # Path to the image to enhance
```

### Running with Custom Images

To use your own images, place them in the project directory and update the `IMG_PATH` variable in `AG_confs.py`.

## Results and Analysis

The algorithm outputs several files to analyze results:

1. **CSV files**:
   - Individual run histories (`historiales/historial_run_X.csv`)
   - Run summaries with best, average, worst solutions (`resumenes/resumen_run_X.csv`)
   - Global summary across all runs (`resumenes/resumen_global_corridas.csv`)

2. **Visualizations**:
   - Original vs. enhanced images
   - Fitness evolution graphs
   - Surface plots of the objective function

### Sample Output

After execution, the program displays:
- Progress information for each run
- Summary tables with statistics
- Visual comparison between original and enhanced images

## Experimental Analysis and Interpretation

When analyzing results, consider:

1. **Convergence speed**: How quickly does each objective function reach optimal values?
2. **Solution stability**: Do multiple runs converge to similar solutions?
3. **Visual quality**: Which objective function produces better perceptual contrast?
4. **Parameter sensitivity**: How do changes in GA parameters affect results?

## Contributing

Contributions to improve the algorithm or add features are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## Acknowledgments

This project was developed as part of a practical assignment for biomedical image processing. Special thanks to the original authors of the SBX crossover and polynomial mutation operators.