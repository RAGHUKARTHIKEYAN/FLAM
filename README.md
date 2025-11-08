# Parametric Curve Optimization

This project finds the optimal values of unknown parameters (θ, M, X) in a parametric curve equation by fitting it to given data points.

## Problem Statement

Given a parametric curve:
- **x** = t·cos(θ) - e^(M|t|) · sin(0.3t) · sin(θ) + X
- **y** = 42 + t·sin(θ) + e^(M|t|) · sin(0.3t) · cos(θ)

With constraints:
- 0° < θ < 50°
- -0.05 < M < 0.05
- 0 < X < 100
- 6 < t < 60

Find the values of θ, M, and X that minimize the L1 distance between the curve and the given data points.

## Approach

### Methodology

1. **Data Loading**: Load the provided data points from `xy_data.csv`

2. **Optimization Strategy**: 
   - Use a two-phase optimization approach:
     - **Phase 1**: Global optimization using Differential Evolution with uniform sampling (500 samples)
     - **Phase 2**: Local refinement using L-BFGS-B with increased sampling density (1000 samples)

3. **Distance Metric - Uniform Sampling Approach**:
   - Uniformly sample t values from [6, 60] (e.g., 500-1000 points)
   - For each sampled t, compute predicted (x, y) coordinates using current parameter values (θ, M, X)
   - For each predicted point, find the nearest data point using L1 (Manhattan) distance
   - Sum all minimum distances to get total L1 distance
   - This approach is efficient and works well when the curve is well-sampled

4. **Optimization Algorithm**:
   - **Differential Evolution**: Global search algorithm that explores the entire parameter space to avoid local minima
     - Population size: 20
     - Maximum iterations: 200
     - Uses polynomial mutation and binomial crossover
   - **L-BFGS-B**: Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm for local refinement
     - Bounded optimization to respect parameter constraints
     - Maximum iterations: 300
     - Fine-tunes the solution from Phase 1

5. **Parameter Constraints**:
   - θ: 0° < θ < 50° (converted to radians for computation)
   - M: -0.05 < M < 0.05
   - X: 0 < X < 100
   - t: 6 < t < 60 (parameter range, not optimized)

### Key Steps

1. **Define Parametric Curve Function**: Implement the mathematical equations for x(t) and y(t)
2. **Implement L1 Distance Computation**: Create function to compute total L1 distance using uniform sampling
3. **Set Up Optimization**: Define parameter bounds and optimization algorithms
4. **Run Two-Phase Optimization**: 
   - Phase 1: Global search with Differential Evolution
   - Phase 2: Local refinement with L-BFGS-B
5. **Validate Results**: Check that parameters are within bounds and L1 distance is minimized
6. **Generate Output**: Create visualization and LaTeX equation for Desmos

### Why This Approach Works

- **Uniform Sampling**: By sampling t uniformly, we ensure good coverage of the curve and can efficiently compute distances
- **Nearest Neighbor Matching**: For each predicted point, finding the nearest data point gives us a robust distance measure
- **Two-Phase Optimization**: Combining global search (Differential Evolution) with local refinement (L-BFGS-B) ensures we find a good solution efficiently
- **L1 Distance**: Manhattan distance is robust to outliers and directly matches the assessment criteria

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the optimization script:
```bash
python optimize_curve.py
```

This will:
- Load the data points
- Perform optimization
- Display results
- Generate a visualization (`curve_fit.png`)
- Save results to `results.txt`

## Results

After running the optimization, the script will output:
- Optimal values of θ, M, and X
- Final L1 distance
- LaTeX equation format for Desmos calculator
- Visualization comparing data points and fitted curve

## Output Format

The results include a LaTeX equation that can be directly used in Desmos:
```
\left(t*\cos(θ)-e^{M\left|t\right|}\cdot\sin(0.3t)\sin(θ)\ +X,42+\ t*\sin(θ)+e^{M\left|t\right|}\cdot\sin(0.3t)\cos(θ)\right)

```
Where θ, M, and X are the optimized values.

### Output from Desmos:

https://www.desmos.com/calculator/ntomuokcm2

## Files

- `optimize_curve.py`: Main optimization script
- `xy_data.csv`: Input data points
- `requirements.txt`: Python dependencies
- `README.md`: This file
- `results.txt`: Output results (generated after run)
- `curve_fit.png`: Visualization (generated after run)

## Notes

- The optimization may take several minutes to complete
- The algorithm uses uniform sampling for efficiency while maintaining accuracy
- Results are validated against the constraint bounds

