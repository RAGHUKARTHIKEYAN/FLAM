"""
Parametric Curve Optimization
Finds optimal values of θ, M, and X for the parametric equation:
    x = t*cos(θ) - e^(M|t|) * sin(0.3t) * sin(θ) + X
    y = 42 + t*sin(θ) + e^(M|t|) * sin(0.3t) * cos(θ)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def parametric_curve(t, theta, M, X):
    """
    Compute parametric curve coordinates for given parameter values.
    
    Parameters:
    -----------
    t : array-like
        Parameter values (6 <= t <= 60)
    theta : float
        Angle in radians
    M : float
        Exponential coefficient
    X : float
        X offset
    
    Returns:
    --------
    x, y : arrays
        Coordinates of the curve
    """
    t = np.array(t)
    abs_t = np.abs(t)
    
    # Common terms
    exp_term = np.exp(M * abs_t)
    sin_03t = np.sin(0.3 * t)
    
    # X coordinate
    x = t * np.cos(theta) - exp_term * sin_03t * np.sin(theta) + X
    
    # Y coordinate
    y = 42 + t * np.sin(theta) + exp_term * sin_03t * np.cos(theta)
    
    return x, y


def compute_l1_distance_uniform_sampling(theta, M, X, data_points, n_samples=1000):
    """
    Compute L1 distance by uniformly sampling t values and matching to nearest data points.
    
    Parameters:
    -----------
    theta : float
        Angle in radians
    M : float
        Exponential coefficient
    X : float
        X offset
    data_points : array
        Array of shape (n_points, 2) with (x, y) coordinates
    n_samples : int
        Number of uniform samples of t
    
    Returns:
    --------
    total_l1_distance : float
        Sum of L1 distances from sampled curve points to nearest data points
    """
    # Uniformly sample t values
    t_samples = np.linspace(6, 60, n_samples)
    
    # Compute predicted curve points
    x_pred, y_pred = parametric_curve(t_samples, theta, M, X)
    pred_points = np.column_stack([x_pred, y_pred])
    
    # Compute distances from each predicted point to all data points
    distances = cdist(pred_points, data_points, metric='cityblock')  # L1 distance
    
    # Find minimum distance for each predicted point
    min_distances = np.min(distances, axis=1)
    
    # Sum of L1 distances
    total_l1_distance = np.sum(min_distances)
    
    return total_l1_distance


def compute_l1_distance_data_driven(theta, M, X, data_points):
    """
    Compute L1 distance by finding best t for each data point.
    
    Parameters:
    -----------
    theta : float
        Angle in radians
    M : float
        Exponential coefficient
    X : float
        X offset
    data_points : array
        Array of shape (n_points, 2) with (x, y) coordinates
    
    Returns:
    --------
    total_l1_distance : float
        Sum of L1 distances from data points to curve
    """
    total_distance = 0.0
    
    # For each data point, find the t that minimizes distance
    for x_data, y_data in data_points:
        # Objective function: minimize L1 distance
        def objective(t):
            if t < 6 or t > 60:
                return 1e10  # Penalty for out of range
            x_pred, y_pred = parametric_curve(t, theta, M, X)
            return abs(x_pred - x_data) + abs(y_pred - y_data)
        
        # Find best t using optimization
        result = minimize(objective, x0=33.0, bounds=[(6, 60)], method='L-BFGS-B')
        total_distance += result.fun
    
    return total_distance


def objective_function_uniform(params, data_points, n_samples=1000):
    """
    Objective function for optimization using uniform sampling.
    
    Parameters:
    -----------
    params : array
        [theta, M, X]
    data_points : array
        Array of shape (n_points, 2) with (x, y) coordinates
    n_samples : int
        Number of uniform samples
    
    Returns:
    --------
    total_l1_distance : float
        Total L1 distance
    """
    theta, M, X = params
    
    # Check bounds
    theta_deg = np.degrees(theta)
    if theta_deg <= 0 or theta_deg >= 50:
        return 1e10
    if M <= -0.05 or M >= 0.05:
        return 1e10
    if X <= 0 or X >= 100:
        return 1e10
    
    return compute_l1_distance_uniform_sampling(theta, M, X, data_points, n_samples)


def objective_function_data_driven(params, data_points):
    """
    Objective function for optimization using data-driven approach.
    This is more accurate but slower.
    """
    theta, M, X = params
    
    # Check bounds
    theta_deg = np.degrees(theta)
    if theta_deg <= 0 or theta_deg >= 50:
        return 1e10
    if M <= -0.05 or M >= 0.05:
        return 1e10
    if X <= 0 or X >= 100:
        return 1e10
    
    return compute_l1_distance_data_driven(theta, M, X, data_points)


def main():
    """Main optimization routine with improved multi-strategy approach."""
    print("Loading data...")
    # Load data points
    df = pd.read_csv('xy_data.csv')
    data_points = df[['x', 'y']].values
    print(f"Loaded {len(data_points)} data points")
    
    # Parameter bounds
    # theta: 0° < theta < 50° (in radians)
    theta_min = np.radians(0.01)  # Slightly above 0
    theta_max = np.radians(49.99)  # Slightly below 50
    
    # M: -0.05 < M < 0.05
    M_min = -0.049
    M_max = 0.049
    
    # X: 0 < X < 100
    X_min = 0.01
    X_max = 99.99
    
    bounds = [
        (theta_min, theta_max),
        (M_min, M_max),
        (X_min, X_max)
    ]
    
    print("\nStarting optimization...")
    print("This may take several minutes...")
    
    # Use differential evolution for global optimization
    print("\nPhase 1: Global optimization with Differential Evolution...")
    result_phase1 = differential_evolution(
        objective_function_uniform,
        bounds,
        args=(data_points, 1000),  # Use 1000 samples for accuracy
        seed=42,
        maxiter=300,
        popsize=30,  # Larger population for better exploration
        atol=1e-8,
        tol=1e-8,
        mutation=(0.5, 1.5),  # Wider mutation range
        recombination=0.9,
        workers=1,
        updating='immediate',
        polish=True
    )
    
    print(f"Phase 1 completed. Best parameters:")
    print(f"  Theta = {np.degrees(result_phase1.x[0]):.6f} degrees ({result_phase1.x[0]:.6f} radians)")
    print(f"  M = {result_phase1.x[1]:.6f}")
    print(f"  X = {result_phase1.x[2]:.6f}")
    print(f"  L1 Distance = {result_phase1.fun:.6f}")
    
    # Refine with local optimization
    print("\nPhase 2: Local refinement with L-BFGS-B...")
    result_phase2 = minimize(
        objective_function_uniform,
        result_phase1.x,
        args=(data_points, 1000),  # Use 1000 samples
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10}
    )
    
    print(f"Phase 2 completed. Best parameters:")
    print(f"  Theta = {np.degrees(result_phase2.x[0]):.6f} degrees ({result_phase2.x[0]:.6f} radians)")
    print(f"  M = {result_phase2.x[1]:.6f}")
    print(f"  X = {result_phase2.x[2]:.6f}")
    print(f"  L1 Distance = {result_phase2.fun:.6f}")
    
    # Final parameters
    theta_opt = result_phase2.x[0]
    M_opt = result_phase2.x[1]
    X_opt = result_phase2.x[2]
    final_distance = result_phase2.fun
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Theta: {theta_opt:.6f} radians = {np.degrees(theta_opt):.6f} degrees")
    print(f"M: {M_opt:.6f}")
    print(f"X: {X_opt:.6f}")
    print(f"Final L1 Distance: {final_distance:.6f}")
    
    # Generate LaTeX equation
    print("\n" + "="*60)
    print("LaTeX Equation for Desmos:")
    print("="*60)
    latex_eq = f"\\left(t*\\cos({theta_opt:.6f})-e^{{{M_opt:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta_opt:.6f})\\ +{X_opt:.6f},42+\\ t*\\sin({theta_opt:.6f})+e^{{{M_opt:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_opt:.6f})\\right)"
    print(latex_eq)
    
    # Visualization
    print("\nGenerating visualization...")
    t_plot = np.linspace(6, 60, 1000)
    x_plot, y_plot = parametric_curve(t_plot, theta_opt, M_opt, X_opt)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(data_points[:, 0], data_points[:, 1], alpha=0.3, s=1, label='Data Points', color='blue')
    plt.plot(x_plot, y_plot, 'r-', linewidth=2, label='Fitted Curve')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Parametric Curve Fitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('curve_fit.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'curve_fit.png'")
    
    # Save results to file
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("Parametric Curve Optimization Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Theta: {theta_opt:.6f} radians = {np.degrees(theta_opt):.6f} degrees\n")
        f.write(f"M: {M_opt:.6f}\n")
        f.write(f"X: {X_opt:.6f}\n")
        f.write(f"Final L1 Distance: {final_distance:.6f}\n\n")
        f.write("LaTeX Equation for Desmos:\n")
        f.write(latex_eq + "\n")
    
    print("\nResults saved to 'results.txt'")
    
    return theta_opt, M_opt, X_opt


if __name__ == "__main__":
    theta, M, X = main()

