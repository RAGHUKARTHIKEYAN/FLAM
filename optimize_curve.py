import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def parametric_curve(t, theta, M, X):
    
    t = np.array(t)
    abs_t = np.abs(t)
    
    
    exp_term = np.exp(M * abs_t)
    sin_03t = np.sin(0.3 * t)
    
    
    x = t * np.cos(theta) - exp_term * sin_03t * np.sin(theta) + X
    
    
    y = 42 + t * np.sin(theta) + exp_term * sin_03t * np.cos(theta)
    
    return x, y


def compute_l1_distance_uniform_sampling(theta, M, X, data_points, n_samples=1000):
    
    t_samples = np.linspace(6, 60, n_samples)
    
    
    x_pred, y_pred = parametric_curve(t_samples, theta, M, X)
    pred_points = np.column_stack([x_pred, y_pred])
    
    
    distances = cdist(pred_points, data_points, metric='cityblock')  
    
    
    min_distances = np.min(distances, axis=1)
    
    
    total_l1_distance = np.sum(min_distances)
    
    return total_l1_distance


def compute_l1_distance_data_driven(theta, M, X, data_points):
   
    total_distance = 0.0
    
    
    for x_data, y_data in data_points:
        
        def objective(t):
            if t < 6 or t > 60:
                return 1e10  
            x_pred, y_pred = parametric_curve(t, theta, M, X)
            return abs(x_pred - x_data) + abs(y_pred - y_data)
        
        
        result = minimize(objective, x0=33.0, bounds=[(6, 60)], method='L-BFGS-B')
        total_distance += result.fun
    
    return total_distance


def objective_function_uniform(params, data_points, n_samples=1000):
    
    theta, M, X = params
    
    
    theta_deg = np.degrees(theta)
    if theta_deg <= 0 or theta_deg >= 50:
        return 1e10
    if M <= -0.05 or M >= 0.05:
        return 1e10
    if X <= 0 or X >= 100:
        return 1e10
    
    return compute_l1_distance_uniform_sampling(theta, M, X, data_points, n_samples)


def objective_function_data_driven(params, data_points):
    
    theta, M, X = params
    
    
    theta_deg = np.degrees(theta)
    if theta_deg <= 0 or theta_deg >= 50:
        return 1e10
    if M <= -0.05 or M >= 0.05:
        return 1e10
    if X <= 0 or X >= 100:
        return 1e10
    
    return compute_l1_distance_data_driven(theta, M, X, data_points)


def main():
    
    df = pd.read_csv('xy_data.csv')
    data_points = df[['x', 'y']].values
    print(f"Loaded {len(data_points)} data points")
    
   
    theta_min = np.radians(0.01)  
    theta_max = np.radians(49.99)  
    
    
    M_min = -0.049
    M_max = 0.049
    
    
    X_min = 0.01
    X_max = 99.99
    
    bounds = [
        (theta_min, theta_max),
        (M_min, M_max),
        (X_min, X_max)
    ]
    
    print("\nStarting optimization...")
    print("This may take several minutes...")
    
    
    print("\nPhase 1: Global optimization with Differential Evolution...")
    result_phase1 = differential_evolution(
        objective_function_uniform,
        bounds,
        args=(data_points, 1000), 
        seed=42,
        maxiter=300,
        popsize=30,  
        atol=1e-8,
        tol=1e-8,
        mutation=(0.5, 1.5),  
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
    
   
    print("\nPhase 2: Local refinement with L-BFGS-B...")
    result_phase2 = minimize(
        objective_function_uniform,
        result_phase1.x,
        args=(data_points, 1000),  
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10}
    )
    
    print(f"Phase 2 completed. Best parameters:")
    print(f"  Theta = {np.degrees(result_phase2.x[0]):.6f} degrees ({result_phase2.x[0]:.6f} radians)")
    print(f"  M = {result_phase2.x[1]:.6f}")
    print(f"  X = {result_phase2.x[2]:.6f}")
    print(f"  L1 Distance = {result_phase2.fun:.6f}")
    
    
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
    
    
    print("\n" + "="*60)
    print("LaTeX Equation for Desmos:")
    print("="*60)
    latex_eq = f"\\left(t*\\cos({theta_opt:.6f})-e^{{{M_opt:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta_opt:.6f})\\ +{X_opt:.6f},42+\\ t*\\sin({theta_opt:.6f})+e^{{{M_opt:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_opt:.6f})\\right)"
    print(latex_eq)
    
    
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
    
    #
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

