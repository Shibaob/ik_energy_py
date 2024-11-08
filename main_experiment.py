import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy import stats
from qp_weighted import IK
import time

def create_rotation_matrix(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def create_target_pose(position, orientation):
    R = create_rotation_matrix(*orientation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T

scenarios = [
    {
        "name": "Nominal Reach",
        "target": create_target_pose([0.5, 0, 0.5], [0, 0, 0]),
    },
    {
        "name": "Extended Reach",
        "target": create_target_pose([0.8, 0, 0.2], [0, pi/4, 0]),
    },
    {
        "name": "Overhead Reach",
        "target": create_target_pose([0, 0.3, 0.9], [pi/2, 0, 0]),
    },
    {
        "name": "Low Reach",
        "target": create_target_pose([0.4, -0.4, 0.1], [0, -pi/4, pi/4]),
    },
    {
        "name": "Complex Orientation",
        "target": create_target_pose([0.5, 0.5, 0.5], [pi/4, pi/6, -pi/3]),
    },
]

def run_scenario(ik_solver, scenario, num_runs=100):
    results = []
    for _ in range(num_runs):
        seed = np.random.uniform(ik_solver.lower, ik_solver.upper)
        start_time = time.time()
        solution, success, rollout = ik_solver.inverse(scenario["target"], seed)
        comp_time = time.time() - start_time
        
        if success:
            _, T0e = ik_solver.fk.forward(solution)
            final_displacement, final_axis = ik_solver.displacement_and_axis(scenario["target"], T0e)
            linear_error = np.linalg.norm(final_displacement)
            angular_error = np.linalg.norm(final_axis)
            
            results.append({
                "success": success,
                "linear_error": linear_error,
                "angular_error": angular_error,
                "iterations": len(rollout),
                "computation_time": comp_time
            })
        else:
            results.append({
                "success": success,
                "iterations": len(rollout),
                "computation_time": comp_time
            })
    
    return results

def evaluate_scenarios(ik_solver, scenarios):
    all_results = {}
    for scenario in scenarios:
        print(f"Running scenario: {scenario['name']}")
        results = run_scenario(ik_solver, scenario)
        all_results[scenario['name']] = results
    return all_results

def analyze_results(uniform_results, weighted_results):
    for scenario in scenarios:
        name = scenario['name']
        uniform = uniform_results[name]
        weighted = weighted_results[name]
        
        print(f"\nScenario: {name}")
        
        # Success rate
        uniform_success = sum(r['success'] for r in uniform) / len(uniform)
        weighted_success = sum(r['success'] for r in weighted) / len(weighted)
        print(f"Success rate - Uniform: {uniform_success:.2%}, Weighted: {weighted_success:.2%}")
        
        # For successful runs only
        uniform_successful = [r for r in uniform if r['success']]
        weighted_successful = [r for r in weighted if r['success']]
        
        if uniform_successful and weighted_successful:
            metrics = ['linear_error', 'angular_error', 'iterations', 'computation_time']
            for metric in metrics:
                uniform_values = [r[metric] for r in uniform_successful]
                weighted_values = [r[metric] for r in weighted_successful]
                
                u_mean, u_std = np.mean(uniform_values), np.std(uniform_values)
                w_mean, w_std = np.mean(weighted_values), np.std(weighted_values)
                
                print(f"{metric.capitalize()}:")
                print(f"  Uniform  - Mean: {u_mean:.4f}, Std: {u_std:.4f}")
                print(f"  Weighted - Mean: {w_mean:.4f}, Std: {w_std:.4f}")
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(uniform_values, weighted_values)
                print(f"  T-test - t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
                
                # Calculate Cohen's d for effect size
                pooled_std = np.sqrt((np.std(uniform_values)**2 + np.std(weighted_values)**2) / 2)
                cohen_d = (np.mean(weighted_values) - np.mean(uniform_values)) / pooled_std
                print(f"  Cohen's d: {cohen_d:.4f}")

def plot_results(uniform_results, weighted_results):
    metrics = ['linear_error', 'angular_error', 'iterations', 'computation_time']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5*len(metrics)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        u_data = []
        w_data = []
        
        for scenario in scenarios:
            name = scenario['name']
            u_values = [r[metric] for r in uniform_results[name] if r['success']]
            w_values = [r[metric] for r in weighted_results[name] if r['success']]
            u_data.append(u_values)
            w_data.append(w_values)
        
        ax.boxplot([u_data[j] + w_data[j] for j in range(len(scenarios))], 
                   labels=[f"{s['name']}\nU W" for s in scenarios])
        ax.set_title(f"{metric.capitalize()} Comparison")
        ax.set_ylabel(metric.replace('_', ' ').capitalize())
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ik_solver_uniform = IK()
    ik_solver_uniform.joint_weights = np.ones(7)  # Set uniform weights

    ik_solver_weighted = IK()
    # The weights are already set in the constructor, but you can modify them if needed:
    # ik_solver_weighted.joint_weights = np.array([1.0, 1.2, 1.0, 0.8, 1.0, 0.8, 0.6])

    print("Evaluating uniform weights:")
    results_uniform = evaluate_scenarios(ik_solver_uniform, scenarios)

    print("\nEvaluating custom weights:")
    results_weighted = evaluate_scenarios(ik_solver_weighted, scenarios)

    analyze_results(results_uniform, results_weighted)
    plot_results(results_uniform, results_weighted)
