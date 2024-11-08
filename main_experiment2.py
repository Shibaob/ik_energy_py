import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy import stats
from qp_weighted import IK as QP_IK
from selec_dls_ik_solver import IK as SD_IK
import time

#
# 这里新增了weighted_qp与SD的比较
#

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

def analyze_results(uniform_results, weighted_results, sd_results):
    for scenario in scenarios:
        name = scenario['name']
        uniform = uniform_results[name]
        weighted = weighted_results[name]
        sd = sd_results[name]
        
        print(f"\nScenario: {name}")
        
        # Success rate
        uniform_success = sum(r['success'] for r in uniform) / len(uniform)
        weighted_success = sum(r['success'] for r in weighted) / len(weighted)
        sd_success = sum(r['success'] for r in sd) / len(sd)
        print(f"Success rate - Uniform: {uniform_success:.2%}, Weighted: {weighted_success:.2%}, Selective Damping: {sd_success:.2%}")
        
        # For successful runs only
        uniform_successful = [r for r in uniform if r['success']]
        weighted_successful = [r for r in weighted if r['success']]
        sd_successful = [r for r in sd if r['success']]
        
        if uniform_successful and weighted_successful and sd_successful:
            metrics = ['linear_error', 'angular_error', 'iterations', 'computation_time']
            for metric in metrics:
                uniform_values = [r[metric] for r in uniform_successful]
                weighted_values = [r[metric] for r in weighted_successful]
                sd_values = [r[metric] for r in sd_successful]
                
                u_mean, u_std = np.mean(uniform_values), np.std(uniform_values)
                w_mean, w_std = np.mean(weighted_values), np.std(weighted_values)
                sd_mean, sd_std = np.mean(sd_values), np.std(sd_values)
                
                print(f"{metric.capitalize()}:")
                print(f"  Uniform  - Mean: {u_mean:.4f}, Std: {u_std:.4f}")
                print(f"  Weighted - Mean: {w_mean:.4f}, Std: {w_std:.4f}")
                print(f"  Selective Damping - Mean: {sd_mean:.4f}, Std: {sd_std:.4f}")
                
                # Perform ANOVA
                f_stat, p_value = stats.f_oneway(uniform_values, weighted_values, sd_values)
                print(f"  ANOVA - F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")

def plot_results(uniform_results, weighted_results, sd_results):
    metrics = ['linear_error', 'angular_error', 'iterations', 'computation_time']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 6*len(metrics)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = []
        labels = []
        
        for scenario in scenarios:
            name = scenario['name']
            u_values = [r[metric] for r in uniform_results[name] if r['success']]
            w_values = [r[metric] for r in weighted_results[name] if r['success']]
            sd_values = [r[metric] for r in sd_results[name] if r['success']]
            
            if u_values or w_values or sd_values:  # Only add data if at least one method has successful results
                data.extend([u_values, w_values, sd_values])
                labels.extend([f"{name}\nU", f"{name}\nW", f"{name}\nSD"])
        
        if data:  # Only create box plot if there's data to plot
            bp = ax.boxplot(data, labels=labels)
            ax.set_title(f"{metric.capitalize()} Comparison")
            ax.set_ylabel(metric.replace('_', ' ').capitalize())
            
            # Color the boxes
            colors = ['blue', 'orange', 'green']
            for j, box in enumerate(bp['boxes']):
                box.set_color(colors[j % 3])
        else:
            ax.text(0.5, 0.5, "No successful solutions", ha='center', va='center')
            ax.set_title(f"{metric.capitalize()} Comparison (No Data)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ik_solver_uniform = QP_IK()
    ik_solver_uniform.joint_weights = np.ones(7)  # Set uniform weights

    ik_solver_weighted = QP_IK()
    # The weights are already set in the constructor, but you can modify them if needed:
    # ik_solver_weighted.joint_weights = np.array([1.0, 1.2, 1.0, 0.8, 1.0, 0.8, 0.6])

    ik_solver_sd = SD_IK()  # Selective Damping IK solver

    print("Evaluating uniform weights:")
    results_uniform = evaluate_scenarios(ik_solver_uniform, scenarios)

    print("\nEvaluating custom weights:")
    results_weighted = evaluate_scenarios(ik_solver_weighted, scenarios)

    print("\nEvaluating selective damping:")
    results_sd = evaluate_scenarios(ik_solver_sd, scenarios)

    analyze_results(results_uniform, results_weighted, results_sd)
    plot_results(results_uniform, results_weighted, results_sd)
