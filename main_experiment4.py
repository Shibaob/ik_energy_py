import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy import stats
from qp_weighted import IK as QP_IK
from qp_weighted_adap import IK as AQP_IK
from selec_dls_ik_solver import IK as SD_IK
from dls_ik_solver import IK as DLS_IK  # Added basic DLS IK
import time

#
# the latest experiment results
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
            position_error = np.linalg.norm(final_displacement)  # Changed from linear_error
            orientation_error = np.arccos(np.clip((np.trace(scenario["target"][:3, :3].T @ T0e[:3, :3]) - 1) / 2, -1, 1))  # Changed from angular_error
            
            results.append({
                "success": success,
                "position_error": position_error,  # Changed name
                "orientation_error": orientation_error,  # Changed name
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

def analyze_results(weighted_results, adaptive_results, sd_results, dls_results):
    metrics = ['success_rate', 'position_error', 'orientation_error', 'iterations', 'computation_time']
    for metric in metrics:
        print(f"\n{'=' * 40}")
        print(f"{metric.replace('_', ' ').capitalize()}")
        print(f"{'=' * 40}")
        
        for scenario in scenarios:
            name = scenario['name']
            weighted = weighted_results[name]
            adaptive = adaptive_results[name]
            sd = sd_results[name]
            dls = dls_results[name]
            
            print(f"\nScenario: {name}")
            
            if metric == 'success_rate':
                weighted_success = sum(r['success'] for r in weighted) / len(weighted)
                adaptive_success = sum(r['success'] for r in adaptive) / len(adaptive)
                sd_success = sum(r['success'] for r in sd) / len(sd)
                dls_success = sum(r['success'] for r in dls) / len(dls)
                print(f"  Weighted QP: {weighted_success:.2%}")
                print(f"  Adaptive QP: {adaptive_success:.2%}")
                print(f"  Selective Damping: {sd_success:.2%}")
                print(f"  Basic DLS: {dls_success:.2%}")
            else:
                weighted_values = [r[metric] for r in weighted if r['success']]
                adaptive_values = [r[metric] for r in adaptive if r['success']]
                sd_values = [r[metric] for r in sd if r['success']]
                dls_values = [r[metric] for r in dls if r['success']]
                
                if all([weighted_values, adaptive_values, sd_values, dls_values]):
                    print_metric_stats("Weighted QP", weighted_values)
                    print_metric_stats("Adaptive QP", adaptive_values)
                    print_metric_stats("Selective Damping", sd_values)
                    print_metric_stats("Basic DLS", dls_values)
                    
                    # One-way ANOVA
                    f_stat, p_value = stats.f_oneway(weighted_values, adaptive_values, sd_values, dls_values)
                    print(f"\n  Statistical Analysis:")
                    print(f"  ANOVA - F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
                    
                    # Post-hoc analysis if ANOVA shows significance
                    if p_value < 0.05:
                        print("\n  Pairwise t-tests (with Bonferroni correction):")
                        methods = [
                            ("WQP vs AQP", weighted_values, adaptive_values),
                            ("WQP vs SD", weighted_values, sd_values),
                            ("WQP vs DLS", weighted_values, dls_values),
                            ("AQP vs SD", adaptive_values, sd_values),
                            ("AQP vs DLS", adaptive_values, dls_values),
                            ("SD vs DLS", sd_values, dls_values)
                        ]
                        
                        # Bonferroni correction for multiple comparisons
                        alpha = 0.05 / len(methods)
                        
                        for name, group1, group2 in methods:
                            t_stat, p_val = stats.ttest_ind(group1, group2)
                            significant = "significant" if p_val < alpha else "not significant"
                            print(f"    {name}:")
                            print(f"      t-statistic: {t_stat:.4f}")
                            print(f"      p-value: {p_val:.4e} ({significant})")
                            print(f"      Mean difference: {np.mean(group1) - np.mean(group2):.6f}")
                else:
                    print("  Insufficient successful runs for comparison")

def print_metric_stats(method_name, values):
    """
    Print statistical metrics for a given method's results
    """
    mean = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    print(f"  {method_name}:")
    print(f"    Mean: {mean:.6f} ± {std:.6f}")
    print(f"    Median: {median:.6f}")
    print(f"    Range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"    95% CI: [{mean - 1.96*std/np.sqrt(len(values)):.6f}, {mean + 1.96*std/np.sqrt(len(values)):.6f}]")

def plot_results(weighted_results, adaptive_results, sd_results, dls_results):
    metrics = ['position_error', 'orientation_error', 'iterations', 'computation_time']
    for metric in metrics:
        plt.figure(figsize=(15, 6))
        data = []
        labels = []
        
        for scenario in scenarios:
            name = scenario['name']
            w_values = [r[metric] for r in weighted_results[name] if r['success']]
            a_values = [r[metric] for r in adaptive_results[name] if r['success']]
            sd_values = [r[metric] for r in sd_results[name] if r['success']]
            dls_values = [r[metric] for r in dls_results[name] if r['success']]
            
            if w_values or a_values or sd_values or dls_values:
                data.extend([w_values, a_values, sd_values, dls_values])
                labels.extend([f"{name}\nWQP", f"{name}\nAQP", f"{name}\nSD", f"{name}\nDLS"])
        
        if data:
            bp = plt.boxplot(data, labels=labels)
            plt.title(f"{metric.replace('_', ' ').capitalize()} Comparison")
            plt.ylabel(metric.replace('_', ' ').capitalize())
            
            colors = ['orange', 'red', 'green', 'blue']  # Added color for DLS
            for j, box in enumerate(bp['boxes']):
                box.set_color(colors[j % 4])
        else:
            plt.text(0.5, 0.5, "No successful solutions", ha='center', va='center')
            plt.title(f"{metric.replace('_', ' ').capitalize()} Comparison (No Data)")
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    ik_solver_weighted = QP_IK()
    ik_solver_adaptive = AQP_IK()
    ik_solver_sd = SD_IK()
    ik_solver_dls = DLS_IK()

    print("Evaluating weighted QP:")
    results_weighted = evaluate_scenarios(ik_solver_weighted, scenarios)

    print("\nEvaluating adaptive QP:")
    results_adaptive = evaluate_scenarios(ik_solver_adaptive, scenarios)

    print("\nEvaluating selective damping:")
    results_sd = evaluate_scenarios(ik_solver_sd, scenarios)

    print("\nEvaluating basic DLS:")
    results_dls = evaluate_scenarios(ik_solver_dls, scenarios)

    print("\nAnalyzing Results:")
    analyze_results(results_weighted, results_adaptive, results_sd, results_dls)

    print("\nGenerating Plots...")
    plot_results(results_weighted, results_adaptive, results_sd, results_dls)