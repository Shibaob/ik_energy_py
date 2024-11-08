import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming the IK and FK classes are defined above

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

# Define test scenarios
scenarios = [
    {
        "name": "Nominal Reach",
        "target": create_target_pose([0.5, 0, 0.5], [0, 0, 0]),
        "seed": np.zeros(7),
    },
    {
        "name": "Extended Reach",
        "target": create_target_pose([0.8, 0, 0.2], [0, pi/4, 0]),
        "seed": np.zeros(7),
    },
    {
        "name": "Overhead Reach",
        "target": create_target_pose([0, 0.3, 0.9], [pi/2, 0, 0]),
        "seed": np.zeros(7),
    },
    {
        "name": "Low Reach",
        "target": create_target_pose([0.4, -0.4, 0.1], [0, -pi/4, pi/4]),
        "seed": np.zeros(7),
    },
    {
        "name": "Complex Orientation",
        "target": create_target_pose([0.5, 0.5, 0.5], [pi/4, pi/6, -pi/3]),
        "seed": np.zeros(7),
    },
]

def run_scenario(ik_solver, scenario):
    target = scenario["target"]
    seed = scenario["seed"]
    
    solution, success, rollout = ik_solver.inverse(target, seed)
    
    if success:
        _, T0e = ik_solver.fk.forward(solution)
        final_displacement, final_axis = IK.displacement_and_axis(target, T0e)
        linear_error = np.linalg.norm(final_displacement)
        angular_error = np.linalg.norm(final_axis)
        
        return {
            "success": success,
            "solution": solution,
            "linear_error": linear_error,
            "angular_error": angular_error,
            "iterations": len(rollout),
        }
    else:
        return {
            "success": success,
            "iterations": len(rollout),
        }

def evaluate_scenarios(ik_solver, scenarios):
    results = []
    for scenario in scenarios:
        print(f"Running scenario: {scenario['name']}")
        result = run_scenario(ik_solver, scenario)
        result["name"] = scenario["name"]
        results.append(result)
        
        if result["success"]:
            print(f"  Success: True")
            print(f"  Linear error: {result['linear_error']:.6f}")
            print(f"  Angular error: {result['angular_error']:.6f}")
            print(f"  Iterations: {result['iterations']}")
        else:
            print(f"  Success: False")
            print(f"  Iterations: {result['iterations']}")
        print()
    
    return results

def plot_results(results):
    successful_results = [r for r in results if r["success"]]
    
    names = [r["name"] for r in successful_results]
    linear_errors = [r["linear_error"] for r in successful_results]
    angular_errors = [r["angular_error"] for r in successful_results]
    iterations = [r["iterations"] for r in successful_results]
    
    x = range(len(names))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    ax1.bar(x, linear_errors)
    ax1.set_ylabel("Linear Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    
    ax2.bar(x, angular_errors)
    ax2.set_ylabel("Angular Error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    
    ax3.bar(x, iterations)
    ax3.set_ylabel("Iterations")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create two IK solvers with different weights
    ik_solver_uniform = IK()
    ik_solver_weighted = IK()
    ik_solver_weighted.joint_weights = np.array([1.0, 1.5, 1.0, 0.8, 1.2, 0.7, 0.5])

    print("Evaluating uniform weights:")
    results_uniform = evaluate_scenarios(ik_solver_uniform, scenarios)
    plot_results(results_uniform)

    print("\nEvaluating custom weights:")
    results_weighted = evaluate_scenarios(ik_solver_weighted, scenarios)
    plot_results(results_weighted)

    # Compare the two sets of results
    for uniform, weighted in zip(results_uniform, results_weighted):
        if uniform["success"] and weighted["success"]:
            print(f"\nScenario: {uniform['name']}")
            print(f"Uniform weights - Linear error: {uniform['linear_error']:.6f}, Angular error: {uniform['angular_error']:.6f}, Iterations: {uniform['iterations']}")
            print(f"Custom weights  - Linear error: {weighted['linear_error']:.6f}, Angular error: {weighted['angular_error']:.6f}, Iterations: {weighted['iterations']}")
        elif uniform["success"] != weighted["success"]:
            print(f"\nScenario: {uniform['name']}")
            print(f"Uniform weights - Success: {uniform['success']}")
            print(f"Custom weights  - Success: {weighted['success']}")
