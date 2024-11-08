import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from qp_weighted import IK as QP_IK, FK
from selec_dls_ik_solver import IK as SD_IK

def generate_workspace_samples(num_samples=1000):
    # Define workspace limits (adjust these based on your robot's actual workspace)
    x_range = (-0.8, 0.8)
    y_range = (-0.8, 0.8)
    z_range = (0.1, 1.5)
    
    positions = np.random.uniform(
        low=[x_range[0], y_range[0], z_range[0]],
        high=[x_range[1], y_range[1], z_range[1]],
        size=(num_samples, 3)
    )
    
    orientations = R.random(num_samples).as_matrix()
    
    return positions, orientations

def compute_position_error(desired, achieved):
    return np.linalg.norm(desired - achieved)

def compute_orientation_error(desired, achieved):
    error_matrix = np.dot(desired.T, achieved)
    rotation = R.from_matrix(error_matrix)
    angle = np.abs(rotation.as_rotvec()).sum()
    return angle

def compute_joint_space_error(q1, q2):
    return np.abs(q1 - q2).mean()

def evaluate_solver(solver, positions, orientations):
    fk = FK()
    position_errors = []
    orientation_errors = []
    joint_space_errors = []
    pose_reproduction_position_errors = []
    pose_reproduction_orientation_errors = []
    
    for pos, ori in zip(positions, orientations):
        target = np.eye(4)
        target[:3, :3] = ori
        target[:3, 3] = pos
        
        seed = np.random.uniform(solver.lower, solver.upper)
        q_sol, success, _ = solver.inverse(target, seed)
        
        if success:
            # Compute achieved end-effector pose
            _, T_achieved = fk.forward(q_sol)
            achieved_pos = T_achieved[:3, 3]
            achieved_ori = T_achieved[:3, :3]
            
            # Compute errors
            position_errors.append(compute_position_error(pos, achieved_pos))
            orientation_errors.append(compute_orientation_error(ori, achieved_ori))
            
            # Joint space error
            _, T_fk = fk.forward(q_sol)
            q_fk, _, _ = solver.inverse(T_fk, q_sol)
            joint_space_errors.append(compute_joint_space_error(q_sol, q_fk))
            
            # Pose reproduction error
            pose_reproduction_position_errors.append(compute_position_error(target[:3, 3], T_fk[:3, 3]))
            pose_reproduction_orientation_errors.append(compute_orientation_error(target[:3, :3], T_fk[:3, :3]))
    
    return {
        'position_errors': position_errors,
        'orientation_errors': orientation_errors,
        'joint_space_errors': joint_space_errors,
        'pose_reproduction_position_errors': pose_reproduction_position_errors,
        'pose_reproduction_orientation_errors': pose_reproduction_orientation_errors
    }


def summarize_errors(errors):
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'median': np.median(errors),
        '95th_percentile': np.percentile(errors, 95)
    }

def print_results(solver_name, results):
    print(f"\n{'=' * 40}")
    print(f"{solver_name} IK Solver Results:")
    print(f"{'=' * 40}")
    
    for metric, errors in results.items():
        summary = summarize_errors(errors)
        print(f"\n{metric.replace('_', ' ').capitalize()}:")
        print(f"  Mean: {summary['mean']:.6f}")
        print(f"  Std Dev: {summary['std']:.6f}")
        print(f"  Median: {summary['median']:.6f}")
        print(f"  95th Percentile: {summary['95th_percentile']:.6f}")

def plot_results(u_results, w_results, sd_results):
    metrics = ['position_errors', 'orientation_errors', 'joint_space_errors', 
               'pose_reproduction_position_errors', 'pose_reproduction_orientation_errors']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.boxplot([u_results[metric], w_results[metric], sd_results[metric]], 
                    labels=['Uniform', 'Weighted', 'Selective Damping'])
        plt.title(f'{metric.replace("_", " ").capitalize()}')
        plt.ylabel('Error')
        plt.yscale('log')  # Use log scale for better visualization of differences
        
        # Add scatter plots for outliers
        for i, data in enumerate([u_results[metric], w_results[metric], sd_results[metric]], 1):
            outliers = plt.boxplot([data])['fliers'][0].get_ydata()
            x = np.random.normal(i, 0.04, size=len(outliers))
            plt.scatter(x, outliers, alpha=0.5, s=2)
        
        plt.tight_layout()
        plt.show()

def run_experiment():
    print("Generating workspace samples...")
    positions, orientations = generate_workspace_samples()
    
    solvers = [
        ("Uniform-weighted QP", QP_IK(), lambda s: setattr(s, 'joint_weights', np.ones(7))),
        ("Custom-weighted QP", QP_IK(), lambda s: None),  # Assuming weights are set in constructor
        ("Selective Damping", SD_IK(), lambda s: None)
    ]
    
    results = {}
    
    for name, solver, weight_setter in solvers:
        print(f"\nEvaluating {name} IK solver...")
        weight_setter(solver)
        results[name] = evaluate_solver(solver, positions, orientations)
        print_results(name, results[name])
    
    print("\nGenerating plots...")
    plot_results(results["Uniform-weighted QP"], 
                 results["Custom-weighted QP"], 
                 results["Selective Damping"])
    
    return results

if __name__ == "__main__":
    run_experiment()