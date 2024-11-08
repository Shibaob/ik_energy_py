import numpy as np
import matplotlib.pyplot as plt
from dls_ik_solver import IK as DLS_IK
from selec_dls_ik_solver import IK as SDLS_IK

def run_and_plot(ik_solver, target_pose, seed, method_name):
    solution, success, rollout = ik_solver.inverse(target_pose, seed, debug=True)

    if success:
        print(f"{method_name} IK solution found:", solution)
        
        # Verify the solution
        fk = ik_solver.fk
        _, final_pose = fk.forward(solution)
        final_distance, final_angle = ik_solver.distance_and_angle(target_pose, final_pose)
        print(f"Final position error: {final_distance:.6f}")
        print(f"Final orientation error: {final_angle:.6f}")
    else:
        print(f"{method_name} failed to find IK solution")

    # Plot the convergence
    position_errors = []
    orientation_errors = []
    for q in rollout:
        _, pose = fk.forward(q)
        distance, angle = ik_solver.distance_and_angle(target_pose, pose)
        position_errors.append(distance)
        orientation_errors.append(angle)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(position_errors)
    plt.title(f'{method_name} Position Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (m)')

    plt.subplot(1, 2, 2)
    plt.plot(orientation_errors)
    plt.title(f'{method_name} Orientation Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (rad)')

    plt.tight_layout()
    plt.show()

    return len(rollout), final_distance, final_angle

if __name__ == "__main__":
    # Define a target pose (4x4 transformation matrix)
    target_pose = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.7],
        [0, 0, 0, 1]
    ])

    # Define a seed configuration
    seed = np.zeros(7)

    # Run DLS method
    dls_solver = DLS_IK()
    dls_iterations, dls_pos_error, dls_ori_error = run_and_plot(dls_solver, target_pose, seed, "DLS")

    # Run Selective Damping method
    sdls_solver = SDLS_IK()
    sdls_iterations, sdls_pos_error, sdls_ori_error = run_and_plot(sdls_solver, target_pose, seed, "Selective Damping")

    # Compare results
    print("\nComparison:")
    print(f"DLS iterations: {dls_iterations}, Final position error: {dls_pos_error:.6f}, Final orientation error: {dls_ori_error:.6f}")
    print(f"SDLS iterations: {sdls_iterations}, Final position error: {sdls_pos_error:.6f}, Final orientation error: {sdls_ori_error:.6f}")