import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import cos, sin
from quadprog import solve_qp

class FK:
    def __init__(self):
        self.dh_params = self.init_dh_params()
        self.joint_offsets = self.init_joint_offsets()

    def init_dh_params(self):
        dh_params = [
            [0, -np.pi/2, 0.333],
            [0, np.pi/2, 0],
            [0.082, np.pi/2, 0.316],
            [-0.082, -np.pi/2, 0],
            [0, np.pi/2, 0.384],
            [0.088, np.pi/2, 0],
            [0, 0, 0.051 + 0.159]
        ]
        return dh_params

    def init_joint_offsets(self):
        joint_offsets = [
            [0, 0, 0.141],
            [0, 0, 0],
            [0, 0, 0.195],
            [0, 0, 0],
            [0, 0, 0.125],
            [0, 0, -0.015],
            [0, 0, 0.051]
        ]
        return joint_offsets

    def build_dh_transform(self, a, alpha, d, theta):
        A = np.array([
            [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
            [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])
        return np.round(A, 5)

    def forward(self, q):
        o = np.array([0, 0, 0])
        jointPositions = np.zeros((7, 3))
        T0e = np.identity(4)

        for i in range(7):
            jointPositions[i] = np.matmul(T0e, np.append((o + self.joint_offsets[i]), [1]))[:3]
            a, alpha, d = self.dh_params[i]
            T0e = np.matmul(T0e, self.build_dh_transform(a, alpha, d, q[i]))

        T0e = np.matmul(T0e, self.build_dh_transform(0, 0, 0, -np.pi/4))

        return jointPositions, T0e

def calcJacobian(q):
    fk = FK()
    joint_positions, T0e = fk.forward(q)

    o_n = np.matmul(T0e, np.array([0, 0, 0, 1]))[:3]
    o_i = o_n - joint_positions

    z_i = []
    T0e = np.identity(4)
    for i in range(7):
        z = np.matmul(T0e[:3, :3], np.array([0, 0, 1]))
        z_i.append(z / np.linalg.norm(z))

        a, alpha, d = fk.dh_params[i]
        T0e = np.matmul(T0e, fk.build_dh_transform(a, alpha, d, q[i]))

    J_v = np.array([np.cross(z_i[i], o_i[i]) for i in range(7)]).T
    z_i = np.array(z_i).T
    J = np.vstack((J_v, z_i))

    return J

class IK:
    # JOINT LIMITS
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    center = lower + (upper - lower) / 2
    fk = FK()

    def __init__(self, linear_tol=1e-3, angular_tol=1e-2, max_steps=500, min_step_size=1e-5):
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

    @staticmethod
    def displacement_and_axis(target, current):
        current_pos = np.matmul(current, np.array([0, 0, 0, 1]))
        target_pos = np.matmul(target, np.array([0, 0, 0, 1]))
        displacement = (target_pos - current_pos)[:3]

        R_c_w = current[:3, :3]
        R_t_w = target[:3, :3]
        R_w_t = R_t_w.T
        R_c_t = np.matmul(R_w_t, R_c_w)
        S = 0.5 * (R_c_t - R_c_t.T)
        a = np.array([-S[2, 1], -S[0, 2], -S[1, 0]])
        axis = np.matmul(R_t_w, a)

        return displacement, axis

    @staticmethod
    def distance_and_angle(G, H):
        displacement, axis = IK.displacement_and_axis(G, H)
        distance = np.linalg.norm(displacement)
        angle = np.arcsin(np.clip(np.linalg.norm(axis), -1, 1))
        return distance, angle

    def is_valid_solution(self, q, target):
        q = np.array(q)
        if ((q < IK.lower) | (q > IK.upper)).any():
            return False

        _, T0e = IK.fk.forward(q)
        distance, angle = IK.distance_and_angle(target, T0e)

        return (distance <= self.linear_tol) and (angle <= self.angular_tol)

    def adjust_last_joint(self, angle):
        if abs(angle) - IK.upper[-1] < 0.4887:
            angle = np.clip(angle, IK.lower[-1], IK.upper[-1])
        
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi

        return angle

    def inverse(self, target, seed):
        q = seed
        rollout = []

        while True:
            rollout.append(q)

            J = calcJacobian(q)
            _, T0e = IK.fk.forward(q)
            displacement, axis = IK.displacement_and_axis(target, T0e)
            error = np.concatenate([displacement, axis])

            n_joints = 7
            
            # Weighted objective function
            W = np.diag([1.0, 1.2, 1.0, 0.8, 1.0, 0.8, 0.6])  # Example weights
            H = W @ W.T  # Quadratic term in objective function
            
            # Joint centering in null space
            q_center = IK.center
            f = -W @ (q - q_center)  # Linear term in objective function

            # Equality constraint (J * dq = error)
            A_eq = J
            b_eq = error

            # Inequality constraints for joint velocity limits
            dq_max = 0.1 * np.ones(n_joints)  # Example max velocity
            A_ineq = np.vstack([np.eye(n_joints), -np.eye(n_joints)])
            b_ineq = np.concatenate([dq_max, dq_max])

            # Combine equality and inequality constraints
            A = np.vstack([A_eq, A_ineq])
            b = np.concatenate([b_eq, b_ineq])

            # Solve QP problem
            try:
                dq = solve_qp(H, f, A.T, b, A_eq.shape[0])[0]
            except ValueError:
                # Fallback to adaptive DLS
                sigma = np.linalg.svd(J, compute_uv=False)
                lambda_ = 0.01 * sigma[0] / sigma[-1] if sigma[-1] < 1e-5 else 0
                dq = np.linalg.solve(J.T @ J + lambda_ * np.eye(n_joints), J.T @ error - lambda_ * f)

            if (len(rollout) == self.max_steps) or (np.linalg.norm(dq) < self.min_step_size):
                break

            q = q + dq
            q[:-1] = np.clip(q[:-1], IK.lower[:-1], IK.upper[:-1])

        q[-1] = self.adjust_last_joint(q[-1])

        # Solution refinement
        q = self.refine_solution(q, target)

        success = self.is_valid_solution(q, target)
        return q, success, rollout

    def refine_solution(self, q, target):
        J = calcJacobian(q)
        _, T0e = IK.fk.forward(q)
        displacement, axis = IK.displacement_and_axis(target, T0e)
        error = np.concatenate([displacement, axis])

        if np.linalg.norm(error) < self.linear_tol + self.angular_tol:
            n_joints = 7
            H = np.eye(n_joints)
            f = np.zeros(n_joints)
            A = J
            b = np.zeros(6)  # Maintain current end-effector pose

            try:
                dq = solve_qp(H, f, A.T, b, meq=6)[0]
                q_refined = q + dq
                if self.is_valid_solution(q_refined, target):
                    return q_refined
            except ValueError:
                pass  # If refinement fails, return original solution

        return q


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
    ax1.set_title("Linear Error by Scenario")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    
    ax2.bar(x, angular_errors)
    ax2.set_ylabel("Angular Error")
    ax2.set_title("Angular Error by Scenario")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    
    ax3.bar(x, iterations)
    ax3.set_ylabel("Iterations")
    ax3.set_title("Iterations by Scenario")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create an instance of the IK solver
    ik_solver = IK()

    # Evaluate scenarios
    results = evaluate_scenarios(ik_solver, scenarios)

    # Plot results
    plot_results(results)