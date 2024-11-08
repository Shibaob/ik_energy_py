import numpy as np
from numpy import pi, cos, sin
from quadprog import solve_qp

class FK:
    def __init__(self):
        self.dh_params = self.init_dh_params()
        self.joint_offsets = self.init_joint_offsets()

    def init_dh_params(self):
        return [
            [0, -np.pi/2, 0.333],
            [0, np.pi/2, 0],
            [0.082, np.pi/2, 0.316],
            [-0.082, -np.pi/2, 0],
            [0, np.pi/2, 0.384],
            [0.088, np.pi/2, 0],
            [0, 0, 0.051 + 0.159]
        ]

    def init_joint_offsets(self):
        return [
            [0, 0, 0.141],
            [0, 0, 0],
            [0, 0, 0.195],
            [0, 0, 0],
            [0, 0, 0.125],
            [0, 0, -0.015],
            [0, 0, 0.051]
        ]

    def build_dh_transform(self, a, alpha, d, theta):
        return np.array([
            [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
            [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])

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
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    center = lower + (upper - lower) / 2
    fk = FK()

    def __init__(self, linear_tol=1e-3, angular_tol=1e-2, max_steps=1000, min_step_size=1e-6):
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

    def is_valid_solution(self, q, target):
        q = np.array(q)
        if ((q < IK.lower) | (q > IK.upper)).any():
            return False

        _, T0e = IK.fk.forward(q)
        displacement, axis = IK.displacement_and_axis(target, T0e)
        distance = np.linalg.norm(displacement)
        angle = np.arcsin(np.clip(np.linalg.norm(axis), -1, 1))

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

        for step in range(self.max_steps):
            rollout.append(q)

            J = calcJacobian(q)
            _, T0e = IK.fk.forward(q)
            displacement, axis = IK.displacement_and_axis(target, T0e)
            error = np.concatenate([displacement, axis])

            linear_error = np.linalg.norm(displacement)
            angular_error = np.linalg.norm(axis)

            if linear_error < self.linear_tol and angular_error < self.angular_tol:
                print(f"Converged after {step} iterations")
                break

            n_joints = 7
            
            # Revised QP formulation with soft constraints
            H = 2 * (J.T @ J + 0.1 * np.eye(n_joints))
            f = -2 * J.T @ error + 0.1 * (q - self.center)

            # Soft joint limit constraints
            A = np.vstack([np.eye(n_joints), -np.eye(n_joints)])
            b = np.concatenate([self.upper - q, q - self.lower])

            # Solve QP problem
            try:
                dq = solve_qp(H, f, A.T, b, meq=0)[0]
            except ValueError as e:
                print(f"QP solver failed at iteration {step}, using enhanced DLS fallback. Error: {str(e)}")
                # Enhanced DLS fallback
                W = np.diag(1 / (np.square(self.upper - self.lower) + 1e-6))
                lambda_ = 0.1 * np.trace(J @ J.T) / n_joints
                dq = np.linalg.solve(J.T @ J + lambda_ * W, J.T @ error - 0.1 * W @ (q - self.center))

            # Adaptive step size
            alpha = 1.0
            for _ in range(10):  # Max 10 attempts to find a good step size
                q_new = np.clip(q + alpha * dq, self.lower, self.upper)
                _, T0e_new = IK.fk.forward(q_new)
                displacement_new, axis_new = IK.displacement_and_axis(target, T0e_new)
                error_new = np.concatenate([displacement_new, axis_new])
                if np.linalg.norm(error_new) < np.linalg.norm(error) or alpha < 0.01:
                    break
                alpha *= 0.5

            if np.linalg.norm(q_new - q) < self.min_step_size:
                print(f"Stopped due to small step size after {step} iterations")
                break

            q = q_new

        q[-1] = self.adjust_last_joint(q[-1])
        q = self.refine_solution(q, target)

        success = self.is_valid_solution(q, target)
        if not success:
            print(f"Failed to converge after {self.max_steps} iterations")
            print(f"Final linear error: {linear_error:.6f}, angular error: {angular_error:.6f}")
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
            b = np.zeros(6)

            try:
                dq = solve_qp(H, f, A.T, b, meq=6)[0]
                q_refined = np.clip(q + dq, self.lower, self.upper)
                if self.is_valid_solution(q_refined, target):
                    return q_refined
            except ValueError:
                pass

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

# Example usage
if __name__ == "__main__":
    ik_solver = IK()
    
    # Define a target pose
    target_pose = create_target_pose([0.5, 0, 0.5], [0, 0, 0])
    
    # Try multiple initial seeds
    seeds = [
        np.zeros(7),
        np.random.uniform(IK.lower, IK.upper),
        IK.center,
        np.array([0, -pi/4, 0, -pi/2, 0, pi/4, 0])
    ]
    
    for i, seed in enumerate(seeds):
        print(f"\nTrying seed {i + 1}")
        solution, success, _ = ik_solver.inverse(target_pose, seed)
        
        if success:
            print("IK solved successfully")
            print("Solution:", solution)
            
            # Verify the solution
            _, T0e = ik_solver.fk.forward(solution)
            displacement, axis = IK.displacement_and_axis(target_pose, T0e)
            print(f"Final linear error: {np.linalg.norm(displacement):.6f}")
            print(f"Final angular error: {np.linalg.norm(axis):.6f}")
            break
        else:
            print("IK failed to find a solution")
    
    if not success:
        print("\nFailed to find a solution with all seeds")
