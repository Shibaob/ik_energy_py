import numpy as np
from numpy import cos, sin
from scipy.optimize import minimize

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
        # Define weights for each joint (adjust these values as needed)
        self.joint_weights = np.array([1.0, 1.2, 1.0, 0.8, 1.0, 0.8, 0.6])

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

        def objective(dq):
            # Weighted objective function
            return np.sum(self.joint_weights * dq**2)

        def constraint(dq):
            nonlocal q, target
            q_new = q + dq
            _, T0e = IK.fk.forward(q_new)
            displacement, axis = IK.displacement_and_axis(target, T0e)
            return np.concatenate([displacement, axis])

        while True:
            rollout.append(q)

            # Set up optimization problem
            cons = {'type': 'eq', 'fun': constraint}
            res = minimize(objective, np.zeros(7), method='SLSQP', constraints=cons,
                           options={'ftol': 1e-6, 'maxiter': 100})

            if not res.success:
                # Fallback to weighted damped least squares
                J = calcJacobian(q)
                _, T0e = IK.fk.forward(q)
                displacement, axis = IK.displacement_and_axis(target, T0e)
                error = np.concatenate([displacement, axis])
                W = np.diag(self.joint_weights)
                lambda_ = 0.01
                dq = np.linalg.solve(J.T @ J + lambda_ * W, J.T @ error)
            else:
                dq = res.x

            if (len(rollout) == self.max_steps) or (np.linalg.norm(dq) < self.min_step_size):
                break

            q = q + dq
            q[:-1] = np.clip(q[:-1], IK.lower[:-1], IK.upper[:-1])

        q[-1] = self.adjust_last_joint(q[-1])

        success = self.is_valid_solution(q, target)
        return q, success, rollout

if __name__ == "__main__":
    # Create an instance of the IK solver
    ik_solver = IK()

    # Define a target pose (4x4 transformation matrix)
    target_pose = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.7],
        [0, 0, 0, 1]
    ])

    # Define a seed configuration
    seed = np.zeros(7)

    # Solve IK
    solution, success, rollout = ik_solver.inverse(target_pose, seed)

    if success:
        print("IK solution found:", solution)
    else:
        print("Failed to find IK solution")

    # Print the final error
    _, T0e = ik_solver.fk.forward(solution)
    final_displacement, final_axis = IK.displacement_and_axis(target_pose, T0e)
    print("Final linear error:", np.linalg.norm(final_displacement))
    print("Final angular error:", np.linalg.norm(final_axis))
