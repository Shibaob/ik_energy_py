import numpy as np
from numpy import cos, sin
from scipy.linalg import null_space

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

def IK_velocity(q_in, v_in, omega_in, alpha=1e-2):
    J = calcJacobian(q_in)
    v_w = np.append(v_in, omega_in, axis=0)
    lambda_ = 0.01  # Damping factor

    v_w_ = []
    J_ = []
    for i in range(len(v_w)):
        if not np.isnan(v_w[i]):
            v_w_.append(v_w[i])
            J_.append(J[i])

    if len(v_w_) == 0:
        return np.zeros((7,))

    J_ = np.array(J_)
    I = np.eye(J_.shape[1])  # Identity matrix    
    Jt = np.transpose(J_)  # Transpose of J_
    regularized_J = np.dot(Jt, J_) + lambda_**2 * I + alpha * I

    v_w_ = np.array(v_w_).reshape((len(v_w_), 1))

    dq = np.linalg.lstsq(regularized_J, np.dot(Jt, v_w_), rcond=None)[0][:, 0]

    return dq

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

    @staticmethod
    def end_effector_task(q, target):
        _, T0e = IK.fk.forward(q)
        displacement, axis = IK.displacement_and_axis(target, T0e)
        dq = IK_velocity(q, displacement, axis)
        return dq

    @staticmethod
    def joint_centering_task(q, rate=5e-1):
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset
        return dq

    def inverse(self, target, seed, debug=False):
        q = seed
        rollout = []

        for iteration in range(self.max_steps):
            rollout.append(q)

            # Calculate current end-effector pose
            _, T0e = IK.fk.forward(q)
            
            # Calculate errors
            displacement, axis = IK.displacement_and_axis(target, T0e)
            position_error = np.linalg.norm(displacement)
            orientation_error = np.linalg.norm(axis)

            # Calculate joint update
            dq_ik = self.end_effector_task(q, target)
            dq_center = self.joint_centering_task(q)

            J = calcJacobian(q)
            nullspace = null_space(J)
            if nullspace.size > 0:
                # Use the first column of the nullspace
                nullspace = nullspace[:, 0]
                dq_center_proj = nullspace * np.dot(dq_center, nullspace) / np.dot(nullspace, nullspace)
            else:
                dq_center_proj = np.zeros_like(q)

            dq = dq_ik + dq_center_proj
            dq_magnitude = np.linalg.norm(dq)

            if debug:
                print(f"Iteration {iteration}:")
                print(f"  Position error: {position_error:.6f}")
                print(f"  Orientation error: {orientation_error:.6f}")
                print(f"  Joint update magnitude: {dq_magnitude:.6f}")

            if dq_magnitude < self.min_step_size:
                if debug:
                    print("Converged: Joint update magnitude below threshold")
                break

            q = q + dq
            q[:-1] = np.clip(q[:-1], IK.lower[:-1], IK.upper[:-1])
            q[-1] = self.adjust_last_joint(q[-1])

            if self.is_valid_solution(q, target):
                if debug:
                    print("Converged: Valid solution found")
                return q, True, rollout

        if debug:
            if iteration == self.max_steps - 1:
                print("Failed to converge: Maximum iterations reached")
            print(f"Final position error: {position_error:.6f}")
            print(f"Final orientation error: {orientation_error:.6f}")

        return q, False, rollout

    def adjust_last_joint(self, angle):
        if abs(angle) - IK.upper[-1] < 0.4887:
            angle = np.clip(angle, IK.lower[-1], IK.upper[-1])
        
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi

        return angle


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

    # Solve IK with debugging output
    solution, success, rollout = ik_solver.inverse(target_pose, seed, debug=True)

    if success:
        print("\nIK solution found:")
        print(solution)
        
        # Verify the solution
        fk = FK()
        _, final_pose = fk.forward(solution)
        final_distance, final_angle = IK.distance_and_angle(target_pose, final_pose)
        print(f"\nFinal position error: {final_distance:.6f}")
        print(f"Final orientation error: {final_angle:.6f}")
        
        print("\nFinal end-effector pose:")
        print(final_pose)
        
        print("\nTarget pose:")
        print(target_pose)
        
        print(f"\nNumber of iterations: {len(rollout)}")
    else:
        print("\nFailed to find IK solution")
    
    # Optionally, you can add visualization of the convergence
    import matplotlib.pyplot as plt
    
    position_errors = []
    orientation_errors = []
    for q in rollout:
        _, pose = fk.forward(q)
        distance, angle = IK.distance_and_angle(target_pose, pose)
        position_errors.append(distance)
        orientation_errors.append(angle)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(position_errors)
    plt.title('Position Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (m)')

    plt.subplot(1, 2, 2)
    plt.plot(orientation_errors)
    plt.title('Orientation Error')
    plt.xlabel('Iteration')
    plt.ylabel('Error (rad)')

    plt.tight_layout()
    plt.show()
