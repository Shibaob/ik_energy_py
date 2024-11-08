import numpy as np
from numpy import cos, sin
from scipy.optimize import minimize

class FK:
    # Keeping the original FK class unchanged since it's working well
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



class IK:
    # JOINT LIMITS
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    center = lower + (upper - lower) / 2

    def __init__(self, linear_tol=1e-3, angular_tol=1e-2, max_steps=500, min_step_size=1e-5):
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size
        self.base_weights = np.array([1.0, 1.2, 1.0, 0.8, 1.0, 0.8, 0.6])
        self.fk = FK()

    def calculate_adaptive_weights(self, q):
        normalized_position = (q - self.lower) / (self.upper - self.lower)
        limit_factor = 1 + np.abs(normalized_position - 0.5) * 2
        center_distance = np.abs(q - self.center) / (self.upper - self.lower)
        center_factor = 1 + center_distance
        return self.base_weights * limit_factor * center_factor

    @staticmethod
    def calcJacobian(fk, q):
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
        if ((q < self.lower) | (q > self.upper)).any():
            return False

        _, T0e = self.fk.forward(q)
        distance, angle = self.distance_and_angle(target, T0e)

        return (distance <= self.linear_tol) and (angle <= self.angular_tol)

    def adjust_last_joint(self, angle):
        if abs(angle) - self.upper[-1] < 0.4887:
            angle = np.clip(angle, self.lower[-1], self.upper[-1])
        
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi

        return angle

    def inverse(self, target, seed):
        """
        Legacy inverse method maintained for backwards compatibility.
        New implementations should use the EnhancedIK class.
        """
        q = seed.copy()
        rollout = []

        def objective(dq):
            weights = self.calculate_adaptive_weights(q + dq)
            return np.sum(weights * dq**2)

        def constraint(dq):
            q_new = q + dq
            _, T0e = self.fk.forward(q_new)
            displacement, axis = self.displacement_and_axis(target, T0e)
            return np.concatenate([displacement, axis])

        for _ in range(self.max_steps):
            rollout.append(q.copy())

            cons = {'type': 'eq', 'fun': constraint}
            res = minimize(objective, np.zeros(7), method='SLSQP', constraints=cons,
                           options={'ftol': 1e-6, 'maxiter': 100})

            if not res.success:
                # Fallback to weighted damped least squares
                J = self.calcJacobian(self.fk, q)
                _, T0e = self.fk.forward(q)
                displacement, axis = self.displacement_and_axis(target, T0e)
                error = np.concatenate([displacement, axis])
                W = np.diag(self.calculate_adaptive_weights(q))
                lambda_ = 0.01
                dq = np.linalg.solve(J.T @ J + lambda_ * W, J.T @ error)
            else:
                dq = res.x

            if np.linalg.norm(dq) < self.min_step_size:
                break

            q += dq
            q[:-1] = np.clip(q[:-1], self.lower[:-1], self.upper[:-1])

        q[-1] = self.adjust_last_joint(q[-1])

        success = self.is_valid_solution(q, target)
        return q, success, rollout


class Task:
    def __init__(self, name, priority, weight):
        self.name = name
        self.priority = priority
        self.weight = weight
        self.active = True

    def compute_error(self, q):
        raise NotImplementedError("Subclasses must implement compute_error")

    def get_jacobian(self, q):
        raise NotImplementedError("Subclasses must implement get_jacobian")
        
    def get_error_size(self):
        raise NotImplementedError("Subclasses must implement get_error_size")

class EndEffectorTask(Task):
    def __init__(self, target, fk, priority=3, weight=1.0):
        super().__init__("end_effector", priority, weight)
        self.target = target
        self.fk = fk

    def compute_error(self, q):
        _, T0e = self.fk.forward(q)
        displacement, axis = IK.displacement_and_axis(self.target, T0e)
        return np.concatenate([displacement, axis])

    def get_jacobian(self, q):
        return IK.calcJacobian(self.fk, q)  # Returns 6x7 Jacobian
        
    def get_error_size(self):
        return 6  # 3 for position + 3 for orientation

class JointLimitTask(Task):
    def __init__(self, lower, upper, priority=2, weight=0.8):
        super().__init__("joint_limits", priority, weight)
        self.lower = lower
        self.upper = upper
        self.center = lower + (upper - lower) / 2
        self.range = upper - lower

    def compute_error(self, q):
        normalized_pos = (q - self.lower) / self.range
        return np.clip(normalized_pos - 0.5, -0.4, 0.4)

    def get_jacobian(self, q):
        return np.eye(7)
        
    def get_error_size(self):
        return 7  # One error term per joint

class EnhancedIK(IK):
    def __init__(self, linear_tol=1e-3, angular_tol=1e-2, max_steps=500, min_step_size=1e-5):
        super().__init__(linear_tol, angular_tol, max_steps, min_step_size)
        self.tasks = []
        self.consistency_threshold = 0.8

    def add_task(self, task):
        self.tasks.append(task)
        self.tasks.sort(key=lambda x: x.priority, reverse=True)

    def check_task_consistency(self, q):
        n_tasks = len(self.tasks)
        consistency_matrix = np.zeros((n_tasks, n_tasks))
        
        for i in range(n_tasks):
            for j in range(i+1, n_tasks):
                if not (self.tasks[i].active and self.tasks[j].active):
                    continue
                    
                J_i = self.tasks[i].get_jacobian(q)
                J_j = self.tasks[j].get_jacobian(q)
                
                # Use only the first 6 rows if the Jacobian is larger
                if J_i.shape[0] > 6:
                    J_i = J_i[:6, :]
                if J_j.shape[0] > 6:
                    J_j = J_j[:6, :]
                
                # Compute SVD for the reduced Jacobians
                U_i, _, _ = np.linalg.svd(J_i, full_matrices=False)
                U_j, _, _ = np.linalg.svd(J_j, full_matrices=False)
                
                # Compute overlap using the available dimensions
                min_dim = min(U_i.shape[1], U_j.shape[1])
                overlap = np.abs(np.dot(U_i[:, :min_dim].T, U_j[:, :min_dim])).max()
                
                consistency_matrix[i,j] = overlap
                consistency_matrix[j,i] = overlap
        
        return consistency_matrix < self.consistency_threshold

    def resolve_task_conflicts(self, q):
        consistency = self.check_task_consistency(q)
        for i in range(len(self.tasks)):
            for j in range(i+1, len(self.tasks)):
                if not consistency[i,j]:
                    self.tasks[j].weight *= 0.5

    def calculate_hierarchical_cost(self, q):
        cost = 0
        priority_scale = 1000
        
        for task in self.tasks:
            if not task.active:
                continue
            error = task.compute_error(q)
            level_weight = priority_scale ** task.priority
            cost += level_weight * task.weight * np.sum(error**2)
        
        return cost

    def create_task_constraints(self, q):
        constraints = []
        
        # Primary task (end-effector) as equality constraint
        primary_task = next(task for task in self.tasks if task.name == "end_effector")
        constraints.append({
            'type': 'eq',
            'fun': lambda dq: primary_task.compute_error(q + dq)
        })
        
        # Joint limits as inequality constraints
        constraints.append({
            'type': 'ineq',
            'fun': lambda dq: self.upper - (q + dq)  # Upper bound
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda dq: (q + dq) - self.lower  # Lower bound
        })
        
        return constraints

    def inverse(self, target, seed):
        q = seed.copy()
        rollout = []

        # Initialize tasks if not already present
        if not any(task.name == "end_effector" for task in self.tasks):
            self.add_task(EndEffectorTask(target, self.fk))
        if not any(task.name == "joint_limits" for task in self.tasks):
            self.add_task(JointLimitTask(self.lower, self.upper))

        def objective(dq):
            return self.calculate_hierarchical_cost(q + dq)

        for _ in range(self.max_steps):
            rollout.append(q.copy())
            
            # Check and resolve task conflicts
            self.resolve_task_conflicts(q)
            
            # Solve QP problem
            cons = self.create_task_constraints(q)
            res = minimize(objective, np.zeros(7), method='SLSQP', 
                         constraints=cons,
                         options={'ftol': 1e-6, 'maxiter': 100})

            if not res.success:
                # Enhanced fallback strategy with task priorities
                errors = []
                jacobians = []
                weights = []
                total_error_size = 0
                
                # First pass: calculate total error size and collect task data
                for task in self.tasks:
                    if not task.active:
                        continue
                    total_error_size += task.get_error_size()
                    
                    J = task.get_jacobian(q)
                    error = task.compute_error(q)
                    errors.append(error)
                    jacobians.append(J)
                    # Create weight array matching error size for this task
                    weights.extend([task.weight] * len(error))
                
                # Create proper sized weight matrix
                W = np.diag(weights)
                error_combined = np.concatenate(errors)
                J_combined = np.vstack(jacobians)
                
                # Weighted damped least squares with proper dimensions
                lambda_ = 0.01
                dq = np.linalg.solve(
                    J_combined.T @ W @ J_combined + lambda_ * np.eye(7),
                    J_combined.T @ W @ error_combined
                )
            else:
                dq = res.x

            if np.linalg.norm(dq) < self.min_step_size:
                break

            q += dq
            q[:-1] = np.clip(q[:-1], self.lower[:-1], self.upper[:-1])

        q[-1] = self.adjust_last_joint(q[-1])
        
        success = self.is_valid_solution(q, target)
        return q, success, rollout

if __name__ == "__main__":
    # Create an instance of the enhanced IK solver
    ik_solver = EnhancedIK()

    # Define a target pose
    target_pose = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.7],
        [0, 0, 0, 1]
    ])

    # Define a seed configuration
    seed = np.zeros(7)

    # Solve IK with enhanced capabilities
    solution, success, rollout = ik_solver.inverse(target_pose, seed)

    if success:
        print("IK solution found:", solution)
        # Print final errors for each task
        for task in ik_solver.tasks:
            error = task.compute_error(solution)
            print(f"{task.name} final error:", np.linalg.norm(error))
    else:
        print("Failed to find IK solution")
