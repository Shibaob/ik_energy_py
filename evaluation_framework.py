import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from qp_weighted_adap import IK as QpWeightedIK
from selec_dls_ik_solver import IK as SDLSolver

@dataclass
class TestCase:
    pose: np.ndarray
    seed: np.ndarray
    description: str
    expected_success: bool

@dataclass
class BenchmarkResults:
    success_rate: float
    convergence_time: float
    position_error: float
    orientation_error: float
    manipulability: float
    joint_velocities: np.ndarray
    smoothness: float


class IKMetrics:
    def __init__(self):
        self.position_threshold = 1e-3
        self.orientation_threshold = 1e-2
        
    def compute_metrics(self, solver, test_case, seed):
        metrics = {
            'success_rate': self.compute_success_rate(solver, test_case, seed),
            'convergence_time': self.measure_convergence_time(solver, test_case, seed),
            'solution_quality': self.evaluate_solution_quality(solver, test_case, seed),
            'joint_motion': self.analyze_joint_motion(solver, test_case, seed),
            'numerical_stability': self.assess_numerical_stability(solver, test_case, seed)
        }
        return metrics
        
    def compute_success_rate(self, solver, poses, seeds, trials=100):
        successes = 0
        for pose, seed in zip(poses, seeds):
            solution, success, _ = solver.inverse(pose, seed)
            if success:
                successes += 1
        return successes / len(poses)

    def measure_convergence_time(self, solver, pose, seed):
        start_time = time.time()
        solution, _, _ = solver.inverse(pose, seed)
        end_time = time.time()
        return end_time - start_time
    
    def evaluate_solution_quality(self, solver, pose, seed):
        solution, _, rollout = solver.inverse(pose, seed)
        
        # Position error
        pos_error = np.linalg.norm(rollout[-1][:3, 3] - pose[:3, 3])
        
        # Orientation error
        R_error = np.matmul(rollout[-1][:3, :3].T, pose[:3, :3])
        angle_error = np.arccos((np.trace(R_error) - 1) / 2)
        
        return {
            'position_error': pos_error,
            'orientation_error': angle_error,
            'iterations': len(rollout)
        }


class ComparativeAnalysis:
    def run_comparison(self, qp_solver, sdls_solver, test_cases):
        results = {
            'singularity_handling': self.compare_singularity_handling(qp_solver, sdls_solver, test_cases['near_singularity']),
            'joint_limits': self.compare_joint_limits(qp_solver, sdls_solver, test_cases['joint_limits']),
            'convergence': self.compare_convergence(qp_solver, sdls_solver, test_cases['standard_poses']),
            'trajectory_tracking': self.compare_trajectory_tracking(qp_solver, sdls_solver, test_cases['dynamic_tracking']),
            'obstacle_avoidance': self.compare_obstacle_avoidance(qp_solver, sdls_solver, test_cases['obstacle_cases'])
        }
        return results
        
    def compare_singularity_handling(self, qp_solver, sdls_solver, singular_poses):
        qp_metrics = []
        sdls_metrics = []
        
        for pose in singular_poses:
            # Test both solvers
            qp_result = self.test_near_singularity(qp_solver, pose)
            sdls_result = self.test_near_singularity(sdls_solver, pose)
            
            qp_metrics.append(qp_result)
            sdls_metrics.append(sdls_result)
            
        return {
            'qp_performance': np.mean(qp_metrics),
            'sdls_performance': np.mean(sdls_metrics)
        }
        
    def test_near_singularity(self, solver, pose):
        # Measure manipulation index
        J = solver.calcJacobian(pose)
        U, s, _ = np.linalg.svd(J)
        manipulability = np.prod(s)
        
        # Measure solution smoothness
        _, _, rollout = solver.inverse(pose, np.zeros(7))
        smoothness = self.compute_solution_smoothness(rollout)
        
        return {
            'manipulability': manipulability,
            'smoothness': smoothness
        }


class IKBenchmark:
    def __init__(self):
        # Initialize both solvers with same parameters for fair comparison
        self.qp_solver = QpWeightedIK(
            linear_tol=1e-3,
            angular_tol=1e-2,
            max_steps=500,
            min_step_size=1e-5
        )
        self.sdls_solver = SDLSolver(
            linear_tol=1e-3,
            angular_tol=1e-2,
            max_steps=500,
            min_step_size=1e-5
        )
        self.metrics = IKMetrics()
        self.visualizer = ResultVisualizer()
        self.analyzer = ComparativeAnalysis()
        
        # Define workspace boundaries based on robot's reach
        self.ws_bounds = {
            'x': (0.2, 0.8),   # Adjusted for your robot
            'y': (-0.4, 0.4),
            'z': (0.1, 0.9)
        }


    def run_solver_tests(self, solver, test_cases: List[TestCase]) -> BenchmarkResults:
        """Run tests for a single solver on a set of test cases"""
        successes = 0
        total_time = 0
        position_errors = []
        orientation_errors = []
        manipulability_indices = []
        velocities = []
        smoothness_values = []
        
        for case in test_cases:
            try:
                # Time the solution
                start_time = time.time()
                solution, success, rollout = solver.inverse(case.pose, case.seed)
                end_time = time.time()
                
                if success:
                    successes += 1
                    total_time += (end_time - start_time)
                    
                    # Get final pose
                    _, final_pose = solver.fk.forward(solution)
                    
                    # Compute errors
                    pos_error = np.linalg.norm(final_pose[:3, 3] - case.pose[:3, 3])
                    R_error = np.matmul(final_pose[:3, :3].T, case.pose[:3, :3])
                    angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
                    
                    position_errors.append(pos_error)
                    orientation_errors.append(angle_error)
                    
                    # Compute manipulability using solver's Jacobian method
                    J = solver.calcJacobian(solution)
                    U, s, _ = np.linalg.svd(J)
                    manipulability_indices.append(np.prod(s))
                    
                    # Compute velocities and smoothness
                    if len(rollout) > 1:
                        vel = np.diff(np.array(rollout), axis=0)
                        velocities.append(vel)
                        smoothness = np.mean(np.sum(np.diff(vel, axis=0)**2, axis=1))
                        smoothness_values.append(smoothness)
                        
            except Exception as e:
                print(f"Error processing test case {case.description}: {str(e)}")
                continue
        
        # Compute average metrics
        avg_pos_error = np.mean(position_errors) if position_errors else float('inf')
        avg_orient_error = np.mean(orientation_errors) if orientation_errors else float('inf')
        avg_manip = np.mean(manipulability_indices) if manipulability_indices else 0
        avg_smoothness = np.mean(smoothness_values) if smoothness_values else float('inf')
        
        # Compute average joint velocities
        if velocities:
            avg_velocities = np.mean(np.concatenate(velocities), axis=0)
        else:
            avg_velocities = np.zeros(7)  # 7 joints
            
        return BenchmarkResults(
            success_rate=successes / len(test_cases),
            convergence_time=total_time / successes if successes > 0 else float('inf'),
            position_error=avg_pos_error,
            orientation_error=avg_orient_error,
            manipulability=avg_manip,
            joint_velocities=avg_velocities,
            smoothness=avg_smoothness
        )

        
    def generate_standard_poses(self) -> List[TestCase]:
        """Generate standard reaching poses within workspace"""
        poses = []
        # Workspace points in a grid
        for x in np.linspace(self.ws_bounds['x'][0], self.ws_bounds['x'][1], 3):
            for y in np.linspace(self.ws_bounds['y'][0], self.ws_bounds['y'][1], 3):
                for z in np.linspace(self.ws_bounds['z'][0], self.ws_bounds['z'][1], 3):
                    # Create different orientation test cases
                    orientations = [
                        np.eye(3),  # Default orientation
                        self.create_rotation_matrix(np.pi/4, 0, 0),  # Rotated around X
                        self.create_rotation_matrix(0, np.pi/4, 0),  # Rotated around Y
                        self.create_rotation_matrix(0, 0, np.pi/4)   # Rotated around Z
                    ]
                    
                    for R in orientations:
                        pose = np.eye(4)
                        pose[:3, :3] = R
                        pose[:3, 3] = [x, y, z]
                        seed = np.zeros(7)  # Default seed
                        poses.append(TestCase(pose, seed, 
                                           f"Standard pose at {x:.2f}, {y:.2f}, {z:.2f}", True))
        return poses

    def generate_singular_poses(self) -> List[TestCase]:
        """Generate poses near known singularities"""
        singular_poses = []
        
        # Wrist singularity (aligned rotational axes)
        wrist_singular = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0],
            [0, 0, 1, 0.4],
            [0, 0, 0, 1]
        ])
        singular_poses.append(TestCase(wrist_singular, np.zeros(7), 
                                     "Wrist singularity", False))
        
        # Elbow singularity (fully extended arm)
        elbow_singular = np.array([
            [1, 0, 0, 0.8],  # Maximum reach
            [0, 1, 0, 0],
            [0, 0, 1, 0.4],
            [0, 0, 0, 1]
        ])
        singular_poses.append(TestCase(elbow_singular, np.zeros(7), 
                                     "Elbow singularity", False))
        
        # Shoulder singularity (aligned first two joints)
        shoulder_singular = np.array([
            [1, 0, 0, 0.3],
            [0, 1, 0, 0],
            [0, 0, 1, 0.9],  # High position
            [0, 0, 0, 1]
        ])
        singular_poses.append(TestCase(shoulder_singular, np.zeros(7), 
                                     "Shoulder singularity", False))
        
        return singular_poses

    def generate_joint_limit_cases(self) -> List[TestCase]:
        """Generate cases near joint limits"""
        cases = []
        
        # Near upper limits
        q_upper = self.qp_solver.upper - 0.1
        _, T_upper = self.qp_solver.fk.forward(q_upper)
        cases.append(TestCase(T_upper, q_upper, "Near upper limits", True))
        
        # Near lower limits
        q_lower = self.qp_solver.lower + 0.1
        _, T_lower = self.qp_solver.fk.forward(q_lower)
        cases.append(TestCase(T_lower, q_lower, "Near lower limits", True))
        
        # Mixed limits
        q_mixed = np.zeros(7)
        q_mixed[::2] = self.qp_solver.upper[::2] - 0.1  # Even indices near upper
        q_mixed[1::2] = self.qp_solver.lower[1::2] + 0.1  # Odd indices near lower
        _, T_mixed = self.qp_solver.fk.forward(q_mixed)
        cases.append(TestCase(T_mixed, q_mixed, "Mixed joint limits", True))
        
        return cases

    def generate_trajectory_cases(self) -> List[TestCase]:
        """Generate trajectory test cases"""
        cases = []
        
        # Circular trajectory in XY plane
        center = np.array([0.5, 0, 0.5])
        radius = 0.2
        points = 20
        for t in np.linspace(0, 2*np.pi, points):
            pose = np.eye(4)
            pose[:3, 3] = center + radius * np.array([np.cos(t), np.sin(t), 0])
            seed = np.zeros(7)
            cases.append(TestCase(pose, seed, f"Circle trajectory {t:.2f}", True))
            
        # Linear trajectory with rotation
        start = np.array([0.3, -0.2, 0.5])
        end = np.array([0.6, 0.2, 0.5])
        for t in np.linspace(0, 1, 10):
            pose = np.eye(4)
            pose[:3, 3] = start + t * (end - start)
            # Add rotation around Z axis
            angle = t * np.pi/2
            pose[:3, :3] = self.create_rotation_matrix(0, 0, angle)
            seed = np.zeros(7)
            cases.append(TestCase(pose, seed, f"Linear trajectory {t:.2f}", True))
            
        return cases

    @staticmethod
    def create_rotation_matrix(rx, ry, rz):
        """Create rotation matrix from Euler angles"""
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(rx), -np.sin(rx)],
                      [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                      [np.sin(rz), np.cos(rz), 0],
                      [0, 0, 1]])
        
        return Rz @ Ry @ Rx


class ResultVisualizer:
    def plot_comparison(self, qp_results: BenchmarkResults, sdls_results: BenchmarkResults, test_type: str):
        """Create comprehensive comparison plots"""
        plt.figure(figsize=(15, 10))
        
        # Success rate comparison
        plt.subplot(231)
        self.plot_bar_comparison(['QP', 'SDLS'],
                               [qp_results.success_rate, sdls_results.success_rate],
                               'Success Rate', test_type)
        
        # Convergence time comparison
        plt.subplot(232)
        self.plot_bar_comparison(['QP', 'SDLS'],
                               [qp_results.convergence_time, sdls_results.convergence_time],
                               'Average Convergence Time (s)', test_type)
        
        # Error comparison
        plt.subplot(233)
        self.plot_error_comparison(qp_results, sdls_results, test_type)
        
        # Manipulability comparison
        plt.subplot(234)
        self.plot_bar_comparison(['QP', 'SDLS'],
                               [qp_results.manipulability, sdls_results.manipulability],
                               'Average Manipulability Index', test_type)
        
        # Joint velocity profiles
        plt.subplot(235)
        self.plot_velocity_profiles(qp_results.joint_velocities, sdls_results.joint_velocities)
        
        # Smoothness comparison
        plt.subplot(236)
        self.plot_bar_comparison(['QP', 'SDLS'],
                               [qp_results.smoothness, sdls_results.smoothness],
                               'Motion Smoothness (lower is better)', test_type)
        
        plt.tight_layout()
        plt.savefig(f'comparison_{test_type}.png')
        plt.close()

    @staticmethod
    def plot_bar_comparison(labels: List[str], values: List[float], title: str, test_type: str):
        plt.bar(labels, values)
        plt.title(f'{title}\n({test_type} cases)')
        plt.grid(True)

    def plot_error_comparison(self, qp_results: BenchmarkResults, sdls_results: BenchmarkResults, test_type: str):
        labels = ['Position Error', 'Orientation Error']
        qp_errors = [qp_results.position_error, qp_results.orientation_error]
        sdls_errors = [sdls_results.position_error, sdls_results.orientation_error]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, qp_errors, width, label='QP')
        plt.bar(x + width/2, sdls_errors, width, label='SDLS')
        plt.title(f'Error Comparison\n({test_type} cases)')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True)

    def plot_velocity_profiles(self, qp_velocities: np.ndarray, sdls_velocities: np.ndarray):
        plt.plot(qp_velocities, 'b-', label='QP')
        plt.plot(sdls_velocities, 'r--', label='SDLS')
        plt.title('Joint Velocity Profiles')
        plt.xlabel('Time step')
        plt.ylabel('Joint velocity')
        plt.legend()
        plt.grid(True)


def main():
    # Create benchmark instance
    benchmark = IKBenchmark()
    
    # Dictionary mapping category names to method names
    category_methods = {
        'standard': 'generate_standard_poses',
        'singular': 'generate_singular_poses',
        'joint_limit': 'generate_joint_limit_cases',
        'trajectory': 'generate_trajectory_cases'
    }
    
    results = {}
    for category, method_name in category_methods.items():
        print(f"\nRunning {category} tests...")
        
        try:
            # Get test cases using the correct method name
            test_cases = getattr(benchmark, method_name)()
            
            # Run benchmarks
            qp_results = benchmark.run_solver_tests(benchmark.qp_solver, test_cases)
            sdls_results = benchmark.run_solver_tests(benchmark.sdls_solver, test_cases)
            
            # Store results
            results[category] = {
                'qp': qp_results,
                'sdls': sdls_results
            }
            
            # Print summary for this category
            print(f"\n{category.upper()} TEST RESULTS:")
            print(f"QP Success Rate: {qp_results.success_rate:.2%}")
            print(f"SDLS Success Rate: {sdls_results.success_rate:.2%}")
            print(f"QP Average Time: {qp_results.convergence_time:.4f}s")
            print(f"SDLS Average Time: {sdls_results.convergence_time:.4f}s")
            print(f"QP Position Error: {qp_results.position_error:.6f}")
            print(f"SDLS Position Error: {sdls_results.position_error:.6f}")
            
            # Create visualization
            benchmark.visualizer.plot_comparison(qp_results, sdls_results, category)
            
        except AttributeError as e:
            print(f"Error: Could not find test generator for {category}: {e}")
        except Exception as e:
            print(f"Error running tests for {category}: {e}")
    
    return results

if __name__ == "__main__":
    main()

