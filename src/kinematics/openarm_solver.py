import jax
import jax.numpy as jnp
import jaxls
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy

from typing import Any
Cost = Any

class OpenArmIK:
    def __init__(
        self,
        urdf_path: str | None = None,
        locked_joint_names: set[str] | list[str] | tuple[str, ...] | None = None,
        left_ee: str | None = None,
        right_ee: str | None = None,
    ) -> None:
        # Load urdf and robot
        urdf = yourdfpy.URDF.load(urdf_path)
        self.robot: pk.Robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        # Get end-effector link indices
        self.L_ee = left_ee or "openarm_left_link7"
        self.R_ee = right_ee or "openarm_right_link7"
        if self.L_ee in self.robot.links.names:
            self.L_ee_link_idx = self.robot.links.names.index(self.L_ee)
        else:
            raise ValueError(f"Link {self.L_ee} not found in URDF")
        if self.R_ee in self.robot.links.names:
            self.R_ee_link_idx = self.robot.links.names.index(self.R_ee)
        else:
            raise ValueError(f"Link {self.R_ee} not found in URDF")

        self._joint_names = list(self.robot.joints.actuated_names)
        self._qidx = {name: i for i, name in enumerate(self._joint_names)}
        self._lower_limits = np.asarray(self.robot.joints.lower_limits, dtype=float)
        self._upper_limits = np.asarray(self.robot.joints.upper_limits, dtype=float)
        self._revolute_joint_indices = [
            i
            for i, name in enumerate(self._joint_names)
            if urdf.joint_map[name].type in {"revolute", "continuous"}
        ]
        locked_joint_names = set(locked_joint_names or [])
        unknown_locked = locked_joint_names.difference(self._joint_names)
        if unknown_locked:
            raise ValueError(f"Locked joint names not found in URDF: {sorted(unknown_locked)}")
        self._locked_joint_indices = [self._qidx[name] for name in locked_joint_names]
        self.q = np.asarray(self.get_default_config(), dtype=float)

        self._jit_solve = jax.jit(self._solve_internal)

    def forward_kinematics(self, config: jax.Array) -> dict[str, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(config)
        return {
            "left": jaxlie.SE3(fk[self.L_ee_link_idx]),
            "right": jaxlie.SE3(fk[self.R_ee_link_idx]),
        }

    def get_default_config(self) -> jax.Array:
        joint_names = list(self.robot.joints.actuated_names)
        default_pose = {
            "openarm_left_joint1": 0.119211,
            "openarm_left_joint2": -0.063134,
            "openarm_left_joint3": 0.021172,
            "openarm_left_joint4": 0.710880,
            "openarm_left_joint5": -0.102808,
            "openarm_left_joint6": 0.026131,
            "openarm_left_joint7": -1.086252,
            "openarm_right_joint1": 0.0,
            "openarm_right_joint2": 0.0,
            "openarm_right_joint3": 0.0,
            "openarm_right_joint4": 0.0,
            "openarm_right_joint5": 0.0,
            "openarm_right_joint6": 0.0,
            "openarm_right_joint7": 0.0,
            "openarm_left_finger_joint1": 0.0,
            "openarm_left_finger_joint2": 0.0,
            "openarm_right_finger_joint1": 0.0,
            "openarm_right_finger_joint2": 0.0,
        }
        config_list = [default_pose.get(name, 0.0) for name in joint_names]
        return jnp.array(config_list)

    def set_q(self, q: jax.Array | np.ndarray) -> None:
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        if q_arr.shape[0] != len(self._joint_names):
            raise ValueError("q length does not match the number of actuated joints")
        self.q = q_arr.copy()

    def get_T_tip(self) -> np.ndarray:
        t_curr_se3 = self.forward_kinematics(jnp.asarray(self.q))["left"]
        t_curr = np.eye(4, dtype=float)
        t_curr[:3, :3] = np.asarray(t_curr_se3.rotation().as_matrix(), dtype=float)
        t_curr[:3, 3] = np.asarray(t_curr_se3.translation(), dtype=float)
        return t_curr

    @staticmethod
    def _closest_equivalent_within_limits(
        angle: float,
        reference: float,
        lower: float,
        upper: float,
    ) -> float:
        tau = 2.0 * np.pi
        k_nearest = np.round((reference - angle) / tau)
        k_min = np.ceil((lower - angle) / tau)
        k_max = np.floor((upper - angle) / tau)
        if k_min <= k_max:
            k = np.clip(k_nearest, k_min, k_max)
            return float(angle + tau * k)
        return float(angle + tau * k_nearest)

    def _stabilize_solution(
        self,
        solution: jax.Array | np.ndarray,
        q_current: jax.Array | np.ndarray,
    ) -> jnp.ndarray:
        solution_np = np.asarray(solution, dtype=float).reshape(-1).copy()
        q_current_np = np.asarray(q_current, dtype=float).reshape(-1)
        for idx in self._revolute_joint_indices:
            solution_np[idx] = self._closest_equivalent_within_limits(
                angle=solution_np[idx],
                reference=q_current_np[idx],
                lower=self._lower_limits[idx],
                upper=self._upper_limits[idx],
            )
        if self._locked_joint_indices:
            solution_np[self._locked_joint_indices] = q_current_np[self._locked_joint_indices]
        return jnp.asarray(solution_np)

    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        q_current: jnp.ndarray | None = None,
    ) -> list[Cost]:
        """
        Build the cost functions for the IK problem.
        costs:
            - Rest cost: penalize the deviation from the current configuration.
            - Manipulability cost: penalize the manipulability of the end-effectors.
            - Pose cost: penalize the deviation from the target pose.
            - Limit cost: penalize the joint limits.
            - Self-collision cost: penalize the self-collisions.
        """
        costs = []
        JointVar = self.robot.joint_var_cls

        joint_mask = jnp.ones(len(self._joint_names))
        if self._locked_joint_indices:
            joint_mask = joint_mask.at[jnp.array(self._locked_joint_indices)].set(0.0)

        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=1.0,
                )
            )
            if self._locked_joint_indices:
                lock_weight = jnp.zeros_like(q_current)
                lock_weight = lock_weight.at[jnp.array(self._locked_joint_indices)].set(1e4)
                costs.append(
                    pk.costs.rest_cost(
                        JointVar(0),
                        rest_pose=q_current,
                        weight=lock_weight,
                    )
                )

        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.L_ee_link_idx, self.R_ee_link_idx], dtype=jnp.int32),
                weight=0.0002,
            )
        )

        if target_L is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_L,
                    jnp.array(self.L_ee_link_idx, dtype=jnp.int32),
                    pos_weight=2000.0,
                    ori_weight=1000.0,
                    joint_mask=joint_mask,
                )
            )

        if target_R is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_R,
                    jnp.array(self.R_ee_link_idx, dtype=jnp.int32),
                    pos_weight=2000.0,
                    ori_weight=1000.0,
                    joint_mask=joint_mask,
                )
            )

        costs.append(pk.costs.limit_cost(self.robot, JointVar(0), weight=20.0))

        costs.append(
            pk.costs.self_collision_cost(
                self.robot,
                self.robot_coll,
                JointVar(0),
                margin=0.05,
                weight=2.0,
            )
        )

        return costs

    def _solve_internal(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        q_current: jnp.ndarray,
    ) -> jnp.ndarray:
        # Build the cost functions for the IK problem.
        costs = self.build_costs(target_L, target_R, q_current)

        # Create the variable for the joint angles.
        var_joints = self.robot.joint_var_cls(jnp.array([0]))
        initial_vals = jaxls.VarValues.make([var_joints.with_value(q_current[jnp.newaxis, :])])
        
        # Solve the IK problem.
        problem = jaxls.LeastSquaresProblem(costs, [var_joints])
        solution = problem.analyze().solve(
            initial_vals=initial_vals,
            verbose=False,
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(max_iterations=200),
        )
        return solution[var_joints][0]

    def solve(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        q_current: jnp.ndarray,
    ) -> jnp.ndarray:
        solution = self._jit_solve(target_L, target_R, q_current)
        return self._stabilize_solution(solution, q_current)
