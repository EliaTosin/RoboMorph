
"""
https://github.com/rail-berkeley/serl_franka_controllers/blob/main/src/cartesian_impedance_controller.cpp
"""

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *

def orientation_error(desired, current):
    """
                                            QUAT = [i, j, k, w]
    cc = quat_conjugate(current) # inverte le parti immaginarie (i,j,k) e mantiene la parte reale (w)
    q_r = quat_mul(desired, cc) # la moltiplicazione tra conj e desired mi dice quanto manca (errore negli assi) per arrivare al desidered (da current)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1) # tira fuori l'errore sui 3 assi. Poi usa 4⁰ componente per aggiustare segno
    """
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

class PIDController:
    """
        This class implements a PID-based Cartesian impedance controller
        for robotic end-effectors, generating joint torques based on
        position and orientation errors, stiffness/damping gains, and
        torque saturation limits.
    """

    def __init__(self, stiffness, damping, ki, j_eef, lower, upper, dt, device):
        """
            Constructor for PIDController.

            Purpose:
            - Initialize control gains (stiffness, damping, integral gain).
            - Store Jacobian matrix for end-effector control.
            - Define torque limits to prevent unsafe values.
            - Prepare tensors for position, orientation, and accumulated error tracking.
            - Set control timestep (dt) and allocate tensors on the specified device.

            Parameters:
            - stiffness: Proportional gain for Cartesian control.
            - damping: Derivative gain for Cartesian control.
            - ki: Integral gain for error compensation.
            - j_eef: End-effector Jacobian for all environments.
            - lower: Lower torque limits per joint.
            - upper: Upper torque limits per joint.
            - dt: Control loop time step.
            - device: Compute device (e.g., "cuda" or "cpu").
        """
        self.stiffness = stiffness
        self.damping = damping
        self.ki = ki
        self.j_eef = j_eef
        self.device = device
        self.torque_limit_lower = torch.tensor(lower, device=device)
        self.torque_limit_upper = torch.tensor(upper, device=device)

        num_envs = j_eef.shape[0]
        self.pos = torch.zeros((num_envs, 3), device=device)
        self.pos_target = torch.zeros_like(self.pos)
        self.orient = torch.zeros((num_envs, 3), device=device)
        self.orient_target = torch.zeros_like(self.orient)
        self.error_sum = torch.zeros((num_envs, 6), device=device)
        self.dt = dt

    def start(self, pos, des_pos, orn, orn_des, vel):
        """
            Initialize the controller state with the current and desired poses.

            Purpose:
            - Set the current and desired positions and orientations.
            - Initialize the joint velocity state for derivative control.

            Parameters:
            - pos: Current Cartesian position of the end-effector.
            - des_pos: Desired Cartesian position of the end-effector.
            - orn: Current orientation of the end-effector (e.g., Euler or quaternion).
            - orn_des: Desired orientation of the end-effector.
            - vel: Current joint velocity vector.
        """
        self.pos = pos
        self.pos_target = des_pos
        self.orient = orn
        self.orient_target = orn_des
        self.dof_vel = vel

    def update(self):
        """
            Compute the control command (joint torques) based on PID logic in Cartesian space.

            Purpose:
            - Calculate position and orientation errors.
            - Formulate the error vector for Cartesian space (6 DoF).
            - Update the integral error term.
            - Compute control effort in task space using stiffness, damping, and integral gains.
            - Map Cartesian control forces to joint torques using the Jacobian transpose.

            Returns:
            - u: Tensor of joint torques for each environment.
        """
        pos_err = self.pos_target - self.pos
        orn_err = orientation_error(self.orient_target, self.orient)
        dpose = torch.cat([pos_err, orn_err], dim=-1)
        self.error_sum[:, :3] += pos_err
        self.error_sum[:, 3:] += orn_err

        j_eef_T = torch.transpose(self.j_eef, 1, 2)

        u = j_eef_T @ (self.stiffness * dpose.unsqueeze(-1) - self.damping * (self.j_eef @ self.dof_vel) + (self.ki * self.error_sum * self.dt).unsqueeze(-1))
        return u


    def saturated_torque(self, actual_torque):
        """
            Apply torque saturation and log any violations of torque limits.

            Purpose:
            - Clamp computed torques to predefined safe ranges for each joint.
            - Identify and report joints that exceeded lower or upper torque thresholds.

            Parameters:
            - actual_torque: Computed joint torque tensor before saturation.

            Returns:
            - Saturated torque tensor within the specified limits.
        """
        actual_torque = actual_torque.squeeze(-1)
        mask_lower = actual_torque < self.torque_limit_lower
        mask_upper = actual_torque > self.torque_limit_upper

        env_idxs, joint_idxs = torch.where(mask_lower)
        for env, joint in zip(env_idxs, joint_idxs):
            print(f"Env {env}, Joint {joint} has exceeded the minimum limit of {self.torque_limit_lower[joint]}: {actual_torque[env, joint]}")

        env_idxs, joint_idxs = torch.where(mask_upper)
        for env, joint in zip(env_idxs, joint_idxs):
            print(f"Env {env}, Joint {joint} has exceeded the maximum limit of {self.torque_limit_upper[joint]}: {actual_torque[env, joint]}")

        return torch.clamp(actual_torque, self.torque_limit_lower, self.torque_limit_upper).unsqueeze(-1)