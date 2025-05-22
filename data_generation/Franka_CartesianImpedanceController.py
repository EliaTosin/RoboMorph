
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

class CartesianImpedanceController:

    def __init__(self, stiffness, damping, j_eef, lower, upper, device):
        """
            init
                stiffness,
                damping,
                joint names,
                posiz,
                orientamento (quaternione)
                posiz_target,
                orientamento_target

        """
        self.stiffness = stiffness
        self.damping = damping
        self.j_eef = j_eef
        self.device = device
        self.torque_limit_lower = torch.tensor(lower, device=device)
        self.torque_limit_upper = torch.tensor(upper, device=device)

        num_envs = j_eef.shape[0]
        self.pos = torch.zeros((num_envs, 3), device=device)
        self.pos_target = torch.zeros_like(self.pos)
        self.orient = torch.zeros((num_envs, 3), device=device)
        self.orient_target = torch.zeros_like(self.orient)

    def start(self, pos, des_pos, orn, orn_des, vel):
        """
            init
                 posiz
                 orientamento
                 posiz_target
                 orientamento_target
        """
        self.pos = pos
        self.pos_target = des_pos
        self.orient = orn
        self.orient_target = orn_des
        self.dof_vel = vel

    def update(self):
        """
            calcola errore posiz dell'eef
            calcola errore orientamento dell'eef

            calcola jacob trasposta

            calcola torque per la task tramite
                trasposta_jeef * (stiffness - dpose - damping * (jeef * velocita_joints)

        """
        pos_err = self.pos_target - self.pos
        orn_err = orientation_error(self.orient_target, self.orient)
        dpose = torch.cat([pos_err, orn_err], dim=-1)

        j_eef_T = torch.transpose(self.j_eef, 1, 2)

        u = j_eef_T @ (self.stiffness * dpose.unsqueeze(-1) - self.damping * (self.j_eef @ self.dof_vel))
        return u


    def saturated_torque(self, actual_torque):
        """
            limita il torque da applicare sui giunti tagliandolo entro i limiti
        """
        actual_torque = actual_torque.squeeze(-1)
        mask_lower = actual_torque < self.torque_limit_lower
        mask_upper = actual_torque > self.torque_limit_upper

        env_idxs, joint_idxs = torch.where(mask_lower)
        for env, joint in zip(env_idxs, joint_idxs):
            print(f"Env {env}, Joint {joint} ha superato la soglia minima di {self.torque_limit_lower[joint]}: {actual_torque[env, joint]}")

        env_idxs, joint_idxs = torch.where(mask_upper)
        for env, joint in zip(env_idxs, joint_idxs):
            print(f"Env {env}, Joint {joint} ha superato la soglia massima di {self.torque_limit_upper[joint]}: {actual_torque[env, joint]}")

        return torch.clamp(actual_torque, self.torque_limit_lower, self.torque_limit_upper).unsqueeze(-1)