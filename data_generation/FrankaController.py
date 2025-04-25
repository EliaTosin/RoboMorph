
"""
In ''generation_franka.py'' usa

set_dof_actuation_force_tensor --> COMANDO IN FORZA

-----------------------------------------------------

Qui si fa controllo in posizione. Si può usare

set_dof_position_target_tensor

"""

import torch
from isaacgym import gymtorch


class JointPositionController:

    def __init__(self, gym, sim, asset, target_pos, num_envs, device):
        """
            salvare gym controller in un attributo locale

            salvare nomi giunti

            inizializzare initial pose e time

            inizializzare target (num_envs, num_joints)

            salvare posizioni default
        """
        self.gym_interface = gym
        if self.gym_interface is None:
            raise ValueError("JointPositionController: Error getting gym interface from hardware!")

        self.sim = sim
        if self.sim is None:
            raise ValueError("JointPositionController: Error getting sim interface from hardware!")

        self.asset = asset
        if self.asset is None:
            raise ValueError("JointPositionController: Error loading asset from gym!")

        self.joint_names = gym.get_asset_dof_names(asset)
        if self.joint_names is None:
            raise ValueError("JointPositionController: Could not parse joint names from gym!")

        if len(self.joint_names) != 9: #7 joints + 2 fingers
            raise ValueError(f"JointPositionController: Wrong number of joint names, got {len(self.joint_names)} instead of 9 names from gym!")

        self.num_envs = num_envs
        if not self.num_envs:
            raise ValueError(f"JointPositionController: Wrong number of environments, got {self.num_envs}!")

        self.device = device
        self.dof_states = torch.zeros((self.num_envs, len(self.joint_names)), device=self.device)

        self.dof_targets = torch.zeros((self.num_envs, len(self.joint_names)), device=self.device)

        self.elapsed_time = 0

    def start(self):
        """
            settaggio initial pose con pose attuali

            settaggio time a 0
        """
        # taking the first [:,:,0] because "acquire_dof_state_tensor" retrieves both positions and velocities
        self.dof_states = gymtorch.wrap_tensor(self.gym_interface.acquire_dof_state_tensor(self.sim)).view(self.num_envs, -1, 2)[:, :, 0]
        self.elapsed_time = 0

    def update(self):
        """
            incrementa time

            per ogni giunto
                se time > 10:
                    reset a posiz default
                else:
                    set pos desiderata

        """