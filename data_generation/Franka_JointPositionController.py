import torch
from isaacgym import gymtorch

class JointPositionController:
    """
        This class implements a simple joint position controller for simulations
        using Isaac Gym. It allows initializing joint states, setting a target trajectory,
        and updating positions over time with interpolation.
    """

    def __init__(self, gym, sim, asset, num_envs, device, dt, max_time_s):
        """
            Constructor for the JointPositionController class.

            Purpose:
            - Validate the provided simulation interfaces (`gym`, `sim`, and `asset`).
            - Retrieve joint names from the robot asset and check that the number matches expectations.
            - Initialize tensors for DOF (Degrees of Freedom) states and position targets.
            - Store timing parameters for controlling motion (time step and maximum time for movement).

            Parameters:
            - gym: The main Isaac Gym interface.
            - sim: Handle to the simulation instance.
            - asset: The loaded robot asset (URDF or equivalent).
            - num_envs: Number of environments being simulated.
            - device: Compute device for tensors (e.g., "cuda:0" or "cpu").
            - dt: Simulation time step in seconds.
            - max_time_s: Maximum allowed time to reach the target position.
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
        self.dt = dt
        self.max_time_s = max_time_s

    def start(self, target_pos):
        """
            Initialize the control process with a given target position.

            Purpose:
            - Acquire the current DOF states (positions) from the simulator.
            - Store the target positions for each DOF in all environments.
            - Reset the elapsed time counter for the interpolation process.

            Details:
            - The DOF state tensor returned by Isaac Gym includes positions and velocities; this method extracts only positions for the interpolation.
            - The target positions (`target_pos`) should be provided as a tensor with shape (num_envs, number_of_joints).

            Parameters:
            - target_pos: Tensor of desired joint positions for all environments.
        """
        # taking the first [:,:,0] because "acquire_dof_state_tensor" retrieves both positions and velocities
        self.dof_states = gymtorch.wrap_tensor(self.gym_interface.acquire_dof_state_tensor(self.sim)).view(self.num_envs, -1, 2)[:, :, 0]

        self.dof_targets = target_pos

        self.elapsed_time = 0

    def update(self):
        """
            Update the joint positions toward the target over time.

            Purpose:
            - Progressively move the joints from the initial position toward the target using linear interpolation over the specified maximum time.
            - After reaching `max_time_s`, apply the target position directly.
            - Simulate one step of the environment after setting new targets.

            Logic:
            - Compute alpha = elapsed_time / max_time_s to define interpolation ratio.
            - Calculate intermediate positions as a weighted sum of initial and target positions.
            - Send the computed positions to the simulator.
            - Step the simulation forward by one iteration.

            Behavior:
            - Before `max_time_s`, motion is smooth and interpolated.
            - After `max_time_s`, positions are fixed to the final target.

            No parameters are needed; it operates on the stored state.
        """
        self.elapsed_time += self.dt

        if self.elapsed_time > self.max_time_s:
            self.gym_interface.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_targets))
        else:
            alpha = self.elapsed_time / self.max_time_s
            intermediate_dof_pos = alpha * self.dof_targets + (self.max_time_s-alpha) * self.dof_states
            self.gym_interface.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(intermediate_dof_pos))

        self.gym_interface.simulate(self.sim)
        self.gym_interface.fetch_results(self.sim, True)