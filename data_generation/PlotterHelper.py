import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec


class Plotter():
    def __init__(self, num_envs):
        self.desired_poses = torch.zeros((1, num_envs, 3))
        self.actual_poses = torch.zeros((1, num_envs, 3))
        self.joint_poses = torch.zeros((1, num_envs, 7))
        self.pos_errors = torch.zeros((1, num_envs, 3))

    def add_desired_pose(self, pos_des):
        self.desired_poses = torch.cat((self.desired_poses, pos_des.cpu().unsqueeze(0)), dim=0)

    def add_actual_pose(self, pos_cur):
        self.actual_poses = torch.cat((self.actual_poses, pos_cur.cpu().unsqueeze(0)), dim=0)

    def add_joint_pose(self, joint_pos):
        self.joint_poses = torch.cat((self.joint_poses, joint_pos.cpu().unsqueeze(0)), dim=0)

    def add_pos_errors(self, pos_error):
        self.pos_errors = torch.cat((self.pos_errors, pos_error.cpu().unsqueeze(0)), dim=0)

    def plot(self):
        axis_labels = ["x", "y", "z"]
        colors = ["red", "green", "blue"]
        dof_labels = [f"j{i}" for i in range(7)]

        for env_idx in range(self.desired_poses.shape[1]):

            fig = plt.figure(figsize=(12, 8))  # altezza aumentata
            fig.suptitle(f"Plotting env {env_idx}")

            # Griglia personalizzata: 3 righe, con la seconda che occupa più spazio
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3, 1])

            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)

            for i, color in enumerate(colors):
                ax1.plot(self.desired_poses[1:, env_idx, i].tolist(), label=axis_labels[i] + " target", color=color)
                ax1.plot(self.actual_poses[1:, env_idx, i].tolist(), alpha=0.5, linestyle='--', label=axis_labels[i] + " actual", color=color)
            ax1.set_title("Target Position vs Actual (dotted)")
            ax1.set_ylabel('Position (world coord)')
            ax1.legend(loc='right')
            ax1.grid(True)

            ax2.plot(self.joint_poses[1:, env_idx, :].tolist(), label=dof_labels)
            ax2.set_title("Joint Poses")
            ax2.legend(loc='right')
            ax2.set_ylabel('Joint positions')
            ax2.grid(True)

            for i, color in enumerate(colors):
                ax3.plot(self.pos_errors[1:, env_idx, i].tolist(), label=axis_labels[i], color=color)
            ax3.set_title("Position Error")
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Error')
            ax3.legend(loc='right')
            ax3.grid(True)

            plt.tight_layout()
            plt.show()