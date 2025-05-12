import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider


class Plotter():
    def __init__(self, num_envs, dof_upper_limits, dof_lower_limits):
        self.desired_poses = torch.zeros((1, num_envs, 3))
        self.actual_poses = torch.zeros((1, num_envs, 3))

        self.joint_poses = torch.zeros((1, num_envs, 7))
        self.joint_target = torch.zeros((1, num_envs, 7))

        self.dof_upper_limits = dof_upper_limits[:7]
        self.dof_lower_limits = dof_lower_limits[:7]

        self.dof_init_pos = torch.zeros((1, 7))

        #plot related variables
        self.axis_labels = ["x", "y", "z"]
        self.colors = ["red", "green", "blue"]
        self.dof_labels = [f"j{i}" for i in range(len(dof_upper_limits))]
        self.dof_colors = ['dodgerblue', 'limegreen', 'tomato', 'magenta', 'darkorange', 'chocolate', 'gold']

        self.fig, self.axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        self.fig.subplots_adjust(left=0.05, right=0.85, wspace=0.4, hspace=0.5)
        self.axes = self.axes.flatten()  # Flatten to easily iterate over

    def add_desired_pose(self, pos_des):
        """Concatenate the desired pose of the EEF in cartesian coordinates"""
        self.desired_poses = torch.cat((self.desired_poses, pos_des.cpu().unsqueeze(0)), dim=0)

    def add_actual_pose(self, pos_cur):
        """Concatenate the actual pose of the EEF in cartesian coordinates"""
        self.actual_poses = torch.cat((self.actual_poses, pos_cur.cpu().unsqueeze(0)), dim=0)

    def add_joint_pose(self, joint_pos):
        """Concatenate the actual position of the joints"""
        self.joint_poses = torch.cat((self.joint_poses, joint_pos.cpu().unsqueeze(0)), dim=0)

    def add_joint_target(self, joint_pos):
        """Concatenate the desired position of the joints (command)"""
        self.joint_target = torch.cat((self.joint_target, joint_pos.cpu().unsqueeze(0)), dim=0)

    def add_joint_init_pos(self, init_pos):
        """Set the initial position of the joints (for each env)"""
        self.dof_init_pos = torch.cat((self.dof_init_pos, torch.tensor(init_pos).unsqueeze(0)), dim=0)

    def plot(self):
        axis_labels = ["x", "y", "z"]
        colors = ["red", "green", "blue"]
        dof_labels = [f"j{i}" for i in range(len(self.dof_upper_limits))]
        dof_colors = ['dodgerblue', 'limegreen', 'tomato', 'magenta', 'darkorange', 'chocolate', 'gold']

        for env_idx in range(self.desired_poses.shape[1]):

            fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)

            dof_inits = self.dof_init_pos[env_idx + 1, :].tolist() # skipping the first env (since it's zeros due to init)
            dof_pretty = [f"{label}: {init_value:.2f}" for label, init_value in zip(dof_labels, dof_inits)]
            fig.suptitle(f"Plotting env {env_idx} \n dof init: {dof_pretty}")
            axes = axes.flatten()

            # 1. Posizione target vs actual
            ax_pos = axes[0]
            for i, color in enumerate(colors):
                ax_pos.plot(self.desired_poses[1:, env_idx, i].tolist(), label=axis_labels[i] + " target", color=color, linestyle='dotted')
                ax_pos.plot(self.actual_poses[1:, env_idx, i].tolist(), alpha=0.5, label=axis_labels[i] + " actual", color=color)
            ax_pos.set_title("Target vs Actual Position")
            ax_pos.set_ylabel('Position')
            ax_pos.grid(True)
            ax_pos.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

            # 2. Giunti singoli
            for i in range(1, 8):  # 7 giunti nei subplot 1–7
                ax = axes[i]
                joint_data = self.joint_poses[1:, env_idx, i - 1]
                joint_target = self.joint_target[1:, env_idx, i - 1]
                color = dof_colors[i - 1]
                upper = self.dof_upper_limits[i - 1]
                lower = self.dof_lower_limits[i - 1]

                ax.plot(joint_data.tolist(), label=dof_labels[i - 1], color=color)
                ax.plot(joint_target.tolist(), label=f"{dof_labels[i - 1]} target", color="black", linestyle='dotted')
                ax.axhline(upper, linestyle="--", color=color, alpha=0.6)
                ax.axhline(lower, linestyle="--", color=color, alpha=0.6)

                ax.set_title(f"Joint {i - 1} [min: {lower:.2f}, max: {upper:.2f}]")
                ax.set_ylabel('Position')
                ax.grid(True)
                ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

                # Etichetta asse x solo per subplot in ultima riga (assumi 4x2 layout)
                if i in [6, 7]:
                    ax.set_xlabel("Timesteps")

            # Disattiva eventuale subplot extra
            if len(axes) > 8:
                for ax in axes[8:]:
                    ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])  # lascia spazio per il titolo

            plt.show()

    def update_plot(self, env_idx):
        # Cancella il contenuto precedente dei grafici
        for ax in self.axes:
            for line in ax.lines:
                line.remove()

        # Dati per l'ambiente specifico
        dof_inits = self.dof_init_pos[env_idx + 1, :].tolist()  # skipping the first env (zeros due to init)
        dof_pretty = [f"{label}: {init_value:.2f}" for label, init_value in zip(self.dof_labels, dof_inits)]
        self.fig.suptitle(f"Plotting env {env_idx} \n dof init: {dof_pretty}")

        # 1. Posizione target vs actual
        ax_pos = self.axes[0]
        for i, color in enumerate(self.colors):
            ax_pos.plot(self.desired_poses[1:, env_idx, i].tolist(), label=self.axis_labels[i] + " target", color=color, linestyle='dotted')
            ax_pos.plot(self.actual_poses[1:, env_idx, i].tolist(), alpha=0.5, label=self.axis_labels[i] + " actual", color=color)
        ax_pos.set_title("Target vs Actual Position")
        ax_pos.set_ylabel('Position')
        ax_pos.grid(True)
        ax_pos.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

        # 2. Giunti singoli
        for i in range(1, 8):  # 7 giunti nei subplot 1–7
            ax = self.axes[i]
            joint_data = self.joint_poses[1:, env_idx, i - 1]
            joint_target = self.joint_target[1:, env_idx, i - 1]
            color = self.dof_colors[i - 1]
            upper = self.dof_upper_limits[i - 1]
            lower = self.dof_lower_limits[i - 1]

            ax.plot(joint_data.tolist(), label=self.dof_labels[i - 1], color=color)
            ax.plot(joint_target.tolist(), label=f"{self.dof_labels[i - 1]} target", color="black", linestyle='dotted')
            ax.axhline(upper, linestyle="--", color=color, alpha=0.6)
            ax.axhline(lower, linestyle="--", color=color, alpha=0.6)

            ax.set_title(f"Joint {i - 1} [min: {lower:.2f}, max: {upper:.2f}]")
            ax.set_ylabel('Position')
            ax.grid(True)
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

            # Etichetta asse x solo per subplot in ultima riga (assumi 4x2 layout)
            if i in [6, 7]:
                ax.set_xlabel("Timesteps")

            ax.autoscale(enable=True, axis='y')

        # Disattiva eventuale subplot extra
        if len(self.axes) > 8:
            for ax in self.axes[8:]:
                ax.axis('off')

        # Aggiorna il layout
        plt.draw()

    def plot_with_slider(self):
        # Crea uno slider in cima per cambiare l'ambiente
        ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor='lightgoldenrodyellow')

        slider = Slider(ax_slider, 'Env Index', 0, self.desired_poses.shape[1] - 1, valinit=0, valstep=1)

        # Collega la funzione di aggiornamento al cambiamento dello slider
        slider.on_changed(self.update_plot)

        # Inizializza il grafico con il primo ambiente
        self.update_plot(0)

        plt.show()