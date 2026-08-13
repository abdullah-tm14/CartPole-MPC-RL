import argparse
import os
import pickle
import torch
import genesis as gs
import numpy as np
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout, QInputDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

from anymal_env import QuadV3
from performance_plots import save_velocity_tracking_plot
from rsl_rl.runners import OnPolicyRunner


class TeleopUI(QWidget):
    def __init__(self, env, policy, plot_dir):
        super().__init__()
        self.env = env
        self.policy = policy
        self.num_envs = env.num_envs
        
        self.max_torques = self.env.env_cfg.get("max_torques", [80.0] * self.env.num_actions)
        
        self.peak_torques = np.zeros(self.env.num_actions)
        
        self.lin_x = 0.0
        self.lin_y = 0.0
        self.ang_z = 0.0
        self.stand_mode = False
        self.manual_reset = False
        self.plot_dir = Path(plot_dir)
        self.eval_times = []
        self.commanded_velocities = []
        self.actual_velocities = []
        self.evaluation_step = 0
        self.plot_saved = False
        
        self.current_kp = self.env.env_cfg.get("kp", 80.0)
        self.current_kd = self.env.env_cfg.get("kd", 2.0)
        
        # Retrieve and initialize link masses
        self.default_masses = self.env.robot.get_links_inertial_mass()[0].clone()
        self.default_base_mass = self.default_masses[0].item()
        self.current_added_mass = 0.0
        self.current_link_scale = 1.0
        self.current_base_mass = self.default_base_mass
        
        self.init_ui()
        
        # environment observation
        self.obs, _ = self.env.reset()
        self.env.sampled_commands = torch.zeros((self.num_envs, 3), dtype=torch.float, device=gs.device)
        self.env.commands.zero_()
        self.obs["policy"][:, 6:9] = 0.0
        
        # Keep nominal trained dynamics for the default evaluation. Press K or
        # M explicitly when a randomized robustness test is desired.
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step_sim)
        self.timer.start(0) 

        self.push_steps_remaining = 0
        self.push_dv_per_step = torch.zeros((self.num_envs, 3), device=gs.device)

    def init_ui(self):
        self.setWindowTitle('ANYmal C Teleoperation')
        self.setFixedSize(700, 550) 
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #ffffff; font-family: Arial; }
            QLabel { font-size: 14px; }
            .header { font-size: 18px; font-weight: bold; color: #4DA6FF; margin-bottom: 5px; margin-top: 10px; }
            .status-walk { background-color: #2e7d32; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }
            .status-stand { background-color: #c62828; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }
            .value { color: #ffb300; font-weight: bold; }
            .torque-label { font-size: 12px; color: #cccccc; }
            .peak-text { color: #888888; font-size: 11px; }
            .instructions { color: #aaaaaa; font-size: 12px; }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title = QLabel("Control Dashboard")
        title.setProperty('class', 'header')
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.status_label = QLabel("WALK MODE ACTIVE")
        self.status_label.setProperty('class', 'status-walk')
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        vel_frame = QFrame()
        vel_layout = QVBoxLayout(vel_frame)
        
        self.lx_label = QLabel(f"Linear X (W/S): <span class='value'>{self.lin_x:+.2f}</span>")
        self.ly_label = QLabel(f"Linear Y (A/D): <span class='value'>{self.lin_y:+.2f}</span>")
        self.az_label = QLabel(f"Angular Z (Q/E): <span class='value'>{self.ang_z:+.2f}</span>")
        self.pd_label = QLabel(f"Motor Gains: kp = <span class='value'>{self.current_kp:.1f}</span> | kd = <span class='value'>{self.current_kd:.2f}</span>")
        self.mass_label = QLabel(f"Masses: base = <span class='value'>{self.current_base_mass:.2f} kg</span> (added: <span class='value'>{self.current_added_mass:+.2f} kg</span>) | link scale = <span class='value'>{self.current_link_scale:.2f}</span>")
        
        vel_layout.addWidget(self.lx_label)
        vel_layout.addWidget(self.ly_label)
        vel_layout.addWidget(self.az_label)
        vel_layout.addWidget(self.pd_label)
        vel_layout.addWidget(self.mass_label)
        layout.addWidget(vel_frame)

        # Tourqe
        torque_title = QLabel("Joint Torques: Current (Peak) / Limit")
        torque_title.setProperty('class', 'header')
        torque_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(torque_title)

        torque_frame = QFrame()
        torque_layout = QGridLayout(torque_frame)
        self.torque_labels = []
        
        joint_names = self.env.env_cfg["joint_names"]
        for i, name in enumerate(joint_names):
            row = i // 3  
            col = i % 3   
            
            display_name = name.replace("_", " ")
            lbl = QLabel(f"{display_name}: <span class='value'>+0.00</span> <span class='peak-text'>(Peak: 0.00)</span> / {self.max_torques[i]:.1f}")
            lbl.setProperty('class', 'torque-label')
            self.torque_labels.append(lbl)
            torque_layout.addWidget(lbl, row, col)
            
        layout.addWidget(torque_frame)

        instructions = QLabel(
            "Controls:<br>"
            "<b>W/S</b>: Forward/Backward | <b>A/D</b>: Strafe Left/Right | <b>Q/E</b>: Turn Left/Right<br>"
            "<b>Space</b>: Emergency Brake | <b>1</b>: Toggle Stand Pose | <b>K</b>: Randomize PD | <b>C</b>: Custom PD<br>"
            "<b>M</b>: Randomize Mass | <b>N</b>: Custom Mass | <b>O</b>: Push X-Axis | <b>P</b>: Push Y-Axis<br>"
            "<b>R</b>: Manual Reset | <b>8</b>: Quit"
        )
        instructions.setProperty('class', 'instructions')
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)

        self.setLayout(layout)

    def keyPressEvent(self, event):
        key = event.key()
        
        if key == Qt.Key_W: self.lin_x += 0.10
        elif key == Qt.Key_S: self.lin_x -= 0.10
        elif key == Qt.Key_A: self.lin_y += 0.10
        elif key == Qt.Key_D: self.lin_y -= 0.10
        elif key == Qt.Key_Q: self.ang_z += 0.10
        elif key == Qt.Key_E: self.ang_z -= 0.10
        elif key == Qt.Key_1:
            self.stand_mode = not self.stand_mode
            if self.stand_mode:
                self.lin_x, self.lin_y, self.ang_z = 0.0, 0.0, 0.0
        elif key == Qt.Key_R:
            self.manual_reset = True
        elif key == Qt.Key_K:
            self.randomize_pd()
        elif key == Qt.Key_C:
            kp_val, ok1 = QInputDialog.getDouble(self, "Custom kp", "Enter kp value:", self.current_kp, 0, 500, 1)
            if ok1:
                kd_val, ok2 = QInputDialog.getDouble(self, "Custom kd", "Enter kd value:", self.current_kd, 0, 100, 2)
                if ok2:
                    self.set_custom_pd(kp_val, kd_val)
        elif key == Qt.Key_M:
            self.randomize_mass()
        elif key == Qt.Key_N:
            added_mass_val, ok1 = QInputDialog.getDouble(self, "Custom added mass", "Enter payload mass to add to base (kg):", self.current_added_mass, -100, 100, 2)
            if ok1:
                link_scale_val, ok2 = QInputDialog.getDouble(self, "Custom link mass scale", "Enter leg link scale multiplier:", self.current_link_scale, 0.01, 10, 2)
                if ok2:
                    self.set_custom_mass(added_mass_val, link_scale_val)
        elif key == Qt.Key_Space:
            self.lin_x, self.lin_y, self.ang_z = 0.0, 0.0, 0.0
            
        elif key == Qt.Key_O:
            duration_seconds = 0.25  
            total_steps = int(duration_seconds / self.env.dt) 
            
            force_newtons = np.random.uniform(700.0, 1000.0)
            robot_mass_kg = self.env.robot.get_links_inertial_mass().sum(dim=1).clamp_min(1.0)
            direction = np.random.choice([-1.0, 1.0])
            
            # (dv = (F/m) * dt)
            dv_per_step = (force_newtons / robot_mass_kg) * self.env.dt * direction
            
            self.push_dv_per_step[:, 0] = dv_per_step
            self.push_dv_per_step[:, 1] = 0.0
            self.push_steps_remaining = total_steps
            
            print(f"Started X-axis push: {force_newtons * direction:+.1f} N over {duration_seconds}s")

        elif key == Qt.Key_P:
            duration_seconds = 0.25  
            total_steps = int(duration_seconds / self.env.dt) 
            
            force_newtons = np.random.uniform(700.0, 1000.0)
            robot_mass_kg = self.env.robot.get_links_inertial_mass().sum(dim=1).clamp_min(1.0)
            direction = np.random.choice([-1.0, 1.0])
            
            dv_per_step = (force_newtons / robot_mass_kg) * self.env.dt * direction
            
            self.push_dv_per_step[:, 0] = 0.0 
            self.push_dv_per_step[:, 1] = dv_per_step
            self.push_steps_remaining = total_steps
            
            print(f"Started Y-axis push: {force_newtons * direction:+.1f} N over {duration_seconds}s")

        elif key == Qt.Key_8:
            print("Simulation stopped. Exiting...")
            self.close()

        self.lin_x = np.clip(self.lin_x, *self.env.command_cfg["lin_vel_x_range"])
        self.lin_y = np.clip(self.lin_y, *self.env.command_cfg["lin_vel_y_range"])
        self.ang_z = np.clip(self.ang_z, *self.env.command_cfg["ang_vel_range"])
        
        self.update_labels()

    def randomize_pd(self):
        kp_range = self.env.env_cfg.get("kp_range", [20.0, 60.0])
        kd_range = self.env.env_cfg.get("kd_range", [1.5, 4.5])
        
        self.current_kp = np.random.uniform(kp_range[0], kp_range[1])
        self.current_kd = np.random.uniform(kd_range[0], kd_range[1])
        
        kp_tensor = torch.full((1, self.env.num_actions), self.current_kp, device=gs.device)
        kd_tensor = torch.full((1, self.env.num_actions), self.current_kd, device=gs.device)
        
        self.env.robot.set_dofs_kp(kp_tensor, self.env.motors_dof_idx, envs_idx=torch.tensor([0], device=gs.device))
        self.env.robot.set_dofs_kv(kd_tensor, self.env.motors_dof_idx, envs_idx=torch.tensor([0], device=gs.device))
        print(f"Set random PD gains: kp = {self.current_kp:.2f}, kd = {self.current_kd:.2f}")
        self.update_labels()

    def set_custom_pd(self, kp, kd):
        self.current_kp = kp
        self.current_kd = kd
        kp_tensor = torch.full((1, self.env.num_actions), self.current_kp, device=gs.device)
        kd_tensor = torch.full((1, self.env.num_actions), self.current_kd, device=gs.device)
        self.env.robot.set_dofs_kp(kp_tensor, self.env.motors_dof_idx, envs_idx=torch.tensor([0], device=gs.device))
        self.env.robot.set_dofs_kv(kd_tensor, self.env.motors_dof_idx, envs_idx=torch.tensor([0], device=gs.device))
        print(f"Set custom PD gains: kp = {self.current_kp:.2f}, kd = {self.current_kd:.2f}")
        self.update_labels()

    def randomize_mass(self):
        added_mass_range = self.env.env_cfg.get("added_mass_range", [-1.0, 3.0])
        link_mass_range = self.env.env_cfg.get("link_mass_range", [0.8, 1.2])
        
        self.current_added_mass = np.random.uniform(added_mass_range[0], added_mass_range[1])
        self.current_link_scale = np.random.uniform(link_mass_range[0], link_mass_range[1])
        
        new_masses = self.default_masses.clone() * self.current_link_scale
        new_masses[0] += self.current_added_mass
        self.current_base_mass = new_masses[0].item()
        
        new_masses_tensor = new_masses.unsqueeze(0).to(gs.device)
        self.env.robot.set_links_inertial_mass(new_masses_tensor, envs_idx=torch.tensor([0], device=gs.device))
        print(f"Set random masses: base = {self.current_base_mass:.2f} kg (added: {self.current_added_mass:+.2f} kg), link scale = {self.current_link_scale:.2f}")
        self.update_labels()

    def set_custom_mass(self, added_mass, link_scale):
        self.current_added_mass = added_mass
        self.current_link_scale = link_scale
        
        new_masses = self.default_masses.clone() * self.current_link_scale
        new_masses[0] += self.current_added_mass
        self.current_base_mass = new_masses[0].item()
        
        new_masses_tensor = new_masses.unsqueeze(0).to(gs.device)
        self.env.robot.set_links_inertial_mass(new_masses_tensor, envs_idx=torch.tensor([0], device=gs.device))
        print(f"Set custom masses: base = {self.current_base_mass:.2f} kg (added: {self.current_added_mass:+.2f} kg), link scale = {self.current_link_scale:.2f}")
        self.update_labels()

    def update_labels(self):
        if self.stand_mode:
            self.status_label.setText("STAND MODE ACTIVE")
            self.status_label.setProperty('class', 'status-stand')
        else:
            self.status_label.setText("WALK MODE ACTIVE")
            self.status_label.setProperty('class', 'status-walk')
        
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

        self.lx_label.setText(f"Linear X (W/S): <span class='value'>{self.lin_x:+.2f}</span>")
        self.ly_label.setText(f"Linear Y (A/D): <span class='value'>{self.lin_y:+.2f}</span>")
        self.az_label.setText(f"Angular Z (Q/E): <span class='value'>{self.ang_z:+.2f}</span>")
        self.pd_label.setText(f"Motor Gains: kp = <span class='value'>{self.current_kp:.1f}</span> | kd = <span class='value'>{self.current_kd:.2f}</span>")
        self.mass_label.setText(f"Masses: base = <span class='value'>{self.current_base_mass:.2f} kg</span> (added: <span class='value'>{self.current_added_mass:+.2f} kg</span>) | link scale = <span class='value'>{self.current_link_scale:.2f}</span>")

    def step_sim(self):
        with torch.no_grad():
            if self.manual_reset:
                self.obs, _ = self.env.reset()
                self.lin_x, self.lin_y, self.ang_z = 0.0, 0.0, 0.0
                self.peak_torques.fill(0.0)
                self.manual_reset = False
                self.update_labels()

            # Apply the current user command before policy inference.
            self.env.sampled_commands[:, 0] = self.lin_x
            self.env.sampled_commands[:, 1] = self.lin_y
            self.env.sampled_commands[:, 2] = self.ang_z
            self.env.commands.copy_(self.env.sampled_commands)
            self.obs["policy"][:, 6:9] = self.env.commands * self.env.commands_scale

            if self.stand_mode:
                actions = torch.zeros((self.num_envs, self.env.num_actions), device=gs.device)
            else:
                actions = self.policy(self.obs)


            if self.push_steps_remaining > 0:
                base_vel_idx = [0, 1, 2]
                current_vel = self.env.robot.get_dofs_velocity(base_vel_idx)
                
                self.env.robot.set_dofs_velocity(current_vel + self.push_dv_per_step, base_vel_idx)
                
                self.push_steps_remaining -= 1
                # if self.push_steps_remaining == 0:
                #     print("Push complete.")

            self.obs, rews, dones, infos = self.env.step(actions)

            self.eval_times.append(self.evaluation_step * self.env.dt)
            self.commanded_velocities.append(float(self.lin_x))
            self.actual_velocities.append(float(self.env.base_lin_vel[0, 0]))
            self.evaluation_step += 1
            
            current_torques = self.env.torques[0].cpu().numpy()
            
            self.peak_torques = np.maximum(self.peak_torques, np.abs(current_torques))
            
            joint_names = self.env.env_cfg["joint_names"]
            
            for i, t in enumerate(current_torques):
                display_name = joint_names[i].replace("_", " ")
                peak_t = self.peak_torques[i]
                

                limit = self.max_torques[i]
                if abs(t) >= (limit * 0.95):  
                    t_str = f"<span style='color: #ff4444; font-weight: bold;'>{t:+.2f}</span>"
                else:
                    t_str = f"<span class='value'>{t:+.2f}</span>"
                    
                peak_str = f"<span class='peak-text'>(Peak: {peak_t:.2f})</span>"
                
                self.torque_labels[i].setText(f"{display_name}: {t_str} {peak_str} / {limit:.1f}")

    def save_evaluation_plot(self):
        if self.plot_saved:
            return
        save_velocity_tracking_plot(
            self.plot_dir,
            self.eval_times,
            self.commanded_velocities,
            self.actual_velocities,
        )
        self.plot_saved = True

    def closeEvent(self, event):
        self.timer.stop()
        self.save_evaluation_plot()
        event.accept()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="Anymal1")
    parser.add_argument("-c", "--ckpt", type=int, default=300)
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(logging_level="warning")
    log_dir = f"logs/{args.exp_name}"
    
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))    
    reward_cfg["reward_scales"] = {}
    # Teleoperation supplies every command; disable random command resampling.
    env_cfg["resampling_time_s"] = 1.0e9

    env = QuadV3(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    env.max_episode_length = float('inf')
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.ckpt is None:
        checkpoints = sorted(
            Path(log_dir).glob("model_*.pt"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No model checkpoints found in {log_dir}")
        resume_path = str(checkpoints[-1])
    else:
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    print(f"Loading checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    app = QApplication(sys.argv)
    window = TeleopUI(env, policy, Path(log_dir) / "performance")
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
