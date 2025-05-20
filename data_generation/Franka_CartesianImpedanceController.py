
"""
https://github.com/rail-berkeley/serl_franka_controllers/blob/main/src/cartesian_impedance_controller.cpp
"""

import torch
from isaacgym import gymtorch


class CartesianImpedanceController:

    def __init__(self):
        """
            init
                stiffness,
                damping,
                joint names,
                posiz,
                orientamento (quaternione)
                posiz_target,
                orientamento_target


            missing
                arm_id,
                equilibrium pose (might be the default pose?),
                state_interface,
                state_handle,
                effort_joint_interface,
                dynamic_reconfigure_compliance_param_node_,
                dynamic_server_compliance_param_,
                model_interface (serve per recuperare handle),
                model_handle (oggetto recuperato),

        """
        pass

    def start(self):
        """
            init
                 stato_robot (oggetto per trovare posiz e orientamento)
                 jacob_eef
                 azione (q_initial) a zero
                 ottiene limiti dei torque dei giunti

                 posiz
                 orientamento
                 posiz_target
                 orientamento_target


        """
        pass

    def update(self):
        """
            calcola errore posiz dell'eef
            calcola errore orientamento dell'eef

            calcola pseudo_inv jacob trasposta

            calcola torque per la task tramite
                - trasposta_jeef * (stiffness - dpose - damping * (jeef * velocita_joints)

            clamp torque entro i limiti con funzione saturated_torque()
        """
        pass

    def saturated_torque(self, actual_torque):
        """
            limita il torque da applicare sui giunti tagliandolo entro i limiti
        """
        pass