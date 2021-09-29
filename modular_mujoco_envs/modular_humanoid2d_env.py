from modular_mujoco_envs.modular_mujoco_env import ModularMujocoEnv
import numpy as np


class ModularHumanoid2dEnv(ModularMujocoEnv):
    """Defines a custom base class for MuJoCo environments that provides 
    a common interface for extracting morphology information from the agent, 
    including the positions, orientations, and range of each limb.
    
    """

    def __init__(self, xml, control_penalty=1e-3, alive_bonus=1.0,
                 include_joint_range_in_obs=True,
                 include_position_in_obs=True, 
                 include_orientation_in_obs=True,
                 include_position_vel_in_obs=True, 
                 include_orientation_vel_in_obs=True):
        """Instantiates a modular MuJoCo environment using a custom xml 
        file defining the structure of the agent, and provides a clean 
        interface to extracting the agent's morphology.

        Arguments:

        xml: str
            the path an xml file on the disk containing an agent with a 
            unique morphology, such as a humanoid with only one leg.
        control_penalty: float
            penalize taking large actions by adding a term proportional 
            to the negative l2 norm of the action to the reward.
        alive_bonus: float
            encourage the agent to stay alive and avoid falling by adding 
            a positive constant to the reward at every step.

        include_joint_range_in_obs: float
            a boolean that specifies whether the observation of the robot
            includes a normalized description of the joint range.
        include_position_in_obs: float
            a boolean that specifies whether the observation of the robot
            includes a normalized description of the body position.
        include_orientation_in_obs: float
            a boolean that specifies whether the observation of the robot
            includes a normalized description of the body orientation.
            
        include_position_vel_in_obs: float
            a boolean that specifies whether the observation of the robot
            includes a normalized description of the body velocity.
        include_orientation_vel_in_obs: float
            a boolean that specifies whether the observation of the robot
            includes a normalized description of the body velocity.

        """

        # build the superclass using modified default arguments
        super(ModularMujocoEnv, self).__init__(
            xml, control_penalty=control_penalty, alive_bonus=alive_bonus,
            include_joint_range_in_obs=include_joint_range_in_obs,
            include_position_in_obs=include_position_in_obs, 
            include_orientation_in_obs=include_orientation_in_obs,
            include_position_vel_in_obs=include_position_vel_in_obs, 
            include_orientation_vel_in_obs=include_orientation_vel_in_obs)

    def viewer_setup(self):
        """Positions the camera in the scene in order to visualize the
        behavior acquired by neural network policies.

        """

        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

    def has_finished(self, action):
        """Determines whether the agent has fallen, or otherwise is unable
        to proceed in the environemnt, and returns True on such 
        conditions, indicating the episode has finished.

        Arguments:

        action: np.ndarray
            a vector of actions for a modular mujoco agent, which
            should match the order of bodies in the xml file.

        Returns:

        done

        """

        torso_height, torso_ang = self.sim.data.qpos[1:3]
        return not (torso_height > 0.4 and torso_height < 2.1 and
                    torso_ang > -1.0 and torso_ang < 1.0)
