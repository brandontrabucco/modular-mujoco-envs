from modular_mujoco_envs.modular_mujoco_env import ModularMujocoEnv


class ModularHalfCheetahEnv(ModularMujocoEnv):
    """Defines a custom base class for MuJoCo environments that provides 
    a common interface for extracting morphology information from the agent, 
    including the positions, orientations, and range of each limb.
    
    """

    def __init__(self, xml, control_penalty=0.1, alive_bonus=0.0,
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
        super(ModularHalfCheetahEnv, self).__init__(
            xml, control_penalty=control_penalty, alive_bonus=alive_bonus,
            include_joint_range_in_obs=include_joint_range_in_obs,
            include_position_in_obs=include_position_in_obs, 
            include_orientation_in_obs=include_orientation_in_obs,
            include_position_vel_in_obs=include_position_vel_in_obs, 
            include_orientation_vel_in_obs=include_orientation_vel_in_obs)

    def reset_model(self):
        """Resets the environment to a random orientation defined by the 
        initial qpos range and initial qvel range specific in the 
        constructor, and return the first observation.

        Returns:

        obs

        """

        # randomly sample an initial position
        init_qpos = self.init_qpos + \
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)

        # randomly sample an initial velocity
        init_qvel = self.init_qvel + \
            self.np_random.randn(self.model.nv) * .1

        # set the initial state
        self.set_state(init_qpos, init_qvel)

        # build a modular observation for the agent
        return self._get_obs()

    def viewer_setup(self):
        """Positions the camera in the scene in order to visualize the
        behavior acquired by neural network policies.

        """
        self.viewer.cam.distance = self.model.stat.extent * 0.5
