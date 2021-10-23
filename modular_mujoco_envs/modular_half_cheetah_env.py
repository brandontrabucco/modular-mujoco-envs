from modular_mujoco_envs.modular_mujoco_env import ModularMujocoEnv


class ModularHalfCheetahEnv(ModularMujocoEnv):
    """Defines a custom base class for MuJoCo environments that provides 
    a common interface for extracting morphology information from the agent, 
    including the positions, orientations, and range of each limb.
    
    """

    def __init__(self, xml, control_penalty=0.1, alive_bonus=0.0,
                 time_skip=5, include_joint_range_in_obs=True,
                 include_position_in_obs=True, 
                 include_orientation_in_obs=True,
                 include_position_vel_in_obs=True, 
                 include_orientation_vel_in_obs=True,
                 one_joint_per_limb=True, hide_root_x_position=True):
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
        time_skip: int
            the number of time steps to run the mujoco simulator for every
            step of the outer reinforcement learning environment.

        include_joint_range_in_obs: bool
            a boolean that specifies whether the observation of the robot
            includes a normalized description of the joint range.
        include_position_in_obs: bool
            a boolean that specifies whether the observation of the robot
            includes a normalized description of the body position.
        include_orientation_in_obs: bool
            a boolean that specifies whether the observation of the robot
            includes a normalized description of the body orientation.
            
        include_position_vel_in_obs: bool
            a boolean that specifies whether the observation of the robot
            includes a normalized description of the body velocity.
        include_orientation_vel_in_obs: bool
            a boolean that specifies whether the observation of the robot
            includes a normalized description of the body velocity.

        one_joint_per_limb: bool
            a boolean that controls whether each observation per limb 
            includes only a single joint if multiple are available.
        hide_root_x_position: bool
            a boolean that controls whether the root x position of the
            agent is not included in the observation for the torso.

        """

        # build the superclass using modified default arguments
        super(ModularHalfCheetahEnv, self).__init__(
            xml, control_penalty=control_penalty, 
            alive_bonus=alive_bonus, time_skip=time_skip,
            include_joint_range_in_obs=include_joint_range_in_obs,
            include_position_in_obs=include_position_in_obs, 
            include_orientation_in_obs=include_orientation_in_obs,
            include_position_vel_in_obs=include_position_vel_in_obs, 
            include_orientation_vel_in_obs=include_orientation_vel_in_obs,
            one_joint_per_limb=one_joint_per_limb, 
            hide_root_x_position=hide_root_x_position)

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
            self.np_random.randn(self.model.nv) * 0.1

        # set the initial state
        self.set_state(init_qpos, init_qvel)

        # build a modular observation for the agent
        return self._get_obs()

    def viewer_setup(self):
        """Positions the camera in the scene in order to visualize the
        behavior acquired by neural network policies.

        """
        self.viewer.cam.distance = self.model.stat.extent * 0.5
