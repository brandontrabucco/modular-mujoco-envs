import numpy as np
from modular_mujoco_envs.utils import quat2expmap
from gym import utils
from gym.envs.mujoco import mujoco_env


class ModularMujocoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Defines a custom base class for MuJoCo environments that provides 
    a common interface for extracting morphology information from the agent, 
    including the positions, orientations, and range of each limb.
    
    """

    def __init__(self, xml, control_penalty=1e-3, alive_bonus=1.0,
                 time_skip=5, include_joint_range_in_obs=True,
                 include_position_in_obs=True, 
                 include_orientation_in_obs=True,
                 include_position_vel_in_obs=True, 
                 include_orientation_vel_in_obs=True,
                 one_joint_per_limb=False, hide_root_x_position=True):
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
        
        # parameters that control the reward function
        self.control_penalty = control_penalty
        self.alive_bonus = alive_bonus
        self.time_skip = time_skip
        self.xml = xml

        # constants that control what is observed by the agent
        self.include_joint_range_in_obs = include_joint_range_in_obs
        self.include_position_in_obs = include_position_in_obs
        self.include_orientation_in_obs = include_orientation_in_obs
        self.include_position_vel_in_obs = include_position_vel_in_obs
        self.include_orientation_vel_in_obs = include_orientation_vel_in_obs
        self.one_joint_per_limb = one_joint_per_limb
        self.hide_root_x_position = hide_root_x_position

        # launch the mujoco simulator with the given xml
        mujoco_env.MujocoEnv.__init__(self, xml, time_skip)
        utils.EzPickle.__init__(self)

    def reset_model(self):
        """Resets the environment to a random orientation defined by the 
        initial qpos range and initial qvel range specific in the 
        constructor, and return the first observation.

        Returns:

        obs

        """

        # randomly sample an initial position
        init_qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq)

        # randomly sample an initial velocity
        init_qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv)

        # set the initial state
        self.set_state(init_qpos, init_qvel)

        # build a modular observation for the agent
        return self._get_obs()

    def viewer_setup(self):
        """Positions the camera in the scene in order to visualize the
        behavior acquired by neural network policies.

        """

        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
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

        return False

    def step(self, action):
        """Defines a basic transition function and single-task reward 
        function for morphology-agnostic agents, encouraging the 
        agent to run forward as quickly as possible.

        Arguments:

        action: np.ndarray
            a vector of actions for a modular mujoco agent, which
            should match the order of bodies in the xml file.

        Returns:

        obs, reward, done, info

        """

        # record the x position before and after taking actions
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter = self.sim.data.qpos[0]

        # reward is how fast the agent has moved on the x axis
        reward_run = (posafter - posbefore) / self.dt
        reward_control = self.control_penalty * np.square(action).sum()
        reward = self.alive_bonus + reward_run - reward_control

        # return a tuple for each step taken in the environment
        return (self._get_obs(), reward, self.has_finished(action), 
                dict(reward_run=reward_run, 
                     reward_control=reward_control))

    def _get_obs_per_body(self, body):
        """Defines a basic observation generator per limb, using only the
        available information in the qpos and qvel exposed by MuJoCo.

        Arguments:

        body: str
            the name of the body to generate an observation vector, such 
            as the "bthigh" in the case of the half cheetah.

        Returns:

        obs_per_limb

        """

        # determine the address of the agent's joint for this body
        body_id = self.sim.model.body_name2id(body)
        jnt_adr = self.sim.model.body_jntadr[body_id]
        next_jnt_adr = (self.sim.model.body_jntadr[body_id + 1] 
                        if body_id + 1 < 
                        self.sim.model.body_jntadr.size else None)

        # if only one joint per limb is included choose the last
        jnt_qposadr_list = self.sim.model.jnt_qposadr[jnt_adr:next_jnt_adr]
        if self.one_joint_per_limb:
            jnt_qposadr_list = [jnt_qposadr_list[-1]]

        observations = []
        for jnt_qposadr in jnt_qposadr_list:

            # do not include the x position of the agent, only velocity
            if jnt_qposadr == 0 and self.hide_root_x_position:
                observations.append([self.data.qvel[jnt_qposadr]])

            # return the qpos and qvel observation for each body
            else:
                observations.append([self.data.qpos[jnt_qposadr], 
                                     self.data.qvel[jnt_qposadr]])

        # whether to observe the joint actuation range for each body
        if self.include_joint_range_in_obs:
            observations.append(self.sim.model.jnt_range[jnt_adr])

        # whether to observe the xyz position for each body
        if self.include_position_in_obs:
            xpos = self.data.get_body_xpos(body)
            observations.append(xpos - self.data.get_body_xpos('torso'))

        # whether to observe the orientation for each body
        if self.include_orientation_in_obs:
            observations.append(
                quat2expmap(self.data.get_body_xquat(body)))

        # whether to observe the positional velocity for each body
        if self.include_position_vel_in_obs:
            observations.append(
                np.clip(self.data.get_body_xvelp(body), -10, 10))

        # whether to observe the angular velocity for each body
        if self.include_orientation_vel_in_obs:
            observations.append(self.data.get_body_xvelr(body))

        # return the final stacked obs for each limb
        return np.concatenate(observations)

    def _get_obs(self):
        """Defines an observation generator for the MuJoCo robot by iterating
        for each body the agent contains and generating an individual
        observation for that part of the agent.

        Returns:

        obs

        """

        # generate an observation for each body
        return np.concatenate([self._get_obs_per_body(b) 
                               for b in self.model.body_names[1:]])
