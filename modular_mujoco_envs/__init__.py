import gym
import os


def pretty(text):
    return text.replace("_", " ").title().replace(" ", "")


def get_xml_path(variant):
    return os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "xmls/" + variant)


for env_variant in [
        "cheetah_2_back.xml",
        "cheetah_2_front.xml",
        "cheetah_3_back.xml",
        "cheetah_3_front.xml",
        "cheetah_3_balanced.xml",
        "cheetah_4_allback.xml",
        "cheetah_4_allfront.xml",
        "cheetah_4_back.xml",
        "cheetah_4_front.xml",
        "cheetah_5_back.xml",
        "cheetah_5_front.xml",
        "cheetah_5_balanced.xml",
        "cheetah_6_back.xml",
        "cheetah_6_front.xml",
        "cheetah_7_full.xml"]:

    gym.envs.register(
        id="ModularHalfCheetah{0}Env-v0".format(pretty(env_variant[8:-4])),
        entry_point="modular_mujoco_envs" + 
        ".modular_half_cheetah_env:ModularHalfCheetahEnv",
        max_episode_steps=200,
        kwargs=dict(xml=get_xml_path(env_variant)))


for env_variant in [
        "walker_2_flipped.xml",
        "walker_2_main.xml",
        "walker_3_flipped.xml",
        "walker_3_main.xml",
        "walker_4_flipped.xml",
        "walker_4_main.xml",
        "walker_5_flipped.xml",
        "walker_5_main.xml",
        "walker_6_flipped.xml",
        "walker_6_main.xml",
        "walker_7_flipped.xml",
        "walker_7_main.xml"]:

    height_threshold = {
        "walker_2_flipped.xml": 0.26,
        "walker_2_main.xml": 0.26,
        "walker_3_flipped.xml": 0.26,
        "walker_3_main.xml": 0.26,
        "walker_4_flipped.xml": 0.136,
        "walker_4_main.xml": 0.136,
        "walker_5_flipped.xml": 0,
        "walker_5_main.xml": 0,
        "walker_6_flipped.xml": 0,
        "walker_6_main.xml": 0,
        "walker_7_flipped.xml": 0,
        "walker_7_main.xml": 0}

    gym.envs.register(
        id="ModularWalker2d{0}Env-v0".format(pretty(env_variant[7:-4])),
        entry_point="modular_mujoco_envs" + 
        ".modular_walker2d_env:ModularWalker2dEnv",
        max_episode_steps=200,
        kwargs=dict(xml=get_xml_path(env_variant), 
                    height_threshold=height_threshold[env_variant]))


for env_variant in [
        "hopper_3.xml",
        "hopper_4.xml",
        "hopper_5.xml"]:

    height_threshold = {
        "hopper_3.xml": 0.45,
        "hopper_4.xml": 0.6,
        "hopper_5.xml": 0.95}

    gym.envs.register(
        id="ModularHopper{0}Env-v0".format(pretty(env_variant[7:-4])),
        entry_point="modular_mujoco_envs" + 
        ".modular_hopper_env:ModularHopperEnv",
        max_episode_steps=200,
        kwargs=dict(xml=get_xml_path(env_variant), 
                    height_threshold=height_threshold[env_variant]))


for env_variant in [
        "humanoid_2d_7_left_arm.xml",
        "humanoid_2d_7_left_leg.xml",
        "humanoid_2d_7_lower_arms.xml",
        "humanoid_2d_7_right_arm.xml",
        "humanoid_2d_7_right_leg.xml",
        "humanoid_2d_8_left_knee.xml",
        "humanoid_2d_8_right_knee.xml",
        "humanoid_2d_9_full.xml"]:

    gym.envs.register(
        id="ModularHumanoid2d{0}Env-v0".format(pretty(env_variant[12:-4])),
        entry_point="modular_mujoco_envs" + 
        ".modular_humanoid2d_env:ModularHumanoid2dEnv",
        max_episode_steps=200,
        kwargs=dict(xml=get_xml_path(env_variant)))
