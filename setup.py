from setuptools import find_packages
from setuptools import setup


F = 'README.md'
with open(F, 'r') as readme:
    LONG_DESCRIPTION = readme.read()


setup(
    name='modular-mujoco-envs',
    version='0.1',
    license='MIT',
    packages=find_packages(include=[
        'modular_mujoco_envs', 
        'modular_mujoco_envs.*']),
    description='Modular MuJoCo Environments',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Brandon Trabucco',
    author_email='brandon@btrabucco.com',
    url='https://github.com/brandontrabucco/modular-mujoco-envs',
    keywords=['Deep Learning', 'Deep Reinforcement Learning'],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'])
