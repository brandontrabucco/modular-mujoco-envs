from setuptools import find_packages
from setuptools import setup
import glob
import os


F = 'README.md'
with open(F, 'r') as readme:
    LONG_DESCRIPTION = readme.read()


CLASSIFIERS = [
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7']


setup(
    name='modular-mujoco-envs', version='1.3', license='MIT',
    packages=find_packages(include=['modular_mujoco_envs', 
                                    'modular_mujoco_envs.*']),
    include_package_data=True,
    description='Modular MuJoCo Environments',
    long_description=LONG_DESCRIPTION, classifiers=CLASSIFIERS,
    long_description_content_type='text/markdown',
    keywords=['Deep Learning', 'Deep Reinforcement Learning'],
    author='Brandon Trabucco', author_email='brandon@btrabucco.com',
    url='https://github.com/brandontrabucco/modular-mujoco-envs',
    download_url='https://github.com/brandontrabucco'
                 '/modular-mujoco-envs/archive/v1_3.tar.gz')
