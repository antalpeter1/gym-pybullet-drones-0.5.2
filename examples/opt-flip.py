"""Script demonstrating the ILC flip algorithm.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python flip.py

Notes
-----


"""
import sys
sys.path.append('C:/Users/Peter/Documents/python/gym-pybullet-drones-0.5.2/')
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.Flip import Flip
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs



def simulate(U1, T1, T3, U5, T5):
    parameters = np.array([U1, T1, T3, U5, T5])
    env.reset()
    START = time.time()
    sections = flip.get_sections(parameters)  # [(ct1, theta_d1, t1), (ct2,...
    T = np.zeros(5)
    for i in range(len(sections)):
        T[i] = sections[i][2]
    T = np.abs(T)
    for i in range(1, len(sections)):
        T[i] = T[i - 1] + T[i]
    T = T * env.SIM_FREQ + env.SIM_FREQ / 10

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(1)}
    for i in range(0, int(duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i % CTRL_EVERY_N_STEPS == 0:
            #### Compute control for the current way point #############
            if i < simulation_freq_hz / 10:
                action['0'], _, _ = PID.computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                                                                state=obs['0']["state"],
                                                                target_pos=[0, 0, 1]
                                                                )
            else:
                try:
                    num_sec = np.min([k for k, x in enumerate(T) if i / x < 1])  # decide in which section we are
                except:
                    final_state = obs['0']["state"]
                    error_norm = np.linalg.norm([final_state[0:3] - [0, 0, 1], final_state[10:13], final_state[7:10]])
                    # env.close()
                    # DUR = time.time()-START
                    # print('Duration of simulation: ', DUR)
                    return 1/error_norm
                finally:
                    action['0'] = flip.compute_control_from_section(sections[num_sec], obs['0']["state"][9:12])

        # logger.log(drone=0,
        #            timestamp=i / env.SIM_FREQ,
        #            state=obs['0']["state"],
        #            control=np.hstack([[0, 0], H, np.zeros(9)])
        #            )
        #### Save the simulation results ###########################
        #logger.save()


        #### Plot the simulation results ###########################


if __name__ == "__main__":
    #### Initialize the simulation #############################
    H = 1
    INIT_XYZS = np.array([[0, 0, H]])
    simulation_freq_hz = 240
    control_freq_hz = 40
    drone = DroneModel("cf2p")
    duration_sec = 6
    AGGR_PHY_STEPS = 1
    physics = Physics("pyb")

    #### Create the environment with or without video capture ##
    env = CtrlAviary(drone_model=drone,
                     num_drones=1,
                     initial_xyzs=INIT_XYZS,
                     physics=physics,
                     neighbourhood_radius=10,
                     freq=simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=False,
                     record=False,
                     obstacles=False,
                     user_debug_gui=False
                     )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=1
                    )

    #### Initialize the controllers ############################
    PID = DSLPIDControl(env)
    flip = Flip()

    # params = flip.get_initial_parameters()
    # params = np.abs(params)

    # params = np.array([18.774, 0.1, 0.11096576, 18.774, 0.1])
    params = np.array([18.78215108032221, 0.08218741361206124, 0.12091343074644069, 17.951940703885207, 0.05507561729533186])
    # U1, T1, T3, U5 = params[0:4]
    # disturb = np.linspace(-5, 5, 10)

    pbounds = {'U1': (12, 24), 'T1': (0.05, 0.15), 'T3': (0, 0.15), 'U5': (12, 24), 'T5': (0.02, 0.15), }
    #pbounds = {'T5': (0.05, 0.15), }
    optimizer = BayesianOptimization(
        f=simulate,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=10,
    )
    # load_logs(optimizer, logs=["./logs-init-400-iter-1000.json"]);

    optimizer.maximize(
        init_points=10,
        n_iter=100,
        acq="ucb",
    )
    print(optimizer.max)


    # logger.plot()
