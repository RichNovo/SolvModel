try:
    #from tblite.interface import Calculator
    from tblite.ase import TBLite as TBLite
except:
    from xtb.ase import XTB as TBLite

import copy
import sys

import ase.optimize
import gymnasium as gym
import numpy as np

from gymnasium import spaces

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
from typing import Dict, Optional

from atom_model.policy import SpecialPartialRotDiagGaussianDistribution

#from xtb.ase.calculator import XTB
#from xtb.libxtb import VERBOSITY_MINIMAL, VERBOSITY_MUTED
#from xtb.interface import Calculator, Param



import math
import os
import json
import signal
#from wrapt_timeout_decorator import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt

import torch as th

import time
from ase.build import molecule
from ase.collections import g2
import ase
from ase.io.trajectory import Trajectory

from ase.build import fcc110
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, Hookean
from ase.optimize.minimahopping import MinimaHopping
from ase import Atom, Atoms
import itertools
from ase.geometry.analysis import Analysis

mol_id_offset = 50


def abstract_mapping_to_node(nodes, nodes_xyz, rot, node_map):
    ret_nodes = []
    ret_nodes_xyz = []

    for i in range(len(nodes)):
        if nodes[i] in node_map:
            for elem in node_map[nodes[i]]:
                ret_nodes.append(elem["id"])
                ret_nodes_xyz.append((rot[i]@elem["xyz"])+nodes_xyz[i])
        else:
            ret_nodes.append(nodes[i])
            ret_nodes_xyz.append(nodes_xyz[i])
    return np.array(ret_nodes), np.array(ret_nodes_xyz)

def fibonacci_latice(number_of_points, radius = 1):

    n = number_of_points
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 *np.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+0.5)/n)
    x, y, z = radius * np.cos(theta) * np.sin(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(phi)
    coords=np.array([x, y, z])
    coords = coords.swapaxes(1,0)
    return coords

def xtb_total_energy_difference(nodes, nodes_xyz, base_energy = 0, error_energy=float('NaN'), solvation = None):
    numbers=nodes
    positions=nodes_xyz
    
    calc = TBLite(method = "GFN2-xTB")
    calc.set(verbosity = 0)
    calc.set(max_iterations = 100)
    calc.set(accuracy = 1.0)
    if solvation is not None:
        calc.set(alpb_solvation = solvation)

    atoms = ase.Atoms(numbers,positions)
    atoms.calc = calc
    original_stdout = sys.stdout

    sys.stdout = open(os.devnull, 'w')

    try:
        #res = calc.singlepoint()
        energy =  atoms.get_potential_energy()
        return (energy-base_energy)
    except TimeoutError:
        print("Energy calculation run out of time")
        return error_energy
    except:
        return error_energy
    finally:
        sys.stdout = original_stdout

class AtomMovingEnvTest(gym.Env):
    """Custom Environment for testing purposes only"""

    metadata = {"render_modes": ["human","none"]}

    def __init__(
        self,
        nested_dict_obs=False,
        node_type_num = 110, 
        render_mode: str = "own",
        use_rot = False,
        mol_setup = "FormicAcidWater",
        additional_seed = 0,
        additional_params = {}
    ):
        super().__init__()

        self.node_type_num = node_type_num
        self.action_space = None
        self.observation_space = None
        self.normalized_reward = None
        self.write_data_to_file = False
        self.render_mode = render_mode
        self.use_rot =use_rot
        self.random_init_state = False
        self.optimal_energy = None
        self.use_rigid = True
        self.rand_box_size = 10.
        self.model_version = 0
        self.resetter = None
        self.optimal_state = None
        
        self.reward_skip = 0
        self.env_number_of_runs = -1

        self.solvation = None
        self.min_energy = float('inf')
        self.mol_id_offset = mol_id_offset
        self.low_pass_filter = 0.
        if "low_pass_filter" in additional_params:
            self.low_pass_filter = additional_params["low_pass_filter"]

        self.low_pass_filter_pos = 0.
        if "low_pass_filter_pos" in additional_params:
            self.low_pass_filter_pos = additional_params["low_pass_filter_pos"]
        
        self.water_mol_number = 10
        if "water_mol_number" in additional_params:
            self.water_mol_number = additional_params["water_mol_number"]

        self.max_step_per_ep = 100
        if "max_step_per_ep" in additional_params:
            self.max_step_per_ep = additional_params["max_step_per_ep"]

        self.until_none_reward = True
        if "until_none_reward" in additional_params:
            self.until_none_reward = additional_params["until_none_reward"]

        self.mol_setup_index = None
        if "mol_setup_index" in additional_params:
            self.mol_setup_index = additional_params["mol_setup_index"]
            
        self.pos_step_scale = None
        if "pos_step_scale" in additional_params:
            self.pos_step_scale = additional_params["pos_step_scale"]
        else:
            self.pos_step_scale = 0.1

        self.max_number_of_nodes = None
        if "max_number_of_nodes" in additional_params:
            self.max_number_of_nodes = additional_params["max_number_of_nodes"]
        else:
            self.max_number_of_nodes = None
        
        self.use_time_embeddings = 0
        if "use_time_embeddings" in additional_params:
            self.use_time_embeddings = additional_params["use_time_embeddings"]
        else:
            self.use_time_embeddings = 0

        self.optimize_last_step = False
        if "optimize_last_step" in additional_params:
            self.optimize_last_step = additional_params["optimize_last_step"]
        
        self.initial_energy = None

        self.rng = np.random.default_rng(int(time.time()*1000000))

        self.node_map = {
            #5: [{'id':0,'xyz':[-0.545,0,0]},{'id':5,'xyz':[0.545,0,0]}]
            100: [{'id':5,'xyz':[0.,0.0,0.0]},
                {'id':0,'xyz':[1.09,0.,0.]},],

            101: [{'id':5,'xyz':[-0.76755,0.0,0.0]},
                #{'id':0,'xyz':[1.09,1.09,1.09]},
                {'id':0,'xyz':[-0.76755-0.39205,1.02133,0.]},
                {'id':0,'xyz':[-0.76755-0.39205,-0.5106,0.8845]},
                {'id':0,'xyz':[-0.76755-0.39205,-0.5106,-0.8845]}],

        }

        molecules = ["CH4", "H2CO","HCOOH","HCCl3","C2H2","CO","CO2","LiF","CN","H3CNH2", "NH3", "NO", "NO2"]
        for mol_id in range(len(molecules)):
            atoms = molecule(molecules[mol_id])
            tmp = []
            for i in range(len(atoms.numbers)):
                tmp.append({'id':atoms.numbers[i] -1, 'xyz':atoms.positions[i]})
            self.node_map[mol_id+self.mol_id_offset] = tmp


        atoms = molecule('H2O')
        tmp = []
        for i in range(len(atoms.numbers)):
            tmp.append({'id':atoms.numbers[i] -1, 'xyz':atoms.positions[i]})
        self.node_map[102] = tmp

        atoms = molecule('HCOOH')
        tmp = []
        for i in range(len(atoms.numbers)):
            tmp.append({'id':atoms.numbers[i] -1, 'xyz':atoms.positions[i]})
        self.node_map[103] = tmp


        self.node_map = self.center_node_mapping(self.node_map)


        benzene = {'nodes':np.array([5,5,5,5,5,5,0,0,0,0,0,0]), \
                    'nodes_xyz':np.array([
                    [1.39* np.sin(np.pi/6.) , -1.39* np.cos(np.pi/6.), 0.] , [-1.39* np.sin(np.pi/6.) , -1.39* np.cos(np.pi/6.), 0.],
                    [1.39* np.sin(np.pi/6.) , 1.39* np.cos(np.pi/6.), 0.] , [-1.39* np.sin(np.pi/6.) , 1.39* np.cos(np.pi/6.), 0.],
                    [1.39, 0., 0.],[-1.39, 0., 0.],

                    [2.48* np.sin(np.pi/6.) , -2.48* np.cos(np.pi/6.), 0.] , [-2.48* np.sin(np.pi/6.) , -2.48* np.cos(np.pi/6.), 0.],
                    [2.48* np.sin(np.pi/6.) , 2.48* np.cos(np.pi/6.), 0.] , [-2.48* np.sin(np.pi/6.) , 2.48* np.cos(np.pi/6.), 0.],
                    [2.48, 0., 0.],[-2.48, 0., 0.],
                    ]), \
                  }

           
        if mol_setup == "C6H6":
            def init_fun():
                #atoms = molecule('HCOOH')
                self.initial_energy = None
                self.initial_state = {'nodes':np.array([5,5,5,5,5,5,0,0,0,0,0,0]), }
                
            
                def gen_positions(num):
                    nodes_xyz_list_tmp = []
                    shell = 1

                    #nodes_xyz_list_tmp.append(np.random.random(list([num])+[3])*5.0-2.5)#fibonacci_latice(num,  5) + (np.random.random(list([num])+[3]) - 0.5) * 1.)
                    nodes_xyz_list_tmp.append(fibonacci_latice(num, 2) + (np.random.random(list([num])+[3]) - 0.5) * 2.)
                    return np.concatenate(nodes_xyz_list_tmp)


                running = 0
                limit = 100
                while running < limit and self.initial_energy is None:
                    
                    self.initial_state['nodes_xyz'] = gen_positions(self.initial_state['nodes'].shape[0])
                    self.initial_state['rot'] =  R.from_euler("xyz",np.zeros_like(self.initial_state['nodes_xyz'])).as_matrix()

                    nodes, nodes_xyz = abstract_mapping_to_node(self.initial_state['nodes'], self.initial_state['nodes_xyz'], self.initial_state['rot'], self.node_map)
                    self.initial_energy = xtb_total_energy_difference(nodes + 1, nodes_xyz ,0.0 ,float('NaN'), self.solvation)

                    if math.isnan(self.initial_energy) == False:
                        running = limit
                    else:
                        running = running + 1
                        self.initial_energy = None

                if 'fix_rot_mask' not in self.initial_state:
                    self.initial_state['fix_rot_mask'] = np.ones([self.initial_state['nodes'].shape[0],9])
                    self.initial_state['fix_rot_mask'][:,3:9]=0.

                if 'rot' not in self.initial_state:
                    self.initial_state['rot'] =  R.from_euler("xyz",np.zeros_like(self.initial_state['nodes_xyz'])).as_matrix()

                if self.initial_energy is None:
                    #self.initial_energy = 0.
                    print("energy calculation failed at init")
                    raise Exception("energy calculation failed at init")

                self.optimal_state = benzene
                self.optimal_energy = xtb_total_energy_difference(self.optimal_state['nodes'] + 1, self.optimal_state['nodes_xyz'] ,0.0 ,float('NaN'), self.solvation)
                if math.isnan(self.optimal_energy):
                    raise Exception("optimal energy is nan")

                #self.max_step_per_ep = 100
                self.reward_skip = 0
            if additional_seed is None:
                self.resetter = init_fun
            init_fun()

        elif mol_setup == "CH4":
            def init_fun():
                self.initial_energy = None
                mol_rng = np.random.default_rng(int(time.time()*1000000)) 
                if additional_seed is not None:
                    mol_rng = np.random.default_rng(additional_seed)

                self.initial_state = {'nodes':np.array([5,0,0,0,0]) }

                def gen_positions(num):
                    nodes_xyz_list_tmp = []
                    shell = 1

                    #nodes_xyz_list_tmp.append(np.random.random(list([num])+[3])*5.0-2.5)#fibonacci_latice(num,  5) + (np.random.random(list([num])+[3]) - 0.5) * 1.)
                    nodes_xyz_list_tmp.append(fibonacci_latice(num,  2) + (np.random.random(list([num])+[3]) - 0.5) * 1.)
                    return np.concatenate(nodes_xyz_list_tmp)


                running = 0
                limit = 100
                while running < limit and self.initial_energy is None:
                    
                    self.initial_state['nodes_xyz'] = gen_positions(self.initial_state['nodes'].shape[0])
                    self.initial_state['rot'] =  R.from_euler("xyz",np.zeros_like(self.initial_state['nodes_xyz'])).as_matrix()

                    nodes, nodes_xyz = abstract_mapping_to_node(self.initial_state['nodes'], self.initial_state['nodes_xyz'], self.initial_state['rot'], self.node_map)
                    self.initial_energy = xtb_total_energy_difference(nodes + 1, nodes_xyz ,0.0 ,float('NaN'), self.solvation)

                    if math.isnan(self.initial_energy) == False:
                        running = limit
                    else:
                        running = running + 1
                        self.initial_energy = None

                if 'fix_rot_mask' not in self.initial_state:
                    self.initial_state['fix_rot_mask'] = np.ones([self.initial_state['nodes'].shape[0],9])
                    self.initial_state['fix_rot_mask'][:,3:9]=0.

                if 'rot' not in self.initial_state:
                    self.initial_state['rot'] =  R.from_euler("xyz",np.zeros_like(self.initial_state['nodes_xyz'])).as_matrix()

                if self.initial_energy is None:
                    #self.initial_energy = 0.
                    print("energy calculation failed at init")
                    raise Exception("energy calculation failed at init")

                ch4 = molecule("CH4")

                self.optimal_state = {'nodes':(ch4.numbers-1), \
                                        'nodes_xyz':ch4.positions, \
                                        }
                
                self.optimal_energy = xtb_total_energy_difference(self.optimal_state['nodes'] + 1, self.optimal_state['nodes_xyz'] ,0.0 ,float('NaN'), self.solvation)
                if math.isnan(self.optimal_energy):
                    raise Exception("optimal energy is nan")

                #self.max_step_per_ep = 100
                self.reward_skip = 0
            if additional_seed is None:
                self.resetter = init_fun
            init_fun()
        
        elif mol_setup == "OHC_mix":
            def init_fun():
                #molecules = ["CH4", "H2CO","HCOOH","HCCl3","C2H2","CO","CO2","LiF","CN","H3CNH2", "NH3", "NO", "NO2"]
                self.initial_energy = None
                mol_rng = np.random.default_rng(int(time.time()*1000000)) 
                if additional_seed is not None:
                    mol_rng = np.random.default_rng(additional_seed)


                traj_path = os.path.join(os.getcwd(),"data", "oh_molecules.traj")
                trajreader = Trajectory(traj_path,'r')

                mol_index = mol_rng.integers(0,len(trajreader)-3)


                if self.mol_setup_index is not None:
                    mol_index = self.mol_setup_index%len(trajreader)

                if self.water_mol_number == 10:
                    opt_path = f"./minima_hopping/200_iter/test_10_water_{str(mol_index)}/result/minima.traj"
                    if os.path.exists(opt_path):
                        opt_trajreader = Trajectory(opt_path,'r')
                        opt_mol = opt_trajreader[-1]
                        min_value = float('inf')
                        for elem in opt_trajreader:
                            if elem.calc.results['energy']<min_value:
                                min_value = elem.calc.results['energy']
                                opt_mol = elem

                        
                        self.optimal_state = {'nodes':(opt_mol.numbers-1), \
                                        'nodes_xyz':opt_mol.positions, \
                                        }
                        
                elif self.water_mol_number == 50:
                    opt_path = f"./minima_hopping/k1500/test_50_water_{str(mol_index)}/result/minima.traj"
                    if os.path.exists(opt_path):
                        opt_trajreader = Trajectory(opt_path,'r')
                        opt_mol = opt_trajreader[-1]
                        
                        self.optimal_state = {'nodes':(opt_mol.numbers-1), \
                                        'nodes_xyz':opt_mol.positions, \
                                        }

                        
                
                atoms_list_tmp = trajreader[mol_index]
                trajreader.close()

                solve_nodes = []
                solve_nodes_xyz = []
                for i in range(len(atoms_list_tmp)):
                    solve_nodes.append(atoms_list_tmp.numbers[i]-1)
                    solve_nodes_xyz.append(atoms_list_tmp.positions[i])

                #move to center:
                center = np.zeros(3)
                for elem in solve_nodes_xyz:
                    center += elem
                center = center/float(len(solve_nodes_xyz))
                for i in range(len(solve_nodes_xyz)):
                    solve_nodes_xyz[i] = solve_nodes_xyz[i] - center


                water_num = 10#mol_rng.integers(3,4)
                if self.water_mol_number>0:
                    water_num = self.water_mol_number
                
                for i in range(water_num):
                    solve_nodes.append(102)

                self.initial_state = {'nodes':np.array(solve_nodes) }

                #water_positions = np.array([[]])


                def gen_positions(water_num_tmp):
                    nodes_xyz_list_tmp = []
                    shell = 1
                    while water_num_tmp > 0:
                        tmp = int((shell**2)*10)
                        if tmp > water_num_tmp:
                            tmp = water_num_tmp
                        nodes_xyz_list_tmp.append(fibonacci_latice(tmp,  5. + shell * 2.) + (np.random.random(list([tmp])+[3]) - 0.5) * 1.)
                        water_num_tmp = water_num_tmp-tmp

                        shell+=1
                    return np.concatenate(nodes_xyz_list_tmp)

                running = 0
                limit = 100
                while running < limit and self.initial_energy is None:
                    
                    self.initial_state['nodes_xyz'] = np.concatenate( (solve_nodes_xyz, gen_positions(water_num) ) )
                    if 'rot' not in self.initial_state:
                        self.initial_state['rot'] =  R.from_euler("xyz",np.zeros_like(self.initial_state['nodes_xyz'])).as_matrix()

                    nodes, nodes_xyz = abstract_mapping_to_node(self.initial_state['nodes'], self.initial_state['nodes_xyz'], self.initial_state['rot'], self.node_map)
                    self.initial_energy = xtb_total_energy_difference(nodes + 1, nodes_xyz ,0.0 ,float('NaN'), self.solvation)

                    if math.isnan(self.initial_energy) == False:
                        running = limit
                    else:
                        running = running + 1
                        self.initial_energy = None
                if self.initial_energy is None:
                    #self.initial_energy = 0.
                    print("energy calculation failed at init")
                    raise Exception("energy calculation failed at init")
                
                if 'fix_rot_mask' not in self.initial_state:
                    self.initial_state['fix_rot_mask'] = np.ones([self.initial_state['nodes'].shape[0],9])
                self.initial_state['fix_rot_mask'][0:len(solve_nodes_xyz),0:9] = 0.

                if self.optimal_state is not None:
                    self.optimal_energy = xtb_total_energy_difference(self.optimal_state['nodes'] + 1, self.optimal_state['nodes_xyz'] ,0.0 ,float('NaN'), self.solvation)
                    if math.isnan(self.optimal_energy):
                        raise Exception("optimal energy is nan")

                #self.max_step_per_ep = 100
                self.reward_skip = 0

            if additional_seed is None:
                self.resetter = init_fun
            init_fun()
        else:
            raise Exception("Invalid Test Setup ")

        self.nodes_num = self.initial_state['nodes'].shape[0]
        assert self.nodes_num == self.initial_state['nodes_xyz'].shape[0] and self.nodes_num == self.initial_state['rot'].shape[0]


        if self.max_number_of_nodes is not None:
            self.nodes_num = self.max_number_of_nodes

        if self.action_space is None:
            self.action_space = spaces.Box(low=np.array([[-1.,-1.,-1.,  -1.0,-1.0,-1.0, -1.,-1.,-1.]] * self.nodes_num),
                                        high=np.array([[1.,1.,1.,   1.0,1.0,1.0, 1.,1.,1.]] * self.nodes_num),
                                        shape=(self.nodes_num, 3 + 6), dtype=np.float32)


    

        if self.optimal_state is not None:   
            self.optimal_energy = xtb_total_energy_difference(self.optimal_state['nodes'] + 1, self.optimal_state['nodes_xyz'] ,0.0 ,float('NaN'), self.solvation)
            if math.isnan(self.optimal_energy):
                raise Exception("optimal energy is nan")

        if self.observation_space is None:
            self.observation_space = spaces.Dict(
                {
                    "nodes_xyz": spaces.Box(low=-100., high=100., shape=(self.nodes_num, 3), dtype=np.float32),
                    "nodes": spaces.Box(low=0, high=self.node_type_num, shape=(self.nodes_num,), dtype=np.int32),
                    "fix_rot_mask": spaces.Box(low=0, high=1.0, shape=(self.nodes_num,9), dtype=np.float32),
                    "timestep": spaces.Box(low=0, high=float('inf'), shape=None, dtype=np.int32),
                }
            )


        if nested_dict_obs:
            # Add dictionary observation inside observation space
            self.observation_space.spaces["nested-dict"] = spaces.Dict({"nested-dict-discrete": spaces.Discrete(4)})
            raise Exception("nested_dict_obs is not supported")
        
        self.current_state = None
        self.error_energy_reward = -0
        self.step_num = 0
        self.reward_sum = 0.
        self.partial_reward = 0.
        self.last_valid_energy = 0.
        self.failed_step_sum = 0

        self.coords_at_step = []
        self.energy_at_step = []
        self.terminated = False
        self.reward_list = []

    def seed(self, seed=None):
        if seed is not None:
            self.observation_space.seed(seed)

    def center_node_mapping(self,node_map):
        ret = {}
        for node_key in node_map:
            center = np.zeros(3)
            tmp = []
            for elem in node_map[node_key]:
                center += elem['xyz']
            center = center/float(len(node_map[node_key]))
            for elem in node_map[node_key]:
                tmp.append({'id':elem['id'] ,'xyz':elem['xyz'] - center})
            ret[node_key] = tmp

        return ret

    def partial_sum_reward_fun(self,nodes_xyz,rot = None):
        nodes = self.current_state['nodes']
        nodes, nodes_xyz = abstract_mapping_to_node(nodes, nodes_xyz, rot, self.node_map)
        energy_diff = xtb_total_energy_difference(nodes + 1, nodes_xyz ,self.last_valid_energy ,float('NaN'), self.solvation)
        if math.isnan(energy_diff) == False:
            self.last_valid_energy += energy_diff
            reward = -energy_diff
            if self.last_valid_energy < self.min_energy:
                self.min_energy = self.last_valid_energy
            return reward, energy_diff + self.last_valid_energy
        else:
            return None, None

    def calculate_translation_and_rot(self,nodes,nodes_xyz,rot,node_map,delta_xyz):
        index = 0
        ret_nodes_xyz = []
        ret_nodes_rot = []
        for i in range(len(nodes)):
            if nodes[i] in node_map:
                direction_sum = np.zeros(3)
                torque_sum = np.zeros(3)
                for elem in node_map[nodes[i]]:
                    direction_sum += delta_xyz[index]
                    torque_sum += np.cross(rot[i]@elem['xyz'],delta_xyz[index])

                    index +=1
                
                ret_nodes_xyz.append(nodes_xyz[i] + direction_sum)#/len(node_map[nodes[i]]))
                ret_nodes_rot.append(rot[i]@R.from_rotvec(torque_sum*10.).as_matrix())#/len(node_map[nodes[i]])).as_matrix())
            else:
                ret_nodes_xyz.append(nodes_xyz[i] + delta_xyz[index])
                ret_nodes_rot.append(rot[i])
                index +=1
        return np.array(ret_nodes_xyz), np.array(ret_nodes_rot)

    def step(self, action):
        rot_tmp = self.current_state['rot']
        action = SpecialPartialRotDiagGaussianDistribution.action_normalizer(th.tensor(action).unsqueeze(0))[0]
        pos_action = action[:,0:3][:self.current_state['nodes'].shape[0]].numpy() * self.pos_step_scale
        if self.use_time_embeddings != 0:
            pos_action = pos_action * np.exp(-self.use_time_embeddings * self.step_num)
        if self.use_rot:
            self.current_state['last_delta'] = pos_action
            if self.low_pass_filter_pos == 0.:
                nodes_xyz_tmp=self.current_state['nodes_xyz'] + pos_action * np.expand_dims(self.initial_state['fix_rot_mask'][:,0],-1)
            else:
                nodes_xyz_tmp=self.current_state['nodes_xyz'] + pos_action * np.expand_dims(self.initial_state['fix_rot_mask'][:,0],-1)
            rot_vec_a = action[:,3:6]
            rot_vec_b = action[:,6:9]
            new_rot = th.stack([rot_vec_a,rot_vec_b,rot_vec_a.cross(rot_vec_b, dim = -1)], dim = -1) 
            if self.low_pass_filter == 0. or self.step_num < 1:
                rot_tmp = new_rot
            else:

                c = np.empty((rot_tmp.shape[0] + new_rot.shape[0],3,3), dtype=self.initial_state['rot'].dtype)
                c[0::2] = rot_tmp
                c[1::2] = new_rot.numpy()

                slerp = Slerp(range(len(c)),R.from_matrix(c))
                rot_tmp = slerp([2*i + (1./(1. + self.low_pass_filter)) for i in range(rot_tmp.shape[0])]).as_matrix()
        else:
            nodes_xyz_tmp, rot_tmp = self.calculate_translation_and_rot(self.current_state['nodes'],self.current_state['nodes_xyz'],self.current_state['rot'],self.node_map,action)

        #reward = 1./np.maximum(0.1,((action - np.zeros_like(action))**2).sum())
        self.terminated = truncated = False
        #reward = np.sign(reward)*np.log(np.abs(reward)+1.0)
        reward = None
        if not ( self.reward_skip > 0 and ((self.step_num % self.reward_skip) != 0 and not (self.step_num >= self.max_step_per_ep)) ):
            reward, state_energy = self.partial_sum_reward_fun(nodes_xyz_tmp, rot_tmp)
            self.current_state['state_energy'] = state_energy

        if reward is not None:
            self.reward_sum += reward

        self.step_num = self.step_num + 1
        self.reward_list.append(reward)
        #reward = 0.
        
        if self.step_num > self.max_step_per_ep or (self.until_none_reward and (reward is None) ):
            #print(self.failed_step_sum)
            #print(action)
            if self.optimize_last_step is False:
                if reward is None:
                    print("last reward is None")
                    reward = 0#1*self.reward_sum #np.exp(self.reward_sum) -1.#0.#-np.abs(self.reward_sum)*0.9
                else:
                    reward = self.reward_sum#reward#1*self.reward_sum #np.exp(self.reward_sum) -1. #*(self.max_step_per_ep*0.1)
            else:
                if reward is None:
                    print("last reward is None")
                    reward = 0
                else:
                    nodes_r, nodes_xyz_r = abstract_mapping_to_node(self.current_state['nodes'], nodes_xyz_tmp, rot_tmp, self.node_map)
                    optimized_atoms = self.optimize(nodes_r, nodes_xyz_r, self.current_state['fix_rot_mask'][:,0])
                    reward = self.initial_energy - optimized_atoms.get_potential_energy()
                    self.reward_sum = reward
            self.terminated = True
            truncated = False

            if self.optimal_energy is not None:
                self.normalized_reward = self.reward_sum/(self.initial_energy - self.optimal_energy)
                

            print(f"reward list:  {self.reward_list}")
            print(f"sum reward: {str(self.reward_sum)}, normalized_reward: {self.normalized_reward},  last step energy: {self.initial_energy - self.reward_sum}, base energy: {self.optimal_energy}" )
        
        if reward is None:
            reward = 0.

        self.current_state['nodes_xyz'] = nodes_xyz_tmp

        self.current_state['rot'] = rot_tmp
        self.record_state()
        if self.use_rot:
            return {
                    "nodes": np.concatenate([self.current_state['nodes'],np.zeros(self.observation_space["nodes"].shape[0]-self.current_state["nodes"].shape[0])]),
                    'nodes_xyz': np.concatenate([self.current_state['nodes_xyz'], np.repeat([[float("inf"),float("inf"),float("inf")]], self.observation_space["nodes"].shape[0]-self.current_state["nodes"].shape[0], axis = 0)]),
                    'fix_rot_mask': np.concatenate([self.current_state['fix_rot_mask'],np.zeros([self.observation_space["nodes"].shape[0]-self.current_state["nodes"].shape[0],9])]),
                    'timestep':np.array([self.step_num]), 
                    }, reward, self.terminated, truncated, {}
        else:
            return {"nodes": self.current_state['nodes'], 'nodes_xyz': nodes_xyz_tmp}, reward, self.terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if self.resetter is not None:
            self.resetter()
        if seed is not None:
            self.observation_space.seed(seed)

        self.step_num = 0
        self.reward_sum = 0.
        self.failed_step_sum = 0
        self.coords_at_step = []
        self.energy_at_step = []
        self.terminated = False
        self.reward_list = []
        self.min_energy = float('inf')
        self.env_number_of_runs += 1

        limit = 100
        running = 0
        self.current_state = copy.deepcopy(self.initial_state)

        self.last_valid_energy = self.initial_energy
        self.current_state['state_energy'] = self.initial_energy

        if self.initial_energy is None:
            #self.initial_energy = 0.
            print("energy calculation failed")
            raise Exception("energy calculation failed")
        

        nodes, nodes_xyz = abstract_mapping_to_node(self.current_state['nodes'], self.current_state['nodes_xyz'], self.current_state['rot'], self.node_map)
        self.coords_at_step.append(copy.deepcopy(nodes_xyz))
        self.energy_at_step.append(self.current_state['state_energy'])

        if self.use_rot:
            return {
                    "nodes": np.concatenate([self.current_state['nodes'],np.zeros(self.observation_space["nodes"].shape[0]-self.current_state["nodes"].shape[0])]),
                    'nodes_xyz': np.concatenate([self.current_state['nodes_xyz'], np.repeat([[float("inf"),float("inf"),float("inf")]], self.observation_space["nodes"].shape[0]-self.current_state["nodes"].shape[0], axis = 0)]),
                    'fix_rot_mask': np.concatenate([self.current_state['fix_rot_mask'],np.zeros([self.observation_space["nodes"].shape[0]-self.current_state["nodes"].shape[0],9])]),
                    'timestep':np.array([0]),
                    }, {}
        else:
            return {"nodes": nodes, 'nodes_xyz': nodes_xyz}, {}
    
    def record_state(self):

        nodes, nodes_xyz = abstract_mapping_to_node(self.current_state['nodes'], self.current_state['nodes_xyz'], self.current_state['rot'], self.node_map)
        self.coords_at_step.append(copy.deepcopy(nodes_xyz))
        self.energy_at_step.append(self.current_state['state_energy'])

        
        if self.terminated:
            norm_reward = self.normalized_reward
            if self.normalized_reward is None:
                norm_reward = -1
            
            if self.write_data_to_file:
                #ase_atoms = ase.Atoms([ase.Atom('N', (0, 0, 0)), ase.Atom('N', (0, 0, d))])
                min_ene_coords_index = -1
                min_energy = float('inf')
                if self.mol_setup_index is None:
                    traj_path = os.path.join(os.getcwd(),"renders", f"example_{self.env_number_of_runs}.traj")
                else:
                    traj_path = os.path.join(os.getcwd(),"renders", f"example_{self.mol_setup_index}_{self.env_number_of_runs}.traj")
                traj = Trajectory(traj_path,'w')
                for t in range(len(self.coords_at_step)):
                    a_list = []
                    for i in range(len(nodes)):
                        a_list.append(ase.Atom(nodes[i] + 1, self.coords_at_step[t][i]))
                    if self.energy_at_step[t] is not None:
                        traj.write(ase.Atoms(a_list),energy = self.energy_at_step[t])
                        if self.energy_at_step[t] < min_energy:
                            min_energy = self.energy_at_step[t]
                            min_ene_coords_index=t
                    else:
                        traj.write(ase.Atoms(a_list))


                a_list = []
                if min_ene_coords_index !=-1:
                    for i in range(len(nodes)):
                        a_list.append(ase.Atom(nodes[i] + 1, self.coords_at_step[min_ene_coords_index][i]))
                    traj.write(ase.Atoms(a_list),energy = self.energy_at_step[min_ene_coords_index])
                    
                    try:
                        if False:
                            ret_model_best = self.optimize(nodes, self.coords_at_step[min_ene_coords_index], self.initial_state['fix_rot_mask'][:,0])
                            optimized_energy = xtb_total_energy_difference(ret_model_best.numbers, ret_model_best.positions ,0.0 ,float('NaN'), self.solvation)
                            ret = self.optimize(self.optimal_state['nodes'],  self.optimal_state['nodes_xyz'], self.initial_state['fix_rot_mask'][:,0])
                            optimized_optimal_energy = xtb_total_energy_difference(ret.numbers, ret.positions ,0.0 ,float('NaN'), self.solvation)
                            

                            nodes_init, nodes_xyz_init = abstract_mapping_to_node(self.initial_state['nodes'], self.initial_state['nodes_xyz'], self.initial_state['rot'], self.node_map)
                            initial_ret = self.optimize(nodes_init,  nodes_xyz_init, self.initial_state['fix_rot_mask'][:,0])
                            initial_optimized_energy = xtb_total_energy_difference(initial_ret.numbers, initial_ret.positions ,0.0 ,float('NaN'), self.solvation)
                            print(f"Optimized Initial energy: {initial_optimized_energy}, Optimized Initial normalized: {(self.initial_energy - initial_optimized_energy)/(self.initial_energy - self.optimal_energy)}")


                            print(f"Optimized Model energy: {optimized_energy}, Optimized Model normalized: {(self.initial_energy - optimized_energy)/(self.initial_energy - self.optimal_energy)}")
                            print(f"Optimized Optimal Energy: {optimized_optimal_energy} ,Optimized Optimal normalized: {(self.initial_energy - optimized_optimal_energy)/(self.initial_energy - self.optimal_energy)}")
                            traj.write(ret_model_best ,energy = optimized_energy)
                    except Exception as e:
                        print(e)
                    print(f"best normalized: {(self.initial_energy - self.energy_at_step[min_ene_coords_index])/(self.initial_energy - self.optimal_energy)}, index: {str(min_ene_coords_index)}")


                traj.close()


    def optimize(self,nodes,nodes_xyz,fix_mask):
        atoms = ase.Atoms(numbers = nodes+1, positions = nodes_xyz)
        
        constraints = [FixAtoms(indices=np.argwhere(fix_mask==0.).squeeze(axis=-1)),]

        index = 0
        for i in range(len(self.current_state['nodes'])):
            if self.current_state['nodes'][i] in self.node_map:
                connected_indices = []
                for elem in self.node_map[self.current_state['nodes'][i]]:
                    connected_indices.append(index)
                    index +=1

                comb = itertools.combinations(connected_indices, 2)
                for comb_index in comb:
                    constraints.append(Hookean(a1=comb_index[0], a2=comb_index[1],
                                            rt=np.linalg.norm(nodes_xyz[comb_index[0]]-nodes_xyz[comb_index[1]], axis=-1),
                                            k=15.))
            else:
                index +=1
        atoms.set_constraint(constraints)
        

        calc = TBLite(method = "GFN2-xTB")
        calc.set(verbosity = 0)
        calc.set(max_iterations = 1000)
        calc.set(accuracy = 1.0)
        #atoms.calc = ase.calculators.emt.EMT()
        atoms.calc = calc
        dyn = ase.optimize.BFGS(atoms)

        original_stdout = sys.stdout
        try:
            sys.stdout = open(os.devnull, 'w')
            dyn.run(fmax=0.05)
        finally:
            sys.stdout = original_stdout
        return atoms
        

    def render(self):
        self.write_data_to_file = True

    def get_normalized_reward(self):
        if self.normalized_reward is None:
            return 0.
        return self.normalized_reward
    
    def get_min_energy(self):
        return self.min_energy


