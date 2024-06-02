from tblite.ase import TBLite as TBLite
import argparse
import os
from stable_baselines3 import PPO
from atom_model.env import AtomMovingEnvTest
from atom_model.misc import calculate_hist
from atom_model.policy import EquiActorCriticPolicy
from atom_model.feature_extractor import EquiFeatureExtractor
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
import torch as th



def run_calculate_hist():
    additional_seed = int(time.time()*1000000)
    for i in range(0,1):
        env = AtomMovingEnvTest(use_rot = True, mol_setup = "OHC_mix", additional_seed = additional_seed,
                                            additional_params = {"low_pass_filter":0,
                                                                    "water_mol_number":10,
                                                                    "max_step_per_ep":100,
                                                                    "until_none_reward":True,
                                                                    "mol_setup_index":i})
        calculate_hist(env.initial_state['nodes'],
                            env.initial_state['nodes_xyz'],
                            env.initial_state['fix_mask'],
                            env.initial_state['rot'],
                            env.node_map, )


class OwnCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self):
        env = self.training_env
        res = env.env_method("get_normalized_reward")
        #res = env.unwrapped.get_normalized_reward()
        self.logger.record("train/normalized_reward",np.array(res).mean())
        res = env.env_method("get_min_energy")
        if res != float('inf'):
            self.logger.record("train/min_energy",np.array(res).min())



model_save_path = "./saved_models/"
N_ENVS = 10
def main():
    parser = argparse.ArgumentParser(description='Atom moving model')

    parser.add_argument('--subroutine', help='calling a subroutine')
    parser.add_argument('--model_class', help='Class of the model, now it is only PPO', choices=['PPO'], default='PPO')
    parser.add_argument('--only_predict', help='If true it is does not start training, it only preditcs with the current model', 
                         default=0, choices=[0,1], type=int)
    parser.add_argument('--single_env', help='Only ues single environment, no paralellism', default=1, choices=[0,1], type=int)
    parser.add_argument('--mol_setup', help='Name of the environment setup',  
                        choices=["CH4","OHC_mix","C6H6"])
    parser.add_argument('--low_pass_filter', help='Low pass filter parameter for rotations', default=0, choices=[0,0.5,0.25,1], type=float)
    parser.add_argument('--low_pass_filter_pos', help='Low pass filter parameter for position', default=0,  choices=[0], type=float)
    parser.add_argument('--water_mol_number', help='Number of water molecules surrounding the central atom', choices= [2,10,50], type=int)
    parser.add_argument('--max_step_per_ep', help='Maximum number of steps per episode',  default=100, type=int)
    parser.add_argument('--until_none_reward', help='Run until the first None rewards',  default=1, choices=[0,1], type=int)
    parser.add_argument('--model_file_name', help='name of the model file', type=str)
    parser.add_argument('--mol_setup_index', help='Molecule set index', type=int)
    parser.add_argument('--learning_rate', help='Learning rate', default=3e-5, type=float)
    parser.add_argument('--n_eval_episodes', help='eval episode number', default=1, type=int)
    parser.add_argument('--hidden_dim', help='size of the hidden sate for PaiNN', default=128, type=int)
    parser.add_argument('--layer_num', help='Number of layers/interactions for PaiNN', default=5, type=int)
    parser.add_argument('--cutoff', help='cutoff distance for PaiNN', default=10.0, type=float)
    parser.add_argument('--rot_cutoff', help='distance for rotation PaiNN network', type=float)
    parser.add_argument('--pos_step_scale', help='scale for the output max distance in PaiNN network', default=0.1, type=float)
    parser.add_argument('--use_time_embeddings', help='Time embedding parameter', default=0., type=float)
    parser.add_argument('--max_number_of_nodes', help='Set max number of nodes for efficiently training variable size inputs', type=int)
    parser.add_argument('--step_base', help='step base number for rollouts', default=128, type=int)
    parser.add_argument('--n_epochs', help='epoch number for training', default=30, type=int)
    parser.add_argument('--optimizer', help='Optimizer',choices=['Adam','AdamW'], default='Adam')
    parser.add_argument('--optimize_last_step', help='Set wheter optimize the last step and give energy', default=0, choices=[0,1], type=int)
    
    


    args = parser.parse_args()

    if args.subroutine is not None:
        globals()[args.subroutine]()
        return


    
    args.only_predict = bool(args.only_predict)
    args.single_env = bool(args.single_env)
    args.optimize_last_step = bool(args.optimize_last_step)

    
    if args.mol_setup is None:
        raise Exception("You have to define a mol_setup")

    if args.water_mol_number is None:
         raise Exception("Water should be defined")

    args.until_none_reward = bool(args.until_none_reward)

    if args.model_file_name is None:
        raise Exception("Need a model name")

    print(f"Model name: {args.model_file_name}")

    optimizer = th.optim.Adam
    if args.optimizer == "Adam":
        optimizer = th.optim.Adam
    elif args.optimizer == "AdamW":
        optimizer = th.optim.AdamW
    #return
    if os.path.exists(model_save_path) is False:
        os.makedirs(model_save_path)

    tensorboard_log = "./tb_log/"

    feature_extractor_file_name = "c"#"feature_extractor_model_128_formic_acid_sigmoid_2.ply"

    use_rot = True

    hidden_dim = None
    extractor = None
    if os.path.exists(os.path.join(model_save_path,feature_extractor_file_name)):
        saved_model = th.load(os.path.join(model_save_path,feature_extractor_file_name))
        dummy_env = retenv()
        hidden_dim = saved_model['init_params']['hidden_size']
        extractor = EquiFeatureExtractor(dummy_env.observation_space, hidden_dim, False, dummy_env.action_space, 'distinct',True)
        extractor.load_state_dict(saved_model["state_dict"])
        print("pretrained model loaded")


    kwargs = {}
    step_base = args.step_base
    n_steps = step_base * 10*1


    if hidden_dim is None:
        hidden_dim = 128

    model_class = args.model_class
    if model_class in {'PPO'}:
        model_class = PPO
        kwargs = dict(
            n_steps=step_base,
            policy_kwargs=dict(
                net_arch=[32],
                features_extractor_kwargs=dict(
                    hidden_size=hidden_dim,
                    layer_num = args.layer_num,
                    cutoff =  args.cutoff,
                    use_time_embeddings = args.use_time_embeddings,
                    use_rot = True,
                    rot_cutoff = args.rot_cutoff
                                               ),
                model_class_name = model_class.__name__,
                use_rot = use_rot,
                optimizer_class = optimizer,
                optimizer_kwargs = None,
            ),
        )
    else:
        raise Exception("No other model then PPO")

    #model = model_class("MultiInputPolicy", env, gamma=0.5, seed=1, **kwargs)
    

    MAX_RUN_PER_ENV = 10*50

    for run_per_env in range(MAX_RUN_PER_ENV):
        #try:
            additional_seed = int(time.time()*1000000)
            if args.max_number_of_nodes is not None:
                additional_seed = None

            def retenv():
                return AtomMovingEnvTest(use_rot = use_rot, mol_setup = args.mol_setup, additional_seed = additional_seed, 
                                         additional_params = {"low_pass_filter":args.low_pass_filter,
                                                              "water_mol_number":args.water_mol_number,
                                                              "max_step_per_ep":args.max_step_per_ep,
                                                              "until_none_reward":args.until_none_reward,
                                                              "low_pass_filter_pos":args.low_pass_filter_pos,
                                                              "mol_setup_index":args.mol_setup_index,
                                                              "pos_step_scale": args.pos_step_scale,
                                                              "max_number_of_nodes": args.max_number_of_nodes,
                                                              "use_time_embeddings": args.use_time_embeddings,
                                                              "optimize_last_step": args.optimize_last_step})

            if args.single_env == True:
                env = retenv()
            else:
                env = make_vec_env(retenv, N_ENVS, 1, 0, None, None,None,SubprocVecEnv)
                print("multiple env")

            n_epochs = args.n_epochs
            model = model_class(EquiActorCriticPolicy, env, gamma=0.5, seed=1, verbose=1,batch_size=64, n_epochs = n_epochs, clip_range_vf = 0.2, learning_rate=args.learning_rate, tensorboard_log = tensorboard_log, **kwargs)

            
            if extractor is not None:
                model.policy.features_extractor = extractor.to(model.policy.features_extractor.device)

            if os.path.exists(os.path.join(model_save_path,args.model_file_name + '__temp.ply')):
                if os.path.exists(os.path.join(model_save_path,args.model_file_name + '.ply')):
                    raise f"Safe save was unsuccesful for file, it is maybe corrupted: {args.model_file_name + '__temp.ply' }"
                else:
                    os.rename(os.path.join(model_save_path,args.model_file_name + '__temp.ply'),os.path.join(model_save_path,args.model_file_name + '.ply'))

            if os.path.exists(os.path.join(model_save_path,args.model_file_name + '.ply')):
                modified_kwargs = {"observation_space": env.observation_space,
                                    "action_space": env.action_space,
                                    "learning_rate":args.learning_rate,
                                    "n_epochs" : n_epochs}
                model = model.load(os.path.join(model_save_path,args.model_file_name + '.ply'),env, **modified_kwargs)
                #if extractor is not None:
                #    model.policy.features_extractor = extractor
                #model.set_env(env)
            else:
                if args.only_predict:
                    raise Exception("Only predict without model")

            MAX_RUN = 1*1
            reset_num_timesteps = False
            for run in range(MAX_RUN):

                if args.only_predict is False:
                    model.learn(total_timesteps=n_steps,tb_log_name=args.model_file_name, reset_num_timesteps=reset_num_timesteps, callback = OwnCallback())

                    print("model save started")
                    model.save(os.path.join(model_save_path,args.model_file_name + '__temp.ply'))
                    if os.path.exists(os.path.join(model_save_path,args.model_file_name + '.ply')):
                        os.remove(os.path.join(model_save_path,args.model_file_name + '.ply'))
                    os.rename(os.path.join(model_save_path,args.model_file_name + '__temp.ply'),os.path.join(model_save_path,args.model_file_name + '.ply'))
                    print("model is saved")

                env = make_vec_env(retenv, 1, 1, 0, None, None,None,DummyVecEnv)
                #if args.use_time_embeddings != 0:
                #    model.policy.features_extractor.use_time_embeddings = args.use_time_embeddings
                mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.n_eval_episodes, warn=False, render = True)

                if args.only_predict is False:
                    #logger = utils.configure_logger(False, tensorboard_log = tensorboard_log, tb_log_name = rl_model_file_name, reset_num_timesteps = reset_num_timesteps)
                    model.logger.record("eval/mean_reward",mean_reward)
                    model.logger.record("eval/std_reward",std_reward)
                    res = model.env.env_method("get_normalized_reward")
                    model.logger.record("eval/normalized_reward",np.array(res).mean())
                    model.logger.dump(step=model.num_timesteps)

                print(f"eval_mean_reward: {mean_reward}, eval_std_reward: {std_reward}")

                if args.only_predict is True:
                    break
            if args.only_predict is True:
                    break
        #except:
        #    pass

if __name__ == '__main__':
    main()