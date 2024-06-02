# SolvModel

This is the official implementation of SolvModel

## Setup

Create and activate a virtual environment:
    $ conda create -n solvmodel python=3.12 tblite-python
Install the requirements:
    $ pip install -r requirements.txt 

Example training using methane:
 $ python main.py --model_class PPO --only_predict 0 --single_env 1 --mol_setup CH4 --water_mol_number=2 --until_none_reward 0 --max_step_per_ep 120 --rot_cutoff 10.0 --pos_step_scale 0.1 --max_number_of_nodes 5 --step_base 1024 --learning_rate 3e-6 --optimizer AdamW --model_file_name CH4_model

Example training using OHC molecules:

 $ python main.py --model_class PPO --only_predict 0 --single_env 0 --mol_setup OH_mix --water_mol_number=10 --until_none_reward 0 --max_step_per_ep 120 --rot_cutoff 10.0 --pos_step_scale 0.1 --max_number_of_nodes 5 --step_base 128 --learning_rate 3e-6 --optimizer AdamW --model_file_name OHC_model