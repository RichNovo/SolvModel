from typing import Dict, List, Literal, Tuple, Type, Union, Optional, Any

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space, get_action_dim
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device


from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
)

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    NatureCNN,
    create_mlp
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.policies import BaseModel

from stable_baselines3.common.type_aliases import PyTorchObs

from .painn_model import PainnModel
import numpy as np


def reshape_with_action_space_variable_dim(action,action_space):
    first_dim = int(np.round(np.product(action.shape[1:])/np.product(action_space.shape[1:])))
    reshaped = action.reshape([action.shape[0]] +[first_dim] + list(action_space.shape[1:]))
    return reshaped

class EquiFeatureExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_size: int = 128,
        normalized_image: bool = False,
        action_space = None,
        ext_type: Literal["distinct"] = "distinct",
        use_rot = False,
        layer_num = 5,
        cutoff =  10.0,
        rot_cutoff = None,
        use_time_embeddings = 0,
        cnn_output_dim = None
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        if rot_cutoff is None:
            rot_cutoff = cutoff

        self.rot_mask_cache = None

        self.use_rot = use_rot
        self.action_space = action_space
        self.ext_type = ext_type
        self.layer_num = layer_num
        self.cutoff = cutoff
        self.rot_cutoff = rot_cutoff
        self.use_time_embeddings = use_time_embeddings
        extractors: Dict[str, nn.Module] = {}


        total_concat_size = 0

        self.extractors = nn.ModuleDict(extractors)

        self.sigmoid_out = nn.Sigmoid()
        self.softmax_scaling = nn.Softmax()
        self.temperature = 0.01

        self.node_type_num = observation_space["nodes"].high[0]+1

        self.hidden_size=hidden_size

        self.sum_vector_out = nn.Linear(self.hidden_size //2,1, bias=False)
        self.sum_scalar_out = nn.Linear(self.hidden_size //2,1, bias=False)
        self.sum_scalar_out_val = nn.Linear(self.hidden_size //2,1, bias=False)

        self.sum_rot_part = nn.Linear(self.hidden_size // 2,3, bias=False)

        self.sum_axis_part = nn.Linear(self.hidden_size // 2,1, bias=False)

        self.sum_rot_vec = nn.Linear(self.hidden_size // 2,2, bias=False)

        self.sum_rot_vec_single = nn.Linear(self.hidden_size // 2,1, bias=False)


        self.painn_model = PainnModel(self.hidden_size,self.layer_num,self.cutoff ,self.node_type_num, self.use_time_embeddings,self.device)
        self.painn_model_val = PainnModel(self.hidden_size,self.layer_num,self.cutoff ,self.node_type_num, self.use_time_embeddings,self.device)
        if self.use_rot:
            self.painn_model_rot = PainnModel(self.hidden_size,self.layer_num,self.rot_cutoff, self.node_type_num, 0,self.device)



        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:

        nodes_scalar, nodes_vector = self.painn_model.my_forward(observations["nodes"].int(), observations["nodes_xyz"], observations["timestep"])

        
        if th.isnan(nodes_vector).any():
            raise Exception("nodes_vector values are Nan")
        if th.isnan(nodes_scalar).any():
            raise Exception("nodes_scalar values are Nan")
        
        if (nodes_vector>10e+10).any():
            raise Exception("nodes_vector values are too large")
        
        if (nodes_scalar>10e+10).any():
            raise Exception("nodes_scalar values are too large")

        if True:
            nodes_vector = self.sum_vector_out(nodes_vector).squeeze(-1)
        else:
            nodes_vector = nodes_vector[:,:,:,0]*nodes_scalar[:,:,0].unsqueeze(-1)

        if False:
            nodes_vector = th.nn.functional.normalize(nodes_vector, dim = -1) * self.sigmoid_out(th.norm(nodes_vector, dim = -1).unsqueeze(-1))


        


        if self.use_rot:

            rot_part, rot_vec_part = self.painn_model_rot.my_forward(observations["nodes"].int(), observations["nodes_xyz"], observations["timestep"])

            if True:
                rot_vec_part = self.sum_rot_vec(rot_vec_part).squeeze(-1)
                rot_vec_part = rot_vec_part.permute(*range(rot_vec_part.dim()-2),-1,-2) # Use permute, the output of linear has dimension [1, 30, 3, 2] but we need [1, 30, 2, 3]
                
                ###rot_vec_part = th.nn.functional.normalize(rot_vec_part, dim = -1)

                rot_vec_part = rot_vec_part# * observations["rot_mask"].unsqueeze(-1).unsqueeze(-1).repeat([1,1,2,3])
                rot_vec_part = th.flatten(rot_vec_part, start_dim=-2)
            else:
                rot_vec_part = th.cat([rot_vec_part[:,:,:,0]*rot_part[:,:,0].unsqueeze(-1),rot_vec_part[:,:,:,1]*rot_part[:,:,1].unsqueeze(-1)], dim =-1)


        #nodes_scalar = self.sum_scalar_out(nodes_scalar).squeeze(-1)


        if self.use_time_embeddings != 0 and "timestep" in observations:
            nodes_vector = th.nn.functional.normalize(nodes_vector, dim = -1)
        else:
            nodes_vector = nodes_vector#th.nn.functional.normalize(nodes_vector, dim = -1)# * self.sigmoid_out(th.norm(nodes_vector, dim = -1).unsqueeze(-1))

        nodes_scalar_val, nodes_vector_val = self.painn_model_val.my_forward(observations["nodes"].int(), observations["nodes_xyz"], observations["timestep"])
        nodes_scalar_val = self.sum_scalar_out_val(nodes_scalar_val).squeeze(-1)


        if th.isnan(nodes_scalar_val).any():
            raise Exception("nodes_scalar_val is Nan")
        
        if th.isnan(nodes_vector_val).any():
            raise Exception("nodes_vector_val is Nan")
        
        if self.use_rot:
            nodes_vector = th.cat([nodes_vector,rot_vec_part], dim = -1)
            nodes_vector = nodes_vector * observations["fix_rot_mask"]
        else:
            nodes_vector = nodes_vector * observations["fix_mask"].unsqueeze(-1).repeat([1,1,3])

        if th.isnan(nodes_vector).any():
            raise Exception("final nodes_vector is Nan")

        nodes_scalar_val = nodes_scalar_val * observations["fix_rot_mask"][:,:,0]

        return th.flatten(nodes_vector, start_dim=1), nodes_scalar_val.sum(dim=-1).unsqueeze(1)

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")