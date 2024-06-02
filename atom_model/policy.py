from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.sac.policies import SACPolicy, Actor as SACActor
from stable_baselines3.td3.policies import TD3Policy, Actor as TD3Actor

from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor
)
from stable_baselines3.common.preprocessing import get_action_dim
from torch.distributions import Normal
import math

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
    sum_independent_dims
)

import collections

import numpy as np
import torch as th

import warnings

from .feature_extractor import EquiFeatureExtractor
from scipy.spatial.transform import Rotation as R


class SpecialPartialRotDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """
    ones_cache = th.tensor([])
    zeros_cache = th.tensor([])

    def __init__(self, action_space: int):
        self.action_space = action_space
        super().__init__(get_action_dim(self.action_space))
        self.mean_actions = None
        self.log_std = None
        self.rsample = None

    def action_normalizer(action):
        #reshaped = action.reshape([action.shape[0]]+list(self.action_space.shape))

        rot_vec_a = th.nn.functional.normalize(action[:,:,3:6],dim = -1 )
        rot_vec_b = th.nn.functional.normalize(action[:,:,6:9], dim = -1)
        rot_vec_c = th.nn.functional.normalize(rot_vec_a.cross(rot_vec_b, dim = -1), dim = -1)
        rot_vec_b = th.nn.functional.normalize(rot_vec_c.cross(rot_vec_a, dim = -1), dim = -1)

        norm = th.linalg.norm(action[:,:,0:3], dim = -1)
        new_norm = th.max(norm, th.tensor(1.))
        new_translation = action[:,:,0:3]/new_norm.unsqueeze(-1)

        mean_actions = th.cat([new_translation,rot_vec_a,rot_vec_b],dim=-1)
        #mean_actions = mean_actions.reshape(action.shape)

        return mean_actions

    def mat_action_to_rot(self, action):
        reshaped = action.reshape([action.shape[0]]+list(self.action_space.shape))

        mean_actions = SpecialPartialRotDiagGaussianDistribution.action_normalizer(reshaped).reshape(action.shape)
        return mean_actions
    
    def proba_distribution(
        self: DiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor, obs: PyTorchObs
    ) -> DiagGaussianDistribution:
        if mean_actions.shape != SpecialPartialRotDiagGaussianDistribution.ones_cache.shape or \
        mean_actions.device != SpecialPartialRotDiagGaussianDistribution.ones_cache.device:
            SpecialPartialRotDiagGaussianDistribution.ones_cache = th.ones_like(mean_actions)
            SpecialPartialRotDiagGaussianDistribution.zeros_cache = th.zeros_like(mean_actions)

        action_std = SpecialPartialRotDiagGaussianDistribution.ones_cache * log_std.exp()
        self.distribution = Normal(SpecialPartialRotDiagGaussianDistribution.zeros_cache, action_std)
        self.mean_actions = mean_actions
        self.obs = obs
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        
        #log_prob = self.distribution.log_prob(actions - self.mean_actions)
        #log_prob = self.distribution.log_prob(actions - self.mean_actions)
        #(self.mean_actions+(mean_detached-self.mat_action_to_rot(mean_detached)))
        #log_prob = self.distribution.log_prob(actions - (self.mean_actions+(self.mat_action_to_rot(mean_actions_detached))-mean_actions_detached))
        """log_prob = self.distribution.log_prob(actions - self.mean_actions) \
                - self.distribution.log_prob(actions - mean_actions_detached) \
                + self.distribution.log_prob(actions -self.mat_action_to_rot(mean_actions_detached))"""
        
        mean_actions_detached = self.mean_actions.detach()
        var = self.distribution.scale**2
        log_scale = (
            self.distribution.scale.log()
        )
        tmp = self.mat_action_to_rot(actions)
        log_prob = (
            (-((tmp - self.mean_actions) ** 2) + # -actions**2 - self.mean_actions**2 +
            ((tmp - mean_actions_detached) ** 2) -  # + 2*actions * (self.mat_action_to_rot(mean_actions_detached) -  mean_actions_detached + self.mean_actions) + mean_actions_detached**2
            ((actions-self.mat_action_to_rot(mean_actions_detached)) ** 2)) # - self.mat_action_to_rot(mean_actions_detached)**2)
            / (2 * var) 
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
        
        log_prob = log_prob * self.obs['fix_rot_mask'].flatten(1)
        return sum_independent_dims(log_prob)

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        mean_detached = self.mean_actions.detach()
       
        self.rsample = self.distribution.sample()
        transformed_action = self.mat_action_to_rot(mean_detached)+self.rsample

        #transformed_action = self.mat_action_to_rot(mean_detached+self.distribution.rsample())
        
        return self.mean_actions + (transformed_action - mean_detached)

    def mode(self) -> th.Tensor:
        mean_detached = self.mean_actions.detach()
        #transformed_action = self.mat_action_to_rot_2(mean_detached, self.distribution.mean)
        transformed_action = self.mat_action_to_rot(mean_detached)
        return self.mean_actions + (transformed_action - mean_detached)

class EquiActorCriticPolicy(BasePolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = None,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: th.Tensor = None,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = EquiFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        model_class_name = 'PPO',
        n_critics: int = 2,
        clip_mean: float = 2.0,
        use_rot = False,
    ):
        
        ###init
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        share_features_extractor = False
        net_arch = []
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-6 #PPO eps
        clip_mean = 2
        if len(action_space.high[0])>3:
            log_std_init=th.flatten(th.tensor(np.array([
                #np.log((action_space.high - action_space.low) * np.array([1.0/4.,1.0/4.,1.0/4.,1.0/10.,1.0/10.,1.0/10.,1.0/10.,1.0/10.,1.0/10.]))
                np.log(np.ones_like(action_space.high)*np.array([1.0/4.,1.0/4.,1.0/4.,1.0/10.,1.0/10.,1.0/10.,1.0/10.,1.0/10.,1.0/10.]))
            ])), start_dim=1)
        else:
            log_std_init=th.flatten(th.tensor([np.log((action_space.high - action_space.low) * 1.0/4.)]), start_dim=1)
        ###


        self.model_class_name = model_class_name
        self.use_rot = use_rot
        
        if activation_fn == None:
            activation_fn: Type[nn.Module] = nn.Tanh
        
        if log_std_init is None:
            log_std_init: float = 0.0

        if share_features_extractor == None:
            share_features_extractor: bool = True


        self.lr_schedule = lr_schedule
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.use_sde = use_sde
        self.log_std_init = log_std_init
        self.full_std = full_std
        self.use_expln = use_expln
        #self.squash_output = squash_output
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.clip_mean = clip_mean


        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        if "ext_type" not in features_extractor_kwargs:
            features_extractor_kwargs["ext_type"] = "distinct"
            features_extractor_kwargs["use_rot"] = use_rot

        features_extractor = None
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
            features_extractor = features_extractor
        )

        self.features_extractor_kwargs["action_space"] = self.action_space
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        #if self.share_features_extractor:
        self.pi_features_extractor = self.features_extractor
        self.vf_features_extractor = self.features_extractor
        #else:
        #    self.pi_features_extractor = self.features_extractor
        #    self.vf_features_extractor = self.make_features_extractor()

        dist_kwargs = None

        assert not (squash_output and not use_sde), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        #self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        self.action_dist = None
        self.mlp_extractor = nn.Sequential()

        self.actor_features_extractor_kwargs = features_extractor_kwargs.copy()
        self.actor_features_extractor_kwargs["ext_type"] = "policy"

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": [],
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "features_extractor_class": features_extractor_class,
            "features_extractor_kwargs": self.actor_features_extractor_kwargs
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": [],
                "share_features_extractor": share_features_extractor,
            }
        )

        ###
        self.to(device)
        self._build(self.lr_schedule)
        #self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        #self._build_mlp_extractor()

        #latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if self.use_rot:# and ((self.use_rot is True) is False):
            self.action_dist = SpecialPartialRotDiagGaussianDistribution(self.action_space)
        elif isinstance(self.action_space, spaces.Box):
            cls = StateDependentNoiseDistribution if self.use_sde else DiagGaussianDistribution
            self.action_dist = cls(get_action_dim(self.action_space))
        elif isinstance(self.action_space, spaces.Discrete):
            self.action_dist = CategoricalDistribution(int(self.action_space.n))
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.action_dist = MultiCategoricalDistribution(list(self.action_space.nvec))
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        if isinstance(self.action_dist, SpecialPartialRotDiagGaussianDistribution):
            self.action_net, self.log_std = nn.Sequential(), self.log_std_init.to(self.device)
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = nn.Sequential(), self.log_std_init.to(self.device)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = nn.Sequential()
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = nn.Sequential()
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Sequential()#nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi, latent_vf = features
        return self._get_action_dist_from_latent(latent_pi, obs)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_pi, latent_vf  = features#self.mlp_extractor.forward_critic(features)
        return latent_vf
        #return ActorCriticPolicy.predict_values(self, obs)
    
    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = features
        else:
            latent_pi, latent_vf = features
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def extract_features(  # type: ignore[override]
        self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:

        if self.share_features_extractor:
            return super().extract_features(obs, self.features_extractor if features_extractor is None else features_extractor)
        else:
            features = super().extract_features(obs, self.features_extractor if features_extractor is None else features_extractor)
            if len(features) == 2:
                return features
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

        
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                observation_space = self.observation_space,
                action_space = self.action_space,
                lr_schedule = self.lr_schedule,
                net_arch = self.net_arch,
                activation_fn = self.activation_fn,
                ortho_init = self.ortho_init,
                use_sde = self.use_sde,
                log_std_init = self.log_std_init, 
                full_std = self.full_std,
                use_expln = self.use_expln,
                squash_output = self.squash_output,
                features_extractor_class = self.features_extractor_class,
                features_extractor_kwargs = self.features_extractor_kwargs,
                share_features_extractor = self.share_features_extractor,
                normalize_images = self.normalize_images,
                optimizer_class = self.optimizer_class,
                optimizer_kwargs = self.optimizer_kwargs,
                n_critics = self.n_critics,
                clip_mean = self.clip_mean,
                model_class_name = self.model_class_name,
                use_rot = self.use_rot
            )
        )

        return data
        
    def reset_noise(self, n_envs_or_batch_size: int = 1) -> None:
        return ActorCriticPolicy.reset_noise(self, n_envs_or_batch_size)


    def log_std_using_timestep(self,obs):
        timestep_scale = -self.features_extractor_kwargs['use_time_embeddings']*obs['timestep']
        #log_std = self.log_std.repeat([timestep_scale.shape[0],1]) + th.cat([timestep_scale.repeat([1,3]),th.zeros([timestep_scale.shape[0],6], device = timestep_scale.device)],dim = 1).repeat([1,self.action_space.shape[0]])
        #log_std = self.log_std.repeat([timestep_scale.shape[0],1]) + timestep_scale.repeat([1,9]).repeat([1,self.action_space.shape[0]])
        log_std = self.log_std.repeat([timestep_scale.shape[0],1]) + th.cat([timestep_scale.repeat([1,3]),
                                        th.min(timestep_scale.repeat([1,6]),-4.*th.ones([timestep_scale.shape[0],6], device = timestep_scale.device))],dim = 1).repeat([1,self.action_space.shape[0]])
        return log_std


    def _get_action_dist_from_latent(self, latent_pi: th.Tensor,obs: PyTorchObs) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        log_std = self.log_std
        #if 'use_time_embeddings' in self.features_extractor_kwargs and self.features_extractor_kwargs['use_time_embeddings'] > 0:
        #    log_std = self.log_std_using_timestep(obs)

        if isinstance(self.action_dist, SpecialPartialRotDiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, log_std, obs)
        elif isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def forward(self, obs: PyTorchObs, deterministic: bool = None) -> th.Tensor:

        if deterministic is None:
            deterministic = False

        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = features
        else:
            latent_pi, latent_vf = features
        # Evaluate the values for the given observations
        values = latent_vf
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob
        #return ActorCriticPolicy.forward(self, obs, deterministic)
    

    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        if deterministic is None:
            deterministic = False
        return ActorCriticPolicy._predict(self, obs, deterministic)
        

    def set_training_mode(self, mode: bool) -> None:
        ActorCriticPolicy.set_training_mode(self, mode)

