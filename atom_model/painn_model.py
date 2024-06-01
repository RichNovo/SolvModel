from typing import Tuple, List, Optional

import ase
import ase.data
import numpy as np
import torch
import torch.distributions
from torch import nn


from .painn import layer_painn as layer
from .painn import data_painn

from scipy.spatial.transform import Rotation as R

#https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/diffusionmodules/model.py
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 2

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

class AbstractMolDataList():
    
    def __init__(self) -> None:
        self.id_list = []
        #self.pos_list = ase.Atoms()
        self.pos_list = []
        self.rot_list = []
        self.timestep_list = []
        self._cellobj = ase.cell.Cell.new()

    def append(self, id, pos, rot,timestep):
        self.id_list.append(id)
        #self.pos_list.append(Mol(symbol=ase.data.chemical_symbols[0], position = pos))
        self.pos_list.append(pos)
        self.rot_list.append(rot)
        self.timestep_list.append(timestep)

    def get_edges_simple(self):
        #self.pos_list.get_edges_simple()
        raise "Error"
        #return None, None
    
    def get_pbc(self):
        """ peridodic boundary condition """
        return torch.zeros(3).bool()
        #self.pos_list.get_pbc()

    def get_atomic_numbers(self):
        return self.id_list

    def get_positions(self):
        return self.pos_list
        #return self.pos_list.get_positions()
    
    def get_rot_list(self):
        return self.rot_list
    
    def get_timestep_list(self):
        return self.timestep_list
    
    def get_cell(self):
        return self._cellobj
        #return np.array(self.pos_list.get_cell())
        #return [[0.,0.,0.], [0.,0.,0.], [0.,0.,0.]]


class PainnModel(nn.Module):
    def __init__(
        self,
        network_width: int,
        num_interactions: int,
        cutoff: float,
        num_embeddings, #= 119,
        use_time_embeddings:float,# = False,
        device: torch.device# = None,
    ):
        super().__init__()
        # Internal action: stop, focus, element, distance, angle, dihedral, kappa
        self.device = device
        self.use_rot_encoding = False
        self.use_time_embeddings = use_time_embeddings

        self.num_afeats = network_width // 2
        self.num_latent_beta = network_width // 4
        self.num_latent = self.num_afeats + self.num_latent_beta

        # PaiNN variables:
        self.transformer = data_painn.TransformAtomsObjectsToGraphXyz(cutoff=cutoff)
        self.hidden_state_size = network_width // 2
        if device==torch.device("cuda"):
            self.pin=True
        else:
            self.pin=False
        self.cutoff = cutoff
        self.distance_embedding_size = 20

        #num_embeddings = 119  # atomic numbers + 1
        #legacy model compatibility TODO: remove:
        if num_embeddings<119:
            num_embeddings = 119
        edge_size = self.distance_embedding_size

        # Setup atom embeddings
        self.atom_embeddings = nn.Embedding(num_embeddings, self.hidden_state_size)

        # Setup interaction networks
        self.interactions = nn.ModuleList(
            [
                layer.PaiNNInteraction(self.hidden_state_size, edge_size, self.cutoff, self.use_time_embeddings != 0)
                for _ in range(num_interactions)
            ]
        )
        self.scalar_vector_update = nn.ModuleList(
            [layer.PaiNNUpdate(self.hidden_state_size) for _ in range(num_interactions)]
        )

        if self.use_time_embeddings != 0:
            self.scalar_message_mlp = nn.Sequential(
                nn.Linear(self.hidden_state_size, self.hidden_state_size),
                nn.SiLU(),
                nn.Linear(self.hidden_state_size, self.hidden_state_size),
            )

            self.reducer = nn.Sequential(
                nn.Linear(self.hidden_state_size * 2 , self.hidden_state_size),
            )



        self.to(device)


    def my_forward(self, x_atoms, x_pos, timestep, x_rot = None) -> torch.Tensor:

        graph_states = []
        for batch_idx in range(len(x_atoms)):
            mol_data_list = AbstractMolDataList() #atom_data_list = ase.Atoms()
            """atom_list = x_atoms[batch_idx]
            x_pos_list = x_pos[batch_idx]
            if x_rot is not None:
                x_rot_list = x_rot[batch_idx]

            for atom_idx in range(len(atom_list)):
                #new_atom = ase.Atom(symbol=ase.data.chemical_symbols[atom_list[atom_idx]], position = x_pos_list[atom_idx])
                if x_rot is not None:
                    mol_data_list.append(atom_list[atom_idx],x_pos_list[atom_idx],x_rot_list[atom_idx])#atom_data_list.append(new_atom)
                else:
                    mol_data_list.append(atom_list[atom_idx],x_pos_list[atom_idx],None)#atom_data_list.append(new_atom)
            """
            mol_data_list.id_list = x_atoms[batch_idx]
            mol_data_list.pos_list = x_pos[batch_idx]
            if self.use_time_embeddings != 0:
                mol_data_list.timestep_list = timestep[batch_idx]
            if x_rot is not None:
                self.use_rot_encoding = True
                mol_data_list.rot_list = x_rot[batch_idx]


            graph_states.append(self.transformer(mol_data_list))#graph_states.append(self.transformer(atom_data_list))

        
        self.device = self.atom_embeddings.weight.device
        if self.device==torch.device("cuda"):
            self.pin=True
        else:
            self.pin=False
        
        batch_host = data_painn.collate_atomsdata(graph_states, pin_memory=self.pin)

        batch = {
            k: v.to(device=self.device, non_blocking=True)
            for (k, v) in batch_host.items()
        }


        nodes_scalar, nodes_vector, edge_offset = self._get_painn_embeddings(batch)

        new_atom_index_batch = batch['num_nodes'] + edge_offset.squeeze(-1).squeeze(-1)
        # print(f'new_atom_index_batch : {new_atom_index_batch}')
        new_atom_index_batch = new_atom_index_batch - 1
        # print(f'new_atom_index_batch : {new_atom_index_batch}')
        # new_index_batch = agent_num + edge_offset.squeeze(-1).squeeze(-1)
        new_atom_nodes_scalar = nodes_scalar[new_atom_index_batch, :]
        new_atom_nodes_vector = nodes_vector[new_atom_index_batch, :, :]

        

        new_atom_nodes_scalar = torch.stack(torch.split(nodes_scalar,batch['num_nodes'].tolist()))
        new_atom_nodes_vector = torch.stack(torch.split(nodes_vector,batch['num_nodes'].tolist()))

        return new_atom_nodes_scalar, new_atom_nodes_vector
    

    def _get_painn_embeddings(self, input_dict: dict) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        # Unpad and concatenate edges and features into batch (0th) dimension
        edges_displacement = layer.unpad_and_cat(
            input_dict["edges_displacement"], input_dict["num_edges"]
        )
        edge_offset = torch.cumsum(
            torch.cat(
                (
                    torch.tensor([0], device=input_dict["num_nodes"].device),
                    input_dict["num_nodes"][:-1],
                )
            ),
            dim=0,
        )
        edge_offset = edge_offset[:, None, None]
        edges = input_dict["edges"] + edge_offset
        edges = layer.unpad_and_cat(edges, input_dict["num_edges"])

        # Unpad and concatenate all nodes into batch (0th) dimension
        nodes_xyz = layer.unpad_and_cat(
            input_dict["nodes_xyz"], input_dict["num_nodes"]
        )

        nodes_scalar = layer.unpad_and_cat(input_dict["nodes"], input_dict["num_nodes"])
        nodes_scalar = self.atom_embeddings(nodes_scalar)
        nodes_scalar_orig = nodes_scalar

        nodes_vector = torch.zeros(
            (nodes_scalar.shape[0], 3, self.hidden_state_size),
            dtype=nodes_scalar.dtype,
            device=nodes_scalar.device,
        )


        timestep_embedding_state = None
        if self.use_time_embeddings != 0:
            timestep_embedding_state = get_timestep_embedding(input_dict["timestep"],self.hidden_state_size).repeat([input_dict["nodes"].shape[1],1])
            #get_timestep_embedding(input_dict["timestep"],self.hidden_state_size)
            #nodes_scalar = nodes_scalar + timestep_embedding_state
            nodes_scalar = nodes_scalar + timestep_embedding_state

        # Compute edge distances
        edges_distance, edges_diff, rot_diff = layer.calc_distance(
            nodes_xyz,
            input_dict["cell"],
            edges,
            edges_displacement,
            input_dict["num_edges"],
            return_diff=True,
        )

        # Expand edge features in Gaussian basis
        edge_state = layer.sinc_expansion(
            edges_distance, [(self.distance_embedding_size, self.cutoff)]
        )

        """rot_edge_state = None
        if self.use_rot_encoding:
            if "nodes_rot" in input_dict:
                nodes_rot = layer.unpad_and_cat(
                    input_dict["nodes_rot"], input_dict["num_nodes"]
                )
                #rot_mat = np.linalg.inv(R.from_euler("xyz", nodes_rot, degrees=True).as_matrix())
                rot_mat = torch.linalg.inv(nodes_rot)

                rot_mat = rot_mat[edges[:,0]]

                #rot_diff = torch.bmm(edges_diff.unsqueeze(-2),torch.tensor(rot_mat, dtype=torch.float32)).squeeze()
                rot_diff = (rot_mat@edges_diff.unsqueeze(-1)).squeeze()

                rot_diff = torch.nn.functional.normalize(rot_diff,dim=-1)
                
                #new_base_batch, new_dist_batch = convert(x_pos)
                rot_edge_state = torch.nan_to_num(rot_diff, nan=0.0)
                #rot_edge_state = layer.rot_expansion(
                #    rot_diff, [(self.distance_embedding_size, self.cutoff)]
                #)
        """

        # Apply interaction layers
        if self.use_time_embeddings != 0:
            
            for int_layer, update_layer in zip(
                self.interactions, self.scalar_vector_update
            ):
                nodes_scalar, nodes_vector = int_layer(
                    nodes_scalar,# + timestep_embedding_state,#self.reducer(torch.cat([nodes_scalar, timestep_embedding_state], dim = 1)),
                    nodes_vector,
                    edge_state,
                    edges_diff,
                    edges_distance,
                    edges,
                    timestep_embedding_state
                )
                nodes_scalar, nodes_vector = update_layer(
                    nodes_scalar,# + timestep_embedding_state,#self.reducer(torch.cat([nodes_scalar, timestep_embedding_state], dim = 1)),
                    nodes_vector
                )
        else:
            for int_layer, update_layer in zip(
                self.interactions, self.scalar_vector_update
            ):
                nodes_scalar, nodes_vector = int_layer(
                    nodes_scalar,
                    nodes_vector,
                    edge_state,
                    edges_diff,
                    edges_distance,
                    edges,
                    None
                )
                nodes_scalar, nodes_vector = update_layer(nodes_scalar, nodes_vector)

        return nodes_scalar, nodes_vector, edge_offset
