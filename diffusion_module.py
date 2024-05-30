# Copyright 2024 jimmieliu @ https://github.com/jimmieliu/OpenAlphaFold3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn

from openfold.model.primitives import Linear, LayerNorm, Attention, Linear
from openfold.utils.tensor_utils import permute_final_dims
from typing import Optional

from openfold.utils.tensor_utils import one_hot
from openfold.utils.rigid_utils import rot_vec_mul

ATOM_NUM_PER_RESIDUE = 14 # 37
TRAIN_AUGMENTATION_SAMPLES = 24

def UniformRandomRotation3D():
    """Generate a 3D random rotation matrix.
    https://github.com/andreyzhitnikov/randrot

    Returns:
        torch.tensor: A 3D rotation matrix.
    """
    x1, x2, x3 = torch.rand(3)
    R = torch.as_tensor([[torch.cos(2 * torch.pi * x1), torch.sin(2 * torch.pi * x1), 0],
                   [-torch.sin(2 * torch.pi * x1), torch.cos(2 * torch.pi * x1), 0],
                   [0, 0, 1]])
    v = torch.as_tensor([[torch.cos(2 * torch.pi * x2) * torch.sqrt(x3)],
                   [torch.sin(2 * torch.pi * x2) * torch.sqrt(x3)],
                   [torch.sqrt(1 - x3)]])
    H = torch.eye(3) - 2 * v * v.T
    M = -H * R
    return M

def batch_uniform_random_rotation_3d(
        shape, # (b, 48, 1, 3, 3)
                                     ):
    x1, x2, x3 = torch.rand(shape[:-2]), torch.rand(shape[:-2]), torch.rand(shape[:-2]) # (b, 48, 1)
    R = torch.zeros(shape)
    R[...,0,0] = torch.cos(2 * torch.pi * x1)
    R[...,0,1] = torch.sin(2 * torch.pi * x1)
    R[...,1,0] = -torch.sin(2 * torch.pi * x1)
    R[...,1,1] = torch.cos(2 * torch.pi * x1)
    R[...,2,2] = 1

    v = torch.zeros((*shape[:-1],1)) # (b, 48, 1, 3, 1)
    v[...,0,0] = torch.cos(2 * torch.pi * x2) * torch.sqrt(x3)
    v[...,1,0] = torch.sin(2 * torch.pi * x2) * torch.sqrt(x3)
    v[...,2,0] = torch.sqrt(1 - x3)

    if len(shape) == 5:
        H = torch.eye(3)[None,None,None,:,:] - 2 * v * permute_final_dims(v, (1, 0))
    elif len(shape) == 4:
        H = torch.eye(3)[None,None,:,:] - 2 * v * permute_final_dims(v, (1, 0))
    else:
        assert False

    M = -H * R
    return M


def centre_random_augmentation(xl, s_trans=1):
    xl = xl - xl.mean(dim=-2, keepdim=True) # by l # (b, a, 3) (b, 48, a, 3)
    R = batch_uniform_random_rotation_3d(
        (*xl.shape[:-2], 1, 3, 3) # (b, 48, 1, 3, 3)
        ).to(xl.device) # 3x3 Rotation matrix
    t = s_trans * torch.rand_like(xl)
    xl = rot_vec_mul(R, xl) + t
    return xl


class FourierEmbedding(nn.Module):
    def __init__(self, c) -> None:
        super().__init__()
        # Randomly generate weight/bias once before training
        def init_fn(w, b):
            torch.nn.init.normal_(w, mean=0.0, std=1.0)
            torch.nn.init.normal_(b, mean=0.0, std=1.0)
            
        self.w = torch.nn.parameter.Parameter(torch.empty((c,)))
        self.b = torch.nn.parameter.Parameter(torch.empty((c,)))
        init_fn(self.w, self.b)

    def forward(self, t_hat,):
        # Compute embeddings
        return torch.cos(2*torch.pi*(t_hat*self.w + self.b)) # (c,)


class RelativePositionEncoding(nn.Module):
    def __init__(self, r_max=32, s_max=2, cz=128):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.clip_by_rmax = 2*self.r_max
        self.clip_by_zero = 0.0
        self.rmax_else = torch.tensor(2*self.r_max+1)
        rmax_vbins = torch.arange(
            start=0, end=self.rmax_else + 1
        )
        self.smax_else = torch.tensor(2*self.s_max+1)
        smax_vbins = torch.arange(
            start=0, end=self.smax_else + 1
        )
        self.linearnobias = Linear(2*(self.rmax_else + 1) + 1 + (self.smax_else + 1), 
                                   cz, 
                                   bias=False, 
                                #    init="default"
                                   )
    
        self.register_buffer("rmax_vbins", rmax_vbins)
        self.register_buffer("smax_vbins", smax_vbins)

    def forward(self, feats):
        _b,_r = feats["aatype"].shape
        bij_same_chain = (
            feats["asym_id"].unsqueeze(-1) == feats["asym_id"].unsqueeze(-2)
        ) # (b, r, r)
        bij_same_residue = (
            feats["residue_index"].unsqueeze(-1) == feats["residue_index"].unsqueeze(-2)
        )
        bij_same_entity = (
            feats["entity_id"].unsqueeze(-1) == feats["entity_id"].unsqueeze(-2)
        )
        # ----
        dij_residue = (
            feats["residue_index"].unsqueeze(-1) == feats["entity_id"].unsqueeze(-2)
        )
        dij_residue = torch.clip(dij_residue + self.r_max, 
                                 self.clip_by_zero, 
                                 self.clip_by_rmax)
        dij_residue.masked_fill_(~bij_same_chain, self.rmax_else)
        #
        aij_rel_pos = one_hot(dij_residue, self.rmax_vbins)
        
        # -----
        bij_same_chain_and_same_residue = bij_same_chain & bij_same_residue
        dij_token = (
            feats["token_index"].unsqueeze(-1) == feats["token_index"].unsqueeze(-2)
        )
        dij_token = torch.clip(dij_token + self.r_max, 
                                 self.clip_by_zero, 
                                 self.clip_by_rmax).repeat(_b, 1, 1)
        dij_token.masked_fill_(~bij_same_chain_and_same_residue, self.rmax_else)
        # 
        aij_rel_token = one_hot(dij_token, self.rmax_vbins)
        # -----
        dij_chain = (
            feats["sym_id"].unsqueeze(-1) == feats["sym_id"].unsqueeze(-2)
        )
        dij_chain = torch.clip(dij_chain, 
                                 self.clip_by_zero, 
                                 self.clip_by_rmax)
        dij_chain.masked_fill_(~bij_same_chain, self.smax_else)
        aij_rel_chain = one_hot(dij_chain, self.smax_vbins)

        pij = self.linearnobias(torch.concat([
            aij_rel_pos, aij_rel_token, bij_same_entity.unsqueeze(-1), aij_rel_chain
        ], dim=-1))
        return pij # (b, r, r, cz)

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_

class Transition(nn.Module):

    def __init__(self, c, n):
        super(Transition, self).__init__()

        self.c = c
        self.n = n

        self.layer_norm = LayerNorm(self.c)
        self.linear_1 = Linear(self.c, self.n * self.c, init="relu")
        self.linear_b = Linear(self.c, self.n * self.c, init="relu")
        init_method_normal(0.02)(self.linear_b.weight)
        init_method_normal(0.02)(self.linear_1.weight)
        self.swish = nn.SiLU()
        self.linear_2 = Linear(self.n * self.c, self.c, init="final")

    def _transition(self, m, mask=None):
        a = self.linear_1(m)
        b = self.linear_b(m)
        m = self.linear_2(self.swish(a) * b)
        if mask is not None:
            m = m * mask.unsqueeze(-1)
        return m
    
    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        m = self.layer_norm(m)

        m = self._transition(m, mask)

        return m


class DiffusionConditioning(nn.Module):
    def __init__(self, config, cz=128, cs=384) -> None:
        super().__init__()
        cz = config["diffcond"]["cz"]
        cs = config["diffcond"]["cs"]
        r_max = config["diffcond"]["r_max"]
        s_max = config["diffcond"]["s_max"]
        self.rela_pos_enc = RelativePositionEncoding(r_max=r_max, s_max=s_max,cz=cz)
        # pair
        in_cz = config["trunk"]["c_z"]+cz
        self.z_linearnobias = Linear(in_dim=in_cz, out_dim=cz, bias=False)
        self.z_ln = LayerNorm(in_cz)
        self.z_transition1 = Transition(cz, n=2)
        self.z_transition2 = Transition(cz, n=2)
        
        # single
        in_cs = config["trunk"]["c_m"]+config["trunk"]["c_s"]
        self.s_linearnobias = Linear(in_dim=in_cs, out_dim=cs, bias=False)
        self.s_ln = LayerNorm(in_cs)
        cf = config["diffcond"]["c_fourier"]
        self.fourier_embedding = FourierEmbedding(cf)
        self.f_ln = LayerNorm(cf)
        self.f_linearnobias = Linear(in_dim=cf, out_dim=cs, bias=False)
        self.s_transition1 = Transition(cs, n=2)
        self.s_transition2 = Transition(cs, n=2)

    def forward(self, t_hat, feats, si_inputs, si_trunk, zij_trunk, sigma_data):
        # Pair conditioning
        zij = torch.concat([zij_trunk, self.rela_pos_enc(feats)], dim=-1)
        zij = self.z_linearnobias(self.z_ln(zij))
        zij = zij + self.z_transition1(zij)
        zij = zij + self.z_transition2(zij)

        # Single conditioning
        si = torch.concat([si_trunk, si_inputs],dim=-1)
        si = self.s_linearnobias(self.s_ln(si))
        n = self.fourier_embedding(torch.log(t_hat/sigma_data)/4) # , 256
        si = si + self.f_linearnobias(self.f_ln(n))
        si = si + self.s_transition1(si)
        si = si + self.s_transition2(si)
        return si, zij


class CustomAttention(Attention):
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        super().__init__(c_q, c_k, c_v, c_hidden, no_heads, gating)
        # No Bias compare to parant
        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, init="glorot"
        )
        # No Bias compare to parant
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q, bias=False, init="final")
        if self.gating:
            # No Bias compare to parant
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, bias=False, init="gating"
            )


class AdaLN(nn.Module):
    def __init__(self, dim1, dim2) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim1, elementwise_affine=False, bias=False)
        self.layer_norm_2 = nn.LayerNorm(dim2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(dim2, dim1)
        self.linear_no_bias = nn.Linear(dim2, dim1, bias=False)

    def forward(self,
        ai,  # (b, r, c)
        si,  # (b, r, c)
        ):
        ai = self.layer_norm_1(ai)
        si = self.layer_norm_2(si)
        ai = self.sigmoid(self.linear(si)) * ai + self.linear_no_bias(si)
        return ai


class ConditionedTransitionBlock(nn.Module):

    def __init__(self, config, multiplier=2, c_token=None, c_s=None):
        super().__init__()
        self.config = config
        self.hidden_dim = config["atom_encoder"]["c_token"] if c_token is None else c_token
        self.s_cond_dim = config["diffcond"]["cs"] if c_s is None else c_s
        self.ada_layernorm = AdaLN(self.hidden_dim, self.s_cond_dim)
        self.linear_1 = nn.Linear(self.hidden_dim, self.hidden_dim * 2, bias=False)
        self.linear_2 = nn.Linear(self.hidden_dim, self.hidden_dim * 2, bias=False)
        self.linear_3 = nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)
        self.swish = nn.SiLU()
        gating_s = nn.Linear(self.s_cond_dim, self.hidden_dim)
        with torch.no_grad():
            gating_s.bias.fill_(-2.0)
        self.gating_s = nn.Sequential(gating_s, nn.Sigmoid())

    def forward(self, a_i, s_i):
        a_i = self.ada_layernorm(a_i, s_i)
        b = self.swish(self.linear_1(a_i)) * self.linear_2(a_i)
        a_i = self.gating_s(s_i) * a_i * self.linear_3(b)
        return a_i


class AttentionPairBias(nn.Module):
    def __init__(self, config, n_head=None, c_token=None, c_s=None, c_pair=None) -> None:
        super().__init__()
        self.config = config
        # n_head, hidden will be provided with value if DiffusionTransformer is submodule of atom encoder or atom decoder
        self.hidden_dim = config["atom_encoder"]["c_token"] if c_token is None else c_token
        self.nhead = config["transformer"]['n_head'] if n_head is None else n_head
        self.s_cond_dim = config["diffcond"]["cs"] if c_s is None else c_s
        self.z_cond_dim = config["diffcond"]["cz"] if c_pair is None else c_pair
        assert self.hidden_dim % self.nhead == 0
        self.head_dim = self.hidden_dim // self.nhead
        self.mha = CustomAttention(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.head_dim, self.nhead)

        self.ada_layernorm = AdaLN(self.hidden_dim, self.s_cond_dim)
        self.layernorm_s = nn.LayerNorm(self.s_cond_dim)
        self.layernorm_z = nn.LayerNorm(self.z_cond_dim)
        self.z_injection = nn.Linear(self.z_cond_dim, self.nhead, bias=False)
        gating_s = nn.Linear(self.s_cond_dim, self.hidden_dim)
        with torch.no_grad():
            gating_s.bias.fill_(-2.0)
        self.gating_s = nn.Sequential(gating_s, nn.Sigmoid())

    def _prep_inputs(self, a_i, s_i, z_ij, beta_ij=None):
        if s_i is not None:
            a_i = self.ada_layernorm(a_i, s_i)
        else:
            a_i = self.layernorm_s(a_i)
        z_ij = self.layernorm_z(z_ij)
        z_ij = self.z_injection(z_ij)  # (b, r, r, cz) -> (b, h, r, r)
        z_ij = permute_final_dims(z_ij, (2, 0, 1))
        if beta_ij is not None:
            beta_ij = beta_ij.unsqueeze(1)  # (b, r, r) -> (b, 1, r, r)
        return a_i, z_ij, beta_ij

    def forward(self, a_i, s_i, z_ij, beta_ij=None):
        a_i, z_ij, beta_ij = self._prep_inputs(a_i, s_i, z_ij, beta_ij)
        bias = [z_ij]
        if beta_ij is not None:
            bias.append(beta_ij)
        a_i = self.mha(a_i, a_i, bias)
        if s_i is not None:
            a_i = self.gating_s(s_i) * a_i
        return a_i
        

class DiffusionTransformer(nn.Module):

    def __init__(self, config, n_block=None, n_head=None, c_token=None, c_s=None, c_pair=None):
        super().__init__()
        self.config = config
        self.nblock = config["transformer"]['n_block'] if n_block is None else n_block
        self.attention_layer = nn.ModuleList([])
        self.transition_layer = nn.ModuleList([])
        for _ in range(self.nblock):
            self.attention_layer.append(AttentionPairBias(config, n_head=n_head, c_token=c_token, c_s=c_s, c_pair=c_pair))
            self.transition_layer.append(ConditionedTransitionBlock(config, c_token=c_token, c_s=c_s))

    def forward(self, a_i, s_i, z_ij, beta_ij=None):
        for attention, transition in zip(self.attention_layer, self.transition_layer):
            b = attention(a_i, s_i, z_ij, beta_ij=beta_ij)
            a_i = b + transition(a_i, s_i)
        return a_i


class InputFeatureEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.atom_attention_encoder = AtomAttentionEncoder()
    
    def forward(self, feats, ):
        a, _, _, _ = self.atom_attention_encoder(feats, # ????
                                                None, 
                                                None, 
                                                None, c_atom=128, c_atompair=16, c_token=384)
        s = torch.concat([a, # (b, r, ?)
                           feats["restype"],  # (b, r, ?) 
                           feats["profile"],  # (b, r, ?)
                           feats["deletion_mean"], # (b, r, 1)
                           ])
        return s

class AtomAttentionDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        c_atom = config["c_atom"]
        c_atompair = config["c_atompair"]
        c_token = config["c_token"]
        self.atom_transformer = AtomTransformer(config, config["atom_decoder"]["atom_tfm"],
                                                c_token=c_atom,
                                                c_s=c_atom,
                                                c_pair=c_atompair,)
        self.q_linearnobias = Linear(in_dim=c_token, 
                                     out_dim=c_atom,
                                     bias=False)
        self.r_linearnobias = Linear(in_dim=c_atom,
                                    out_dim=3,
                                    bias=False)
        self.r_layernorm = LayerNorm(c_atom)
    
    def forward(self, a, q_skip, c_skip, p_skip, feats):
        # Broadcast per-token activiations to per-atom activations and add the skip connection
        _b, _r = feats["aatype"].shape
        
        batch_dims = (_b, TRAIN_AUGMENTATION_SAMPLES) if self.training else (_b,)
        q = q_skip.view((*batch_dims, _r, ATOM_NUM_PER_RESIDUE, -1)) \
            + self.q_linearnobias(a.view((*batch_dims, _r, 1, -1)))
        q = q.reshape(*batch_dims, _r * ATOM_NUM_PER_RESIDUE, -1)
        # Cross attention transformer.
        q = self.atom_transformer(q, c_skip, p_skip, feats)
        # Map to positions update.
        r_update = self.r_linearnobias(self.r_layernorm(q)) # (b,a,c)
        return r_update

class AtomAttentionEncoder(nn.Module):
    def __init__(self, config, withrl=True) -> None:
        super().__init__()
        c_m = config["trunk"]["c_m"]
        c_z = config["trunk"]["c_z"]
        c_token = config["c_token"]
        c_atom = config["c_atom"]
        c_atompair = config["c_atompair"]
        self.atom_transformer = AtomTransformer(config, 
                                                config["atom_encoder"]["atom_tfm"],
                                                c_token=c_atom,
                                                c_s=c_atom,
                                                c_pair=c_atompair,
                                                )
        self.linearnobias1 = Linear(in_dim=3+1+1+128+4*64, 
                                    out_dim=c_atom, 
                                    bias=False)
        
        self.p_linearnobias1 = Linear(in_dim=3, 
                                    out_dim=c_atompair, 
                                    bias=False)
        self.p_linearnobias2 = Linear(in_dim=1, 
                                    out_dim=c_atompair, 
                                    bias=False)
        self.p_linearnobias3 = Linear(in_dim=1, 
                                    out_dim=c_atompair, 
                                    bias=False)

        if withrl:
            self.rl_c_linearnobias = Linear(in_dim=c_m, 
                                    out_dim=c_atom, 
                                    bias=False)
            self.rl_c_ln = LayerNorm(c_m)
            self.rl_p_linearnobias = Linear(in_dim=c_z, 
                                    out_dim=c_atompair, 
                                    bias=False)
            self.rl_p_ln = LayerNorm(c_z)

            self.rl_q_linearnobias = Linear(in_dim=3, 
                                    out_dim=c_atom, 
                                    bias=False)
        self.relu = torch.nn.ReLU()

        self.p_cl_linearnobias = Linear(in_dim=c_atom, 
                                    out_dim=c_atompair, 
                                    bias=False, init="relu")
        self.p_cm_linearnobias = Linear(in_dim=c_atom, 
                                    out_dim=c_atompair, 
                                    bias=False, init="relu")
        self.p_mlp_linearnobias1 = Linear(in_dim=c_atompair, 
                                    out_dim=c_atompair, 
                                    bias=False, init="relu")
        self.p_mlp_linearnobias2 = Linear(in_dim=c_atompair, 
                                    out_dim=c_atompair, 
                                    bias=False, init="relu")
        self.p_mlp_linearnobias3 = Linear(in_dim=c_atompair, 
                                    out_dim=c_atompair, 
                                    bias=False)
        self.q_linearnobias = Linear(in_dim=c_atom, 
                                    out_dim=c_token, 
                                    bias=False, init="relu")

    def forward(self, 
                feats, # ????
                rl,  # (b, a, 3)
                si_trunk,  # (b, r, c)
                zij,  # (b, r, r, c)
                
                ): # 768
        # Create the atom single conditioning: Embed per-aton meta data
        c = self.linearnobias1(torch.concat([feats["ref_pos"], # (b, a, 3)
                                        feats["ref_charge"], # (b, a, 1)
                                        feats["ref_mask"], # (b, a, 1)
                                        feats["ref_element"],  # (b, a, 128)
                                        feats["ref_atom_name_chars"],  # (b, a, 4* 64)
                                        ], 
                                        dim=-1)) # (b, a, c_atom)
        
        # Embed offsets between atom reference positions
        # (b, a, 1, 3) - (b, 1, a, 3) -> (b, a, a, 3)
        d = feats["ref_pos"].unsqueeze(-2) - feats["ref_pos"].unsqueeze(-3)
        
        # (b,a,1) == (b,1,a) -> (b,a,a,1)
        v = (feats["ref_space_uid"].unsqueeze(-1) == feats["ref_space_uid"].unsqueeze(-2)).unsqueeze(-1)

        p = self.p_linearnobias1(d) * v # (b, a, a, c_atompair)
        # Embed pairwise inverse squared distances, and the valid mask.
        p = p + self.p_linearnobias2(1 / (1 + torch.norm(d, dim=-1, keepdim=True))) * v
        p = p + self.p_linearnobias3(v.to(p.dtype)) * v

        # Initialise the atom single representation as the single conditioning
        q = c # (b, a, c_atom)
        # If provided, add trunk embeddings and noisy positions
        _b,_r = feats["aatype"].shape
        _a = _r*ATOM_NUM_PER_RESIDUE
        if self.training:
            batch_dims = (_b, TRAIN_AUGMENTATION_SAMPLES)
            single_dims = (_b, )
        else:
            batch_dims = (_b, )
            single_dims = (_b, )
        
        if rl is not None:
            c_reshaped = c.view((*single_dims, _r, ATOM_NUM_PER_RESIDUE, -1))
            c_reshaped = c_reshaped + self.rl_c_linearnobias(self.rl_c_ln(
                si_trunk
            )).unsqueeze(-2)
            c = c_reshaped.reshape((*single_dims, _a, -1))
            
            p_reshaped = p.view(*single_dims, _r, ATOM_NUM_PER_RESIDUE, _r, ATOM_NUM_PER_RESIDUE, -1)
            p_reshaped = p_reshaped + self.rl_p_linearnobias(self.rl_p_ln(
                zij # (b,r,r,c) -> (b,a,a,c)
            ))[:,:,None,:,None,:]
            p = p_reshaped.reshape((*single_dims, _a, _a, -1))

            q = (q.unsqueeze(1) if self.training else q) + self.rl_q_linearnobias(rl)
            c = c.unsqueeze(1) if self.training else c
            p = p.unsqueeze(1) if self.training else p

        # Add the combined single conditioning to the pair representation
        p = p + self.p_cl_linearnobias(self.relu(c.unsqueeze(-2))) \
            + self.p_cm_linearnobias(self.relu(c.unsqueeze(-3)))
        # Run a small MLP on the pair activations 
        p = p + self.p_mlp_linearnobias3(self.relu(self.p_mlp_linearnobias2(self.relu(self.p_mlp_linearnobias1(self.relu(p))))))
        # Cross attention transformer
        q = self.atom_transformer(q, c, p, feats) # (b, a, c_atom)

        # Aggregate per-atom representation to per-token representation
        a = (
            self.relu(self.q_linearnobias(q)) # (b, a, c_atom)
        ).view((*batch_dims, _r, ATOM_NUM_PER_RESIDUE, -1)).mean(-2) # (b, r, al, c) -> (b, r, c)
        q_skip, c_skip, p_skip = q, c, p
        return a, q_skip, c_skip, p_skip


class AtomTransformer(nn.Module):
    def __init__(self, global_config, config, c_token=None, c_s=None, c_pair=None): # , n_block, n_head, n_queries=32, n_keys=128, s_subset_centres={15.5, 47.5, 79.5, ...}) -> None:
        super().__init__()
        max_seq_len = config["max_seq_len"]
        l = torch.arange(max_seq_len)
        m = l.unsqueeze(-2) # (1, a)
        l.unsqueeze_(-1) # (a, 1)
        con = torch.zeros((max_seq_len, max_seq_len)).to(torch.bool)

        stop = max_seq_len # + config["atom_encoder"]["atom_tfm"]["n_queries"] + 1
        half_nquery = round(config["n_queries"] / 2, 1)
        half_nkey =  round(config["n_keys"] / 2, 1)
        for _c in torch.arange(half_nquery, stop, config["n_queries"]):
            # config["atom_encoder"]["atom_tfm"]["s_subset_centres"]: # this can be done with multiple head or can be cached and reused??
            con1 = (l - _c).abs() < half_nquery  # (a, 1)
            con2 = (m - _c).abs() < half_nkey  # (1, a)
            # print(type(con1), type(con2), con1.dtype, con2.dtype)
            _con = con1 & con2
            # print(type(_con), _con.dtype)
            con = con | _con

        beta = torch.ones((max_seq_len, max_seq_len), dtype=torch.float32) * -1e10
        beta.masked_fill_(con, 0)

        self.register_buffer("beta", beta)

        self.diffussion_transformer = DiffusionTransformer(global_config, 
                    n_block=config["n_block"], 
                    n_head=config["n_head"],
                    c_token=c_token,
                    c_s=c_s,
                    c_pair=c_pair,
                    ) # TODO local attention or sliding window 

    def forward(self, 
                q, # (b, a, c)
                c, # (b, a, c)
                p, # (b, a, a, c)
                feats
                ):
        # sequence-local atom attention is equivalent to self attention within rectangular blocks along the diagonal
        _b, _r = feats["aatype"].shape
        assert self.beta.shape[-1] >= _r * ATOM_NUM_PER_RESIDUE
        q = self.diffussion_transformer(q, c, p, 
                                        self.beta[None,:_r*ATOM_NUM_PER_RESIDUE,:_r*ATOM_NUM_PER_RESIDUE] # (1, a, a)
                                        )  # (b, a, c)
        return q


class DiffusionModule(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.diffusion_conditioning = DiffusionConditioning(config)
        self.atom_attention_encoder = AtomAttentionEncoder(config)
        self.diffusion_transformer = DiffusionTransformer(config)
        self.atom_attention_decoder = AtomAttentionDecoder(config)
        self.sigma_data = config["sigma_data"]
        # sigma_data=16, c_atom=128, c_atompair=16, c_token=768
        self.s_layernorm = LayerNorm(config["diffcond"]["cs"])
        self.a_linearnobias = Linear(in_dim=config["diffcond"]["cs"], out_dim=config["c_token"], bias=False)
        self.a_layernorm = LayerNorm(config["c_token"])

    def forward(self, 
                xl_noisy,  # (b, a, 3)
                t_hat, 
                feats,
                si_inputs, # (b, r, c)
                si_trunk,  # (b, r, c)
                zij_trunk,  # (b, r, r, c)
                ):
        # Conditioning
        si, zij = self.diffusion_conditioning(t_hat, feats, si_inputs, si_trunk, zij_trunk, 
                                              self.sigma_data)
        
        # Scale positions to dimensionless vectors with approximately unit variance
        rl_noisy = xl_noisy / (t_hat**2 + self.sigma_data**2)**0.5 # (b, a, 3)

        # Sequnce-local Atom Attention and aggregation to coarse-grained tokens
        (
            ai, # (b, r, c)
            ql_skip,  # (b, a, c)
            cl_skip,  # (b, a, c)
            plm_skip # (b, a, a, c)
        ) = self.atom_attention_encoder(feats, # ????
                                        rl_noisy,  # (b, a, 3)
                                        si_trunk,  # (b, r, c)
                                        zij,  # (b, r, r, c)
                                        )
        
        # Full self-attention on token level
        if self.training:
            si = si.unsqueeze(1)
            zij = zij.unsqueeze(1)

        ai = ai + self.a_linearnobias(self.s_layernorm(si))
        ai = self.diffusion_transformer(
            ai, si, zij,
        )
        ai = self.a_layernorm(ai)
        # Broadcast token activations to atons and run Sequence-local Atom Attention
        rl_update = self.atom_attention_decoder(ai, ql_skip, cl_skip, plm_skip, feats) # (b, a, c)

        # Rescale updates to positions and combine with input positions
        xl_out = self.sigma_data**2 / (self.sigma_data**2 + t_hat**2) * xl_noisy + self.sigma_data * t_hat / (self.sigma_data**2 + t_hat**2)**0.5 * rl_update
        return xl_out

        
def rand_norm(size, mean=0, std=1):
    return torch.normal(mean=mean, std=std, size=size)

class SampleDiffusion(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.diffusion_module = DiffusionModule(config)
        self.sigma_data = config["sigma_data"]
    
    def get_noise_schedule(self, num_steps=200):
        s_max=160
        s_min=4e-4
        rho=7
        # Time step discretization
        num_steps = 200
        step_indices = torch.arange(num_steps, dtype=torch.float64) / (num_steps-1)
        t_steps = self.sigma_data * (
            s_max**(1/rho) 
            + step_indices * (s_min**(1/rho)-s_max**(1/rho))
            )**rho
        return t_steps

    def forward(self, feats, # dict 
                si_inputs,  # (b, r, c)
                si_trunk,   # (b, r, c)
                zij_trunk,  # (b, r, r, c)
                gamma0=0.8, gammamin=1.0, noise_scale_lambda=1.003, step_scale_eta=1.5):
        
        # Training
        if self.training:
            if ATOM_NUM_PER_RESIDUE == 14:
                y = feats["atom14_gt_positions"] # b, r, 14, 3
            elif ATOM_NUM_PER_RESIDUE == 37:
                y = feats["atom37_gt_positions"] # b, r, 37, 3
            else:
                assert False, "ATOM_NUM_PER_RESIDUE can only be 14 or 37"
            y = y.unsqueeze(1) # b, 1, r, 14, 3
            index_repeats = [1] * y.dim()
            index_repeats[1] = TRAIN_AUGMENTATION_SAMPLES
            y = y.repeat(*index_repeats) # b, 48, r, 14, 3
            yshape = y.shape
            y = y.view((*yshape[:2], yshape[2] * yshape[3], 3)) # b, 48, r * 14, 3
            y = centre_random_augmentation(y)
            sigma = self.sigma_data * torch.exp(-1.2 + 1.5 * torch.normal(mean=0, std=1, size=(1,)))
            sigma = sigma.to(y.device)
            weight = (sigma**2+self.sigma_data**2) / (sigma*self.sigma_data)**2
            n = torch.rand_like(y) * sigma
            D_yn = self.diffusion_module(y+n, sigma, feats, si_inputs, si_trunk, zij_trunk)

            return D_yn, weight

        # inference
        # Main sampling loop.
        if ATOM_NUM_PER_RESIDUE == 14:
           x_shape = (*feats["atom14_atom_exists"].shape, 3, ) # (b, r, 14, 3)
        elif ATOM_NUM_PER_RESIDUE == 37:
            x_shape = (*feats["atom37_atom_exists"].shape, 3, ) # (b, r, 37, 3)
        else:
            assert False, "ATOM_NUM_PER_RESIDUE can only be 14 or 37"
        
        device = feats["input_sequence_tokens"].device
        noise_schedule = self.get_noise_schedule().to(device)
        x_shape = (x_shape[0], x_shape[1]*x_shape[2], x_shape[-1])
        xl = noise_schedule[0] * rand_norm(x_shape, 0, 1).to(device) # (b, a, 3)
        for i, ct in enumerate(noise_schedule):
            if i == 0: 
                continue

            # Increase noise temporarily.
            xl = centre_random_augmentation(xl) # (b, a, 3)
            gamma = gamma0 if ct > gammamin else 0
            ct_1 = noise_schedule[i-1]
            t_hat = ct_1 * (gamma + 1)
            xi_l = noise_scale_lambda * (t_hat**2 - ct_1**2) ** 0.5 * rand_norm(x_shape, 0, 1).to(device) # (b, a, 3)
            xl_noisy = xl + xi_l # (b, a, 3)

            # Euler step
            xl_denoised = self.diffusion_module(xl_noisy, t_hat, feats, si_inputs, si_trunk, zij_trunk) # (b, a, 3)
            delta_l = (xl - xl_denoised) / t_hat
            dt = ct - t_hat
            xl = xl_noisy + step_scale_eta * dt * delta_l
        
        return xl, None

