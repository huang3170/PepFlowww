import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
from tqdm.auto import tqdm
import functools
from torch.utils.data import DataLoader
import os
import argparse

import pandas as pd

from models_con.edge import EdgeEmbedder
from models_con.node import NodeEmbedder
from pepflow.modules.common.layers import sample_from, clampped_one_hot
from models_con.ga import GAEncoder
from pepflow.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms
from pepflow.modules.common.geometry import construct_3d_basis
from pepflow.utils.data import mask_select_data, find_longest_true_segment, PaddingCollate
from pepflow.utils.misc import seed_all
from pepflow.utils.train import sum_weighted_losses
from torch.nn.utils import clip_grad_norm_

from pepflow.modules.so3.dist import centered_gaussian,uniform_so3
from pepflow.modules.common.geometry import batch_align, align

from tqdm import tqdm

import wandb

from data import so3_utils
from data import all_atom

from models_con.pep_dataloader import PepDataset

from pepflow.utils.misc import load_config
from pepflow.utils.train import recursive_to
from easydict import EasyDict

from models_con.utils import process_dic
from models_con.torsion import get_torsion_angle, torsions_mask
import models_con.torus as torus

import gc

from copy import deepcopy
from pepflow.utils.data import PaddingCollate
collate_fn = PaddingCollate(eight=False)
from pepflow.utils.train import recursive_to

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}

class FlowModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._model_cfg = cfg.encoder
        self._interpolant_cfg = cfg.interpolant

        self.node_embedder = NodeEmbedder(feat_dim=cfg.encoder.node_embed_size,
                                          max_num_atoms=max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(feat_dim=cfg.encoder.edge_embed_size,
                                          max_num_atoms=max_num_heavyatoms)
        self.ga_encoder = GAEncoder(cfg.encoder.ipa)
        self.weight_network = nn.Sequential(nn.Linear(1, 256),  # Input shape: (batch_size, 1)
            nn.Mish(),  # Mish activation
            nn.Linear(256, 1) )

        self.sample_structure = self._interpolant_cfg.sample_structure
        self.sample_sequence = self._interpolant_cfg.sample_sequence

        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
    
    def encode(self, batch):
        rotmats_1 =  construct_3d_basis(center=batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
                                        p1=batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
                                        p2=batch['pos_heavyatom'][:, :, BBHeavyAtom.N] )
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        seqs_1 = batch['aa']

        angles_1 = batch['torsion_angle']

        #only take the CA atom positon for the receptor as true. mark which residue is from peptide
        context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
        structure_mask = context_mask if self.sample_structure else None
        sequence_mask = context_mask if self.sample_sequence else None
        node_embed = self.node_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        edge_embed = self.edge_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        
        return rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed
    
    def zero_center_part(self,pos,gen_mask,res_mask):
        """
        move pos by center of gen_mask
        pos: (B,N,3)
        gen_mask, res_mask: (B,N)
        """
        center = torch.sum(pos * gen_mask[...,None], dim=1) / (torch.sum(gen_mask,dim=-1,keepdim=True) + 1e-8) # (B,N,3)*(B,N,1)->(B,3)/(B,1)->(B,3)
        center = center.unsqueeze(1) # (B,1,3)
        # center = 0. it seems not center didnt influence the result, but its good for training stabilty
        pos = pos - center
        pos = pos * res_mask[...,None]
        return pos,center
    
    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k # (B,L,K)
    
    def forward(self, batch):
        
        sigma_data = 0.5
        P_mean = -1 
        P_std = 1.4
        c = 0.1

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask,angle_mask = batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()

        # encode
        
        # rotmats_1 is the rotation matrix for the x,y,z unit axis [B,L,3,3] 不同residue的x,y,z方向
        # trans_1 only the CA atom positions
        # seqs_1 is the aa sequence
        # angle_1 is the torsion_angle
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch) # no generate mask

        # prepare for denoise
        trans_1_c = trans_1 # already centered when constructing dataset
        seqs_1_simplex = self.seq_to_simplex(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)

        with torch.no_grad():
            # generate t 
            sigma = torch.randn((num_batch,1), device=batch['aa'].device)
            sigma = (sigma * P_std + P_mean).exp()
            t = torch.arctan(sigma / sigma_data)

            # t = torch.rand((num_batch,1), device=batch['aa'].device) 
            # t = t*(1-2 * self._interpolant_cfg.min_t) + self._interpolant_cfg.min_t # avoid 0
            if self.sample_structure:
                # corrupt trans
                # initialize random position
                trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device) * sigma_data # scale with sigma?
                trans_0_c,_ = self.zero_center_part(trans_0,gen_mask,res_mask)
                trans_t = torch.cos(t) * trans_1 + torch.sin(t) * trans_0_c
                trans_t_c = torch.where(batch['generate_mask'][...,None],trans_t,trans_1_c)
                dtrans_t_c_dt = torch.cos(t)* trans_0_c - torch.sin(t) * trans_1_c
                dtrans_t_c_dt = torch.where(batch['generate_mask'][...,None],dtrans_t_c_dt,trans_1_c)

                # corrupt rotmats
                rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device) * sigma_data
                rotmats_t = torch.cos(t) * rotmats_1 + torch.sin(t) * rotmats_0
                rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_t,rotmats_1)
                drotmats_t_dt = torch.cos(t) * rotmats_0 - torch.sin(t) * rotmats_1
                drotmats_t_dt = torch.where(batch['generate_mask'][...,None,None],drotmats_t_dt,rotmats_1)

                # corrup angles
                angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype) * sigma_data # (B,L,5)
                angles_t = torch.cos(t) * angles_1 + torch.sin(t) * angles_0
                angles_t = torch.where(batch['generate_mask'][...,None],angles_t,angles_1)
                dangles_t_dt = torch.cos(t) * angles_0 - torch.sin(t) * angles_1
                dangles_t_dt = torch.where(batch['generate_mask'][...,None],dangles_t_dt,angles_1)
            else:
                trans_t_c = trans_1_c.detach().clone()
                rotmats_t = rotmats_1.detach().clone()
                angles_t = angles_1.detach().clone()
            if self.sample_sequence:
                # corrupt seqs
                seqs_0_simplex = self.k * torch.randn_like(seqs_1_simplex) # (B,L,K)
                seqs_t_simplex = torch.cos(t) * seqs_1_simplex + torch.sin(t) * seqs_0_simplex # (B,L,K)
                seqs_t_simplex = torch.where(batch['generate_mask'][...,None],seqs_t_simplex,seqs_1_simplex)
                dseqs_t_simplex_dt = torch.cos(t) * seqs_0_simplex - torch.sin(t) * seqs_1_simplex # (B,L,K)
                dseqs_t_simplex_dt = torch.where(batch['generate_mask'][...,None],dseqs_t_simplex_dt,seqs_1_simplex)

                seqs_t_prob = F.softmax(seqs_t_simplex,dim=-1) # (B,L,K)
                seqs_t = sample_from(seqs_t_prob) # (B,L)
                seqs_t = torch.where(batch['generate_mask'],seqs_t,seqs_1)
            else:
                seqs_t = seqs_1.detach().clone()
                seqs_t_simplex = seqs_1_simplex.detach().clone()
                seqs_t_prob = seqs_1_prob.detach().clone()

        xt = [rotmats_t, trans_t_c, angles_t, seqs_t, node_embed, edge_embed, gen_mask, res_mask]
        dxtdt = [drotmats_t_dt, dtrans_t_c_dt, dangles_t_dt, dseqs_t_simplex_dt,edge_embed,gen_mask,res_mask]

        def f_teacher(x, t):
            return self.ga_encoder(xt, t)
        
        primals = (xt/sigma_data , t)
        tangents = (
            torch.cos(t)* torch.sin(t)*dxtdt
        )

        (teachere_pred_rotmats_1, teachere_pred_trans_1, teacher_pred_angles_1, teacher_pred_seqs_1_prob), cos_sin_dFdt = \
            torch.autograd.functional.jvp(
                f_teacher, primals, tangents
            )
        teacher_output = teacher_output.detach()
        cos_sin_dFdt = cos_sin_dFdt.detach()

        # denoise
        pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob  = self.ga_encoder(xt, t)
        pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
        pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,torch.clamp(seqs_1,0,19))
        pred_trans_1_c,_ = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
        pred_trans_1_c = pred_trans_1 # implicitly enforce zero center in gen_mask, in this way, we dont need to move receptor when sampling

        norm_scale = 1 / (1 - torch.min(t[...,None], torch.tensor(self._interpolant_cfg.t_normalization_clip))) # yim etal.trick, 1/1-t

        # trans vf loss
        trans_loss = torch.sum((pred_trans_1_c - trans_0_c) - (trans_1_c-trans_0_c))**2*gen_mask[...,None],dim=(-1,-2) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        trans_loss = torch.mean(trans_loss)

        # rots vf loss
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        rot_loss = torch.sum(((gt_rot_vf - pred_rot_vf) * norm_scale)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        rot_loss = torch.mean(rot_loss)

        # backbone aux loss
        gt_bb_atoms = all_atom.to_atom37(trans_1_c, rotmats_1)[:, :, :3] 
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1_c, pred_rotmats_1)[:, :, :3]
        
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        bb_atom_loss = torch.mean(bb_atom_loss)
        # bb_atom_loss = torch.mean(torch.where(t[:,0]>=0.75,bb_atom_loss,torch.zeros_like(bb_atom_loss))) # penalty for near gt point

        # seqs vf loss
        seqs_loss = F.cross_entropy(pred_seqs_1_prob.view(-1,pred_seqs_1_prob.shape[-1]),
                                    torch.clamp(seqs_1,0,19).view(-1), reduction='none').view(pred_seqs_1_prob.shape[:-1]) # (N,L), not softmax
        seqs_loss = torch.sum(seqs_loss * gen_mask, dim=-1) / (torch.sum(gen_mask,dim=-1) + 1e-8)
        seqs_loss = torch.mean(seqs_loss)

        # we should not use angle mask, as you dont know aa type when generating
        # angle_mask_loss = torch.cat([angle_mask,angle_mask],dim=-1) # (B,L,10)
        # angle vf loss
        angle_mask_loss = torsions_mask.to(batch['aa'].device) #(22,5)
        angle_mask_loss = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        angle_mask_loss = torch.cat([angle_mask_loss,angle_mask_loss],dim=-1) # (B,L,10)
        angle_mask_loss = torch.logical_and(batch['generate_mask'][...,None].bool(),angle_mask_loss)
        gt_angle_vf = torus.tor_logmap(angles_t, angles_1)
        gt_angle_vf_vec = torch.cat([torch.sin(gt_angle_vf),torch.cos(gt_angle_vf)],dim=-1)
        pred_angle_vf = torus.tor_logmap(angles_t, pred_angles_1)
        pred_angle_vf_vec = torch.cat([torch.sin(pred_angle_vf),torch.cos(pred_angle_vf)],dim=-1)
        # angle_loss = torch.sum(((gt_angle_vf_vec - pred_angle_vf_vec) * norm_scale)**2*gen_mask[...,None],dim=(-1,-2)) / ((torch.sum(gen_mask,dim=-1)) + 1e-8) # (B,)
        angle_loss = torch.sum(((gt_angle_vf_vec - pred_angle_vf_vec) * norm_scale)**2*angle_mask_loss,dim=(-1,-2)) / (torch.sum(angle_mask_loss,dim=(-1,-2)) + 1e-8) # (B,)
        angle_loss = torch.mean(angle_loss)


        # angle aux loss
        angles_1_vec = torch.cat([torch.sin(angles_1),torch.cos(angles_1)],dim=-1)
        pred_angles_1_vec = torch.cat([torch.sin(pred_angles_1),torch.cos(pred_angles_1)],dim=-1)
        # torsion_loss = torch.sum((pred_angles_1_vec - angles_1_vec)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        torsion_loss = torch.sum((pred_angles_1_vec - angles_1_vec)**2*angle_mask_loss,dim=(-1,-2)) / (torch.sum(angle_mask_loss,dim=(-1,-2)) + 1e-8) # (B,)
        torsion_loss = torch.mean(torsion_loss)

        return {
            "trans_loss": trans_loss,
            'rot_loss': rot_loss,
            'bb_atom_loss': bb_atom_loss,
            'seqs_loss': seqs_loss,
            'angle_loss': angle_loss,
            'torsion_loss': torsion_loss,
        }
    
    @torch.no_grad()
    def sample(self, batch, num_steps = 100, sample_bb=True, sample_ang=True, sample_seq=True):

        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask = batch['generate_mask'],batch['res_mask']
        K = self._interpolant_cfg.seqs.num_classes
        k = self._interpolant_cfg.seqs.simplex_value
        angle_mask_loss = torsions_mask.to(batch['aa'].device)

        #encode
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch)
        # trans_1_c,center = self.zero_center_part(trans_1,gen_mask,res_mask)
        trans_1_c = trans_1
        seqs_1_simplex = self.seq_to_simplex(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)

        #initial noise
        if sample_bb:
            rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
            rotmats_0 = torch.where(batch['generate_mask'][...,None,None],rotmats_0,rotmats_1)
            trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device) # scale with sigma?
            # move center and receptor
            trans_0_c,center = self.zero_center_part(trans_0,gen_mask,res_mask)
            trans_0_c = torch.where(batch['generate_mask'][...,None],trans_0_c,trans_1_c)
        else:
            rotmats_0 = rotmats_1.detach().clone()
            trans_0_c = trans_1_c.detach().clone()
        if sample_ang:
            # angle noise
            angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype) # (B,L,5)
            angles_0 = torch.where(batch['generate_mask'][...,None],angles_0,angles_1)
        else:
            angles_0 = angles_1.detach().clone()
        if sample_seq:
            seqs_0_simplex = k * torch.randn((num_batch,num_res,K), device=batch['aa'].device)
            seqs_0_prob = F.softmax(seqs_0_simplex,dim=-1)
            seqs_0 = sample_from(seqs_0_prob)
            seqs_0 = torch.where(batch['generate_mask'],seqs_0,seqs_1)
            seqs_0_simplex = torch.where(batch['generate_mask'][...,None],seqs_0_simplex,seqs_1_simplex)
        else:
            seqs_0 = seqs_1.detach().clone()
            seqs_0_prob = seqs_1_prob.detach().clone()
            seqs_0_simplex = seqs_1_simplex.detach().clone()

        # Set-up time
        ts = torch.linspace(1.e-2, 1.0, num_steps)
        t1 = ts[0]
        # prot_traj = [{'rotmats':rotmats_0,'trans':trans_0_c,'seqs':seqs_0,'seqs_simplex':seqs_0_simplex,'rotmats_1':rotmats_1,'trans_1':trans_1-center,'seqs_1':seqs_1}]
        clean_traj = []
        rotmats_t1, trans_t1_c, angles_t1, seqs_t1, seqs_t1_simplex = rotmats_0, trans_0_c, angles_0, seqs_0, seqs_0_simplex

        # denoise loop
        for t2 in ts[1:]:
            t = torch.ones((num_batch, 1), device=batch['aa'].device) * t1
            # rots
            pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = self.ga_encoder(t, rotmats_t1, trans_t1_c, angles_t1, seqs_t1, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
            pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
            # trans, move center
            # pred_trans_1_c,center = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
            pred_trans_1_c = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_1_c) # move receptor also
            # angles
            pred_angles_1 = torch.where(batch['generate_mask'][...,None],pred_angles_1,angles_1)
            # seqs
            pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
            pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
            pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
            # seq-angle
            torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
            if not sample_bb:
                pred_trans_1_c = trans_1_c.detach().clone()
                # _,center = self.zero_center_part(trans_1,gen_mask,res_mask)
                pred_rotmats_1 = rotmats_1.detach().clone()
            if not sample_ang:
                pred_angles_1 = angles_1.detach().clone()
            if not sample_seq:
                pred_seqs_1 = seqs_1.detach().clone()
                pred_seqs_1_simplex = seqs_1_simplex.detach().clone()
            clean_traj.append({'rotmats':pred_rotmats_1.cpu(),'trans':pred_trans_1_c.cpu(),'angles':pred_angles_1.cpu(),'seqs':pred_seqs_1.cpu(),'seqs_simplex':pred_seqs_1_simplex.cpu(),
                                    'rotmats_1':rotmats_1.cpu(),'trans_1':trans_1_c.cpu(),'angles_1':angles_1.cpu(),'seqs_1':seqs_1.cpu()})
            # reverse step, also only for gen mask region
            d_t = (t2-t1) * torch.ones((num_batch, 1), device=batch['aa'].device)
            # Euler step
            trans_t2 = trans_t1_c + (pred_trans_1_c-trans_0_c)*d_t[...,None]
            # trans_t2_c,center = self.zero_center_part(trans_t2,gen_mask,res_mask)
            trans_t2_c = torch.where(batch['generate_mask'][...,None],trans_t2,trans_1_c) # move receptor also
            # rotmats_t2 = so3_utils.geodesic_t(d_t[...,None] / (1-t[...,None]), pred_rotmats_1, rotmats_t1)
            rotmats_t2 = so3_utils.geodesic_t(d_t[...,None] * 10, pred_rotmats_1, rotmats_t1)
            rotmats_t2 = torch.where(batch['generate_mask'][...,None,None],rotmats_t2,rotmats_1)
            # angles
            angles_t2 = torus.tor_geodesic_t(d_t[...,None],pred_angles_1, angles_t1)
            angles_t2 = torch.where(batch['generate_mask'][...,None],angles_t2,angles_1)
            # seqs
            seqs_t2_simplex = seqs_t1_simplex + (pred_seqs_1_simplex - seqs_0_simplex) * d_t[...,None]
            seqs_t2 = sample_from(F.softmax(seqs_t2_simplex,dim=-1))
            seqs_t2 = torch.where(batch['generate_mask'],seqs_t2,seqs_1)
            # seq-angle
            torsion_mask = angle_mask_loss[seqs_t2.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            angles_t2 = torch.where(torsion_mask.bool(),angles_t2,torch.zeros_like(angles_t2))
            
            if not sample_bb:
                trans_s_c = trans_1_c.detach().clone()
                rotmats_t2 = rotmats_1.detach().clone()
            if not sample_ang:
                angles_t2 = angles_1.detach().clone()
            if not sample_seq:
                seqs_t2 = seqs_1.detach().clone()
            rotmats_t1, trans_t1_c, angles_t1, seqs_t1, seqs_t1_simplex = rotmats_t2, trans_t2_c, angles_t2, seqs_t2, seqs_t2_simplex
            t1 = t2

        # final step
        t1 = ts[-1]
        t = torch.ones((num_batch, 1), device=batch['aa'].device) * t1
        pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = self.ga_encoder(t, rotmats_t1, trans_t1_c, angles_t1, seqs_t1, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
        pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
        # move center
        # pred_trans_1_c,center = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
        pred_trans_1_c = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_1_c) # move receptor also
        # angles
        pred_angles_1 = torch.where(batch['generate_mask'][...,None],pred_angles_1,angles_1)
        # seqs
        pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
        pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
        pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
        # seq-angle
        torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
        if not sample_bb:
            pred_trans_1_c = trans_1_c.detach().clone()
            # _,center = self.zero_center_part(trans_1,gen_mask,res_mask)
            pred_rotmats_1 = rotmats_1.detach().clone()
        if not sample_ang:
            pred_angles_1 = angles_1.detach().clone()
        if not sample_seq:
            pred_seqs_1 = seqs_1.detach().clone()
            pred_seqs_1_simplex = seqs_1_simplex.detach().clone()
        clean_traj.append({'rotmats':pred_rotmats_1.cpu(),'trans':pred_trans_1_c.cpu(),'angles':pred_angles_1.cpu(),'seqs':pred_seqs_1.cpu(),'seqs_simplex':pred_seqs_1_simplex.cpu(),
                                'rotmats_1':rotmats_1.cpu(),'trans_1':trans_1_c.cpu(),'angles_1':angles_1.cpu(),'seqs_1':seqs_1.cpu()})
        
        return clean_traj


