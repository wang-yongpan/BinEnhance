import dgl.function as fn
import dgl
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Identity
from graph_transformer import RGCNLayer

def comp_jaccard_sim_weight(i1, i2):
    i1 = set(i1)
    i2 = set(i2)
    i1_un_i2 = i1.union(i2)
    score2 = len(i1_un_i2)
    if score2 == 0:
        return 0.01
    i1_in_i2 = i1.intersection(i2)
    score1 = len(i1_in_i2)
    sim = score1 / score2
    return sim

def calculate_rtf_similarity(func_names, basic_dict, gpu=0):
    p_sims = []
    n_sims = []
    for func_name in func_names:
        if func_name[0] in basic_dict:
            a_func_dict = basic_dict[func_name[0]]
            if func_name[1] not in basic_dict:
                p_sim = 0.01
            else:
                p_func_dict = basic_dict[func_name[1]]
                p_sim = comp_jaccard_sim_weight(a_func_dict, p_func_dict)
            if func_name[2] not in basic_dict:
                n_sim = 0.01
            else:
                n_func_dict = basic_dict[func_name[2]]
                n_sim = comp_jaccard_sim_weight(a_func_dict, n_func_dict)
        else:
            p_sim = 0.01
            n_sim = 0.01
        p_sims.append(p_sim)
        n_sims.append(n_sim)
    return torch.tensor(p_sims).to(gpu), torch.tensor(n_sims).to(gpu)

class RGCN_Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels,
                 num_bases=-1):
        super(RGCN_Model, self).__init__()
        self.gpu = 0
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.negative_slope = 0.2
        self.num_hidden_layers = 4
        self.build_model()
        self.cos1 = nn.CosineSimilarity(dim=1, eps=1e-10)
        self.combine_sims = nn.Linear(4, 1)
        self.sim_activate = nn.Tanh()
        self.residual = 1
        if self.residual:
            if self.in_dim != self.h_dim:
                self.res_fc = nn.Linear(self.in_dim, self.h_dim, bias=False)
            else:
                self.res_fc = Identity()

    def build_model(self):
        self.layers = nn.ModuleList()
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        for _ in range(self.num_hidden_layers-2):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    def get_graph_embedding(self, g, index):
        g_list = dgl.unbatch(g)
        embed_h = g_list[0].ndata['h'][int(index[0])].unsqueeze(0)
        if self.residual:
            embed_f = g_list[0].ndata['feature'][int(index[0])].unsqueeze(0)
        for i in range(1, len(g_list)):
            embed_h = torch.cat((embed_h, g_list[i].ndata['h'][int(index[i])].unsqueeze(0)), dim=0)
            if self.residual:
                embed_f = torch.cat((embed_f, g_list[i].ndata['feature'][int(index[i])].unsqueeze(0)), dim=0)
        if self.residual:
            embed_r = self.res_fc(embed_f)
            embed = embed_h + embed_r
        else:
            embed = embed_h
        return embed
        pass

    def build_input_layer(self):
        return RGCNLayer(self.in_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.leaky_relu, negative_slope=self.negative_slope)#, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.leaky_relu, negative_slope=self.negative_slope)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=F.leaky_relu, negative_slope=self.negative_slope)
    def forward_once(self, g):
        g = g.to(self.gpu)
        g.ndata['h'] = g.ndata['feature'].to(self.gpu)
        for layer in self.layers:
            g = layer(g)
        embed = g.ndata['h'][0]
        if self.residual:
            embed_r = self.res_fc(g.ndata['feature'][0])
            embed = g.ndata['h'][0] + embed_r
        return embed
        
    def forward(self, batch, batch_size, f_strings, global_data, external_funcs):
        a_g, p_g, n_g, funcNames = batch
        a_g = a_g.to(self.gpu)
        p_g = p_g.to(self.gpu)
        n_g = n_g.to(self.gpu)
        a_g.ndata['h'] = a_g.ndata['feature'].to(self.gpu)
        p_g.ndata['h'] = p_g.ndata['feature'].to(self.gpu)
        n_g.ndata['h'] = n_g.ndata['feature'].to(self.gpu)
        for layer in self.layers:
            a_g = layer(a_g)
            p_g = layer(p_g)
            n_g = layer(n_g)
        a_embed = self.get_graph_embedding(a_g, np.zeros(batch_size).tolist())
        p_embed = self.get_graph_embedding(p_g, np.zeros(batch_size).tolist())
        n_embed = self.get_graph_embedding(n_g, np.zeros(batch_size).tolist())
        cos_dist_p = self.cos1(a_embed, p_embed)
        cos_dist_n = self.cos1(a_embed, n_embed)

        s_sim_p, s_sim_n = calculate_rtf_similarity(funcNames, f_strings, self.gpu)
        d_sim_p, d_sim_n = calculate_rtf_similarity(funcNames, global_data, self.gpu)
        ef_sim_p, ef_sim_n = calculate_rtf_similarity(funcNames, external_funcs, self.gpu)

        sim_p = torch.stack((cos_dist_p, s_sim_p, d_sim_p, ef_sim_p), dim=1)
        sim_n = torch.stack((cos_dist_n, s_sim_n, d_sim_n, ef_sim_n), dim=1)
        sim_p = self.sim_activate(self.combine_sims(sim_p))
        sim_n = self.sim_activate(self.combine_sims(sim_n))
        loss = (torch.mean((1 - sim_p) ** 2) + torch.mean((sim_n + 1) ** 2)) / 2
        return sim_p, sim_n, loss
        pass
