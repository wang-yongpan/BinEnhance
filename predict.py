import argparse
import json
import os
import random
import time
from multiprocessing import Process

import dgl
import networkx as nx
import numpy as np
import torch
# from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity
cos_nn = nn.CosineSimilarity(dim=1, eps=1e-10)
from utils import get_func_classes, generate_eval_data
from sklearn.metrics.pairwise import cosine_similarity

def check_kbg(graph, relss_type):
    ret = []
    ret_g = set(relss_type.keys())
    edge_num = 0
    for i in graph.edges.data():
        if i[2]["rel_type"] in relss_type:
            edge_num += 1
        ret.append(i[2]["rel_type"])
    nodes_num = 0
    for n in graph.nodes():
        if len(n) == 0:
            continue
        if graph.nodes[n]["type"] == "user function":
            nodes_num += 1
        if graph.nodes[n]["type"] == "const strings":
            nodes_num += 1
    ret = set(ret)
    if nodes_num < 5 or len(ret.intersection(ret_g)) < 1 or edge_num > 5000:
        return True
    return False

def load_graph_data(filePath, relss_type):
    graphs = {}
    for file in tqdm(os.listdir(filePath), desc="it is init graphs..."):
        filepath = os.path.join(filePath, file)
        graph = nx.read_gpickle(filepath)
        if check_kbg(graph, relss_type):
            continue
        graphs[file.split(".pkl")[0]] = graph
    return graphs
    pass

def check_kbg_neg(graph, relss_type):
    ret = []
    for i in graph.edges.data():
        # if i[2]["rel_type"] in relss_type:
        #     edge_num += 1
        ret.append(i[2]["rel_type"])
    ret_g = set(relss_type.keys())
    nodes_num = 0
    for n in graph.nodes():
        if len(n) == 0:
            continue
        if graph.nodes[n]["type"] == "user function":
            nodes_num += 1
        if graph.nodes[n]["type"] == "const strings":
            nodes_num += 1
    ret = set(ret)
    if nodes_num < 5 or len(ret.intersection(ret_g)) < 1:
        return True
    return False

def check_is_suitable(graph, user, relss_type):
    if int(graph.nodes[user]["block_num"]) < 5 or graph.nodes[user]["type"] != "user function":
        return False
    re = [i[2]["rel_type"] for i in graph.in_edges(user, data=True)] + [i[2]["rel_type"] for i in graph.out_edges(user, data=True)]
    ret_g = set(relss_type.keys())
    if len(set(re).intersection(ret_g)) < 1 or len(re) <= 5:
        return False
    return True

def load_graph_data_neg(filePath, relss_type):
    graphs = {}
    for file in tqdm(os.listdir(filePath), desc="it is init graphs..."):
        filepath = os.path.join(filePath, file)
        graph = nx.read_gpickle(filepath)
        # if check_kbg_neg(graph, relss_type):
        #     continue
        graphs[file.split(".pkl")[0]] = graph
    return graphs
    pass

def get_graph_nodes(graph):
    user = {}
    strs = {}
    index = 0
    for n in graph.nodes():
        if len(n) == 0:
            continue
        if graph.nodes[n]["type"] == "user function":
            user[n] = str(index) + "_" + str(graph.nodes[n]["block_num"])
            index += 1
        if graph.nodes[n]["type"] == "const strings":
            strs[n] = index
            index += 1
    return user, strs

def get_graph_nodes_sub(graph):
    user = {}
    strs = {}
    for n in graph.nodes():
        if len(n) == 0:
            continue
        if graph.nodes[n]["type"] == "user function":
            user[n] = graph.nodes[n]["block_num"]
        if graph.nodes[n]["type"] == "const strings":
            strs[n] = 1
    return user, strs

def get_next_nodes(g, ns, edge_num, all_edges, all_nodes, max_edge_num, rels, pname, funcs_embeddings):
    nodes = []
    edges = []
    for node in ns:
        if g.nodes[node]["type"] != "user function":
            continue
        for i in g.in_edges(node, data=True):
            in_node = i[0]
            if in_node in all_nodes:
                continue
            if g.nodes[in_node]["type"] == "user function":
                in_fname = pname + "|||" + in_node
                if str(i) in all_edges or i[2]['rel_type'] not in rels or in_fname not in funcs_embeddings:
                    continue
                else:
                    all_edges[str(i)] = 1
                edges.append(i)
                nodes.append(in_node)
                if len(edges) + edge_num > max_edge_num:
                    return nodes, edges, all_edges
            pass
        for i in g.out_edges(node, data=True):
            out_node = i[1]
            if out_node in all_nodes:
                continue
            if g.nodes[out_node]["type"] == "user function":
                out_fname = pname + "|||" + out_node
                if str(i) in all_edges or i[2]['rel_type'] not in rels or out_fname not in funcs_embeddings:
                    continue
                else:
                    all_edges[str(i)] = 1
                edges.append(i)
                nodes.append(out_node)
                if len(edges) + edge_num > max_edge_num:
                    return nodes, edges, all_edges
            
            if g.nodes[out_node]["type"] == "const strings":
                edge = (i[1], i[0], i[2])
                if str(edge) in all_edges or i[2]['rel_type'] not in rels:
                    continue
                else:
                    all_edges[str(edge)] = 1
                edges.append(edge)
                nodes.append(out_node)
                if len(edges) + edge_num > max_edge_num:
                    return nodes, edges, all_edges
    return nodes, edges, all_edges
    pass

def get_subgraph(g, funcName, funcs_embeddings, strs_embeddings, depth, max_edge_num, relss_type, loop=1):
    users, strs = get_graph_nodes_sub(g)
    node_n = funcName.split("|||")[-1]
    try:
        tp = g.nodes[node_n]["type"]
    except:
        return None
    nodes = [node_n]
    edges = []
    pname = funcName.replace("|||" + node_n, "")
    all_edges = {}
    node2id = {}
    node2id[node_n] = 0
    nodes_embeds = [[]]
    for i in range(depth):
        a_nodes, a_edges, all_edges = get_next_nodes(g, nodes, len(edges), all_edges, node2id, max_edge_num, relss_type, pname, funcs_embeddings)
        nodes = []
        for a_node in a_nodes:
            if a_node not in node2id:
                node2id[a_node] = len(node2id)
                nodes_embeds.append([])
                nodes.append(a_node)
        edges.extend(a_edges)
    start_edges = []
    end_edges = []
    relss = []
    if loop == 1:
        for node, nids in node2id.items():
            start_edges.append(nids)
            end_edges.append(nids)
            relss.append(relss_type["self_loop"])
    for edge in edges:
        # al_edge = str(node2id [edge[0]]) + "_" + str(node2id[edge[1]]) + "_" + str(self.rels[edge[2]['rel_type']])
        # if al_edge not in al_edges:
        start_edges.append(node2id[edge[0]])
        end_edges.append(node2id[edge[1]])
        relss.append(relss_type[edge[2]['rel_type']])
            # al_edges[al_edge] = 1
    g = dgl.graph((start_edges, end_edges))
    for node in node2id.keys():
        if node in users:
            fname = funcName.replace("|||" + node_n, "") + "|||" + node
            nodes_embeds[node2id[node]] = funcs_embeddings[fname]
        
        if node in strs:
            str_text = node[9:]
            nodes_embeds[node2id[node]] = strs_embeddings[str_text]
    g.ndata["feature"] = torch.tensor(nodes_embeds)
    g.edata["rel_type"] = torch.tensor(relss)
    return g

def generate_graph(graph, funcs_embeddings, strs_embeddings, fn, relss_type):
    start_edges = []
    end_edges = []
    func_index = {}
    nodes = []
    relss = []
    al_edges = {}
    user_functions, strs = get_graph_nodes(graph)
    for u, ib in user_functions.items():
        u_index = int(user_functions[u].split("_")[0])
        func_index[fn + "|||" + u] = u_index
        for i in graph.in_edges(u, data=True):
            in_func = i[0]
            if in_func in user_functions:
                edge = (int(user_functions[in_func].split("_")[0]), u_index, relss_type[str(i[2]["rel_type"])])
                al_edge = str(user_functions[in_func].split("_")[0]) + "_" + str(u_index) + "_" + str(relss_type[str(i[2]["rel_type"])])
                if al_edge not in al_edges:
                    start_edges.append(edge[0])
                    end_edges.append(edge[1])
                    relss.append(edge[2])
                    al_edges[al_edge] = 1
        for i in graph.out_edges(u, data=True):
            in_func = i[1]
            if in_func in strs:
                index = strs[in_func]
            else:
                continue
            edge = (u_index, int(index), relss_type[str(i[2]["rel_type"])])
            al_edge = str(u_index) + "_" + str(index) + "_" + str(relss_type[str(i[2]["rel_type"])])
            if al_edge not in al_edges:
                start_edges.append(edge[1])
                end_edges.append(edge[0])
                relss.append(edge[2])
                al_edges[al_edge] = 1
    g = dgl.graph((start_edges, end_edges))
    ns = len(g.nodes())
    for i in range(ns):
        nodes.append([])
    for u, ib in user_functions.items():
        fname = fn + "|||" + u
        u_index = int(user_functions[u].split("_")[0])
        if u_index < len(nodes):
            nodes[u_index] = funcs_embeddings[fname]
        else:
            fl = u_index - len(nodes) + 1
            while fl > 0:
                nodes.append([])
                fl -= 1
            nodes[u_index] = funcs_embeddings[fname]
            if u_index >= len(g.nodes()):
                node_num = u_index - len(g.nodes()) + 1
                g.add_nodes(node_num)
    str_texts = []
    str_index = []
    for st, i in strs.items():
        str_texts.append(st[9:])
        str_index.append(int(i))
    if len(str_texts) != 0:
        for i in range(len(str_index)):
            nodes[str_index[i]] = strs_embeddings[str_texts[i]]
    g.ndata["feature"] = torch.tensor(nodes)
    g.edata["rel_type"] = torch.tensor(relss)
    g = dgl.add_self_loop(g, edge_feat_names=['rel_type'], fill_data=len(relss_type))
    return g, func_index

def set_random_seed(seed_value=1234):
    torch.manual_seed(seed_value)  # cpu  vars
    np.random.seed(seed_value)  # cpu vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

def cal_map_metrics(sims, labels, lens):
    q = len(sims)
    map = 0.0
    for i in range(q):
        ap = 0.0
        cos_num = 0.0
        sim = sims[i]
        label = labels[i]
        sorted_list = sorted([(si, li) for si, li in zip(sim, label)], reverse=True)
        # with open("save_sim.txt", "a") as f:
        #     f.write(str(sorted_list) + "\n")
        label = [li for _, li in sorted_list]
        for j in range(min(10, lens[i])):
            if int(label[j]) == 1:
                cos_num += 1.0
                ap = ap + cos_num / float(j + 1)
        ap = ap / lens[i]
        map += ap
    ans = map / float(q)
    return ans

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

def calculate_rtf_similarity_two(func_name, basic_dict):
    if func_name[0] in basic_dict and func_name[1] in basic_dict:
        a_func_dict = basic_dict[func_name[0]]
        p_func_dict = basic_dict[func_name[1]]
        sim = comp_jaccard_sim_weight(a_func_dict, p_func_dict)
    else:
        sim = 0.01
    return sim

def combine_eval(model_name, f_strings, global_vars, external_funcs, hsg_data_path, func_embedding_path, str_embedding_path,  relss_type={}, args=None,
                 eval_type=5, model="", tp="noinline", hsg_path="", poolsize=10000, yarchc="", yoptc="", tarchc="", toptc="", eval_p="", savepaths=""):
    set_random_seed()
    eval_data_path = eval_p + model + "/" + model + "_" + str(poolsize) + "_eval_data_" + str(eval_type) + yarchc + yoptc + tarchc + toptc + ".json"
    with open(func_embedding_path, "r") as f:
        func_embeddings = json.load(f)
    if not os.path.exists(eval_data_path):
        classes = get_func_classes(func_embeddings,  tp=tp, MIN_BLOCKS=eval_type, hsg_path=hsg_path)
        eval_datas = generate_eval_data(classes, poolsize=poolsize, yarchs=yarchc, yoptc=yoptc, tarchs=tarchc, toptc=toptc)
        e = {}
        e["d"] = eval_datas
        if not os.path.exists(eval_p + model):
            os.makedirs(eval_p + model)
        with open(eval_data_path, "w") as f:
            json.dump(e, f)
    else:
        with open(eval_data_path, "r") as f:
            eval_datas = json.load(f)["d"]
    model_path = model_name + "/BinEnhance-best.pt"
    model = torch.load(model_path).to(args.gpu)

    
    with open(str_embedding_path, "r") as f:
        str_embedding = json.load(f)

    predict_num = 0
    lens = []
    sims = []
    labels = []
    subgraphs_embed = {}
    savepaths = model_name + "sub_embed.json" if savepaths == "" else savepaths
    if os.path.exists(savepaths):
        with open(savepaths, "r") as f:
            subgraphs_embed = json.load(f)
    else:
        graphs = load_graph_data_neg(hsg_data_path, relss_type)
        for fn, embeds in tqdm(func_embeddings.items(), desc="it is init subgraphs..."):
            fn_type = fn.split("|||")[0]
            subgraphs_embed[fn] = embeds
            if fn_type not in graphs:
                continue
            graph = graphs[fn_type]
            sub = get_subgraph(graph, fn, func_embeddings, str_embedding, args.num_layers, args.max_edge_num, relss_type=relss_type)
            if sub is None or sub.num_nodes() == 1 or sub.num_edges() == 1:
                continue
            embed = model.forward_once(sub).cpu().detach().numpy().tolist()
            subgraphs_embed[fn] = embed
        with open(savepaths, "w") as f:
            json.dump(subgraphs_embed, f)
    funcNames = []
    predict_num = 0
    lens = []
    labels = []
    cs = []
    ss = []
    es = []
    ds = []
    tq = tqdm(eval_datas, position=0, ncols=80)
    pos = len(eval_datas[0])
    for eds in tq:
        leng = 0
        label = []
        a_fn = eds[0][0]
        if a_fn not in subgraphs_embed:
            continue
        feats_a = [subgraphs_embed[a_fn]]
        feats_b = []
        sim_s = []
        sim_e = []
        sim_d = []
        for ed in eds:
            b_fn = ed[1]
            if b_fn not in subgraphs_embed:
                continue
            funcNames = [a_fn, b_fn]
            feats_b.append(subgraphs_embed[b_fn])
            label.append(ed[2])
            leng += ed[2]
            predict_num += 1
            sim_s.append(calculate_rtf_similarity_two(funcNames, f_strings))
            sim_e.append(calculate_rtf_similarity_two(funcNames, external_funcs))
            sim_d.append(calculate_rtf_similarity_two(funcNames, global_vars))
        if leng == 0:
            continue
        cos_dist = cosine_similarity(np.array(feats_a), np.array(feats_b)).reshape(-1)
        cs.append(cos_dist)
        ss.append(sim_s)
        es.append(sim_e)
        ds.append(sim_d)
        lens.append(leng)
        labels.append(label)
    cs = np.array(cs).flatten()
    ss = np.array(ss).flatten()
    es = np.array(es).flatten()
    ds = np.array(ds).flatten()
    combined = torch.tensor(np.stack((cs, ss, ds, es), axis=1)).float().to(model.gpu) 
    sim = model.combine_sims(combined)
    sim = model.sim_activate(sim).cpu().detach().numpy().tolist()
    sims = [sim[i:i + pos] for i in range(0, len(sim), pos)]
    print("predict num:" + str(predict_num) + "/" + str(sum([len(label) for label in labels])))
    print("it is calculating map score ...")
    ans = cal_map_metrics(sims, labels, lens)
    print("block_num>" + str(eval_type) + ":eval completed! MAP:" + str(ans))
