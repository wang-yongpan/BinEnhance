import json
import multiprocessing
import os
import time
from multiprocessing import Process

from annoy import AnnoyIndex
import numpy as np
import dgl
import torch
from dgl.data import DGLDataset
import networkx as nx
import random
from tqdm import tqdm


class eesg_datasets(DGLDataset):

    def __init__(self, filePath, funcs_embeddings, strs_embeddings, depth, data_base, mode, args):
        self.max_edge_num = args.max_edge_num
        self.max_node_num = args.max_node_num
        self.depth = depth
        name_type = args.name + "_" + str(self.depth) + "_" + str(args.funcDim)
        self.funcs_embeddings = funcs_embeddings
        self.strs_embeddings = strs_embeddings
        self.filePath = filePath
        relss_type = {}
        relss_type["call"] = len(relss_type)
        relss_type["be_called"] = len(relss_type)
        relss_type["string_dependency"] = len(relss_type)
        relss_type["variable_dependency"] = len(relss_type)
        relss_type["string_dependency_loop"] = len(relss_type)
        relss_type["variable_dependency_loop"] = len(relss_type)
        relss_type["address_after"] = len(relss_type)
        relss_type["address_before"] = len(relss_type)
        relss_type["string_use"] = len(relss_type)
        relss_type["string_be_used"] = len(relss_type)
        relss_type["self_loop"] = len(relss_type)
        self.rels = relss_type
        self.funcDim = args.funcDim
        self.sample_max = args.sample_max
        self.negative_rand = args.negative_rand
        self.process_num = 30
        self.random = random.Random(1234)
        self.sub_graphs_path = os.path.join(os.path.join(data_base, str(args.fis)), str(mode) + "_subgraphs" + str(self.max_edge_num) + "_" + str(self.max_node_num) + str(name_type) + ".bin") 
        self.fid_name_path = os.path.join(os.path.join(data_base, str(args.fis)), str(mode) + "_fid2name" + str(self.max_edge_num) + "_" + str(self.max_node_num) + str(name_type) + ".json")
        self.func_classes_path = os.path.join(os.path.join(data_base, str(args.fis)), str(mode) + "_func_classes" + str(self.max_edge_num) + "_" + str(self.max_node_num) + str(name_type) + ".json")
        self.subgraphs = {}
        if mode != "predict":
            if not self.load_sub_data():
                self.init_graphs()
                self.init_func_classes()
                self.multi_init_subgraphs()
            self.init_annoy()
            self.shuffle()
        super(eesg_datasets, self).__init__(name="eesg_datasets")
        pass

    def check_kbg(self, graph):
        ret = []
        for i in graph.edges.data():
            ret.append(i[2]["rel_type"])
        ret_g = set(self.rels.keys())
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
        pass

    def init_graphs(self):
        self.graphs = {}
        self.graph2id = {}
        self.id2graph = {}
        for file in tqdm(os.listdir(self.filePath), desc="it is init graphs...", ncols=80):
            filename = file.split(".pkl")[0]
            filepath = os.path.join(self.filePath, file)
            graph = nx.read_gpickle(filepath)
            if self.check_kbg(graph):
                continue
            self.graph2id[filename] = len(self.graph2id)
            self.id2graph[len(self.id2graph)] = filename
            self.graphs[len(self.graphs)] = graph
        pass

    def init_func_classes(self):
        self.fid2name = {}
        self.fid2gid = {}
        self.func_classes = []
        func_cs = []
        func2id = {}
        cop_func = []
        func_id = 0
        for func_name, func_embed in tqdm(self.funcs_embeddings.items(), desc="it is filter func_class...", ncols=80):
            file_graphid = func_name.replace("|||" + func_name.split("|||")[-1], "")
            if file_graphid in self.graph2id:
                file_graph = self.graphs[self.graph2id[file_graphid]]
            else:
                continue
            if self.check_is_suitable(file_graph, func_name.split("|||")[-1]):
                cop_func.append(func_name)
                funcN = func_name.split("_")[0] + "|||" + func_name.split("_")[5] + "|||" + func_name.split("|||")[-1]
                if funcN not in func2id:
                    func2id[funcN] = len(func2id)
        for fi, id in func2id.items():
            func_cs.append([])
        for func_name in tqdm(cop_func, desc="it is init func_class...", ncols=80):
            funcN = func_name.split("_")[0] + "|||" + func_name.split("_")[5] + "|||" + func_name.split("|||")[-1]
            func_cs[func2id[funcN]].append(func_id)
            self.fid2name[str(func_id)] = func_name
            file_graphid = func_name.replace("|||" + func_name.split("|||")[-1], "")
            self.fid2gid[func_id] = self.graph2id[file_graphid]
            func_id += 1
        for fcs in func_cs:
            if len(fcs) > 1:
                self.func_classes.append(fcs)
        pass

    def init_annoy(self):
        self.annoy = AnnoyIndex(self.funcDim, 'angular')
        for c in tqdm(self.subgraphs.keys(), desc="it is init annoy...", ncols=80):
            self.annoy.add_item(int(c), self.funcs_embeddings[self.fid2name[str(c)]])
        self.annoy.build(10)
        pass

    def check_is_suitable(self, graph, user):
        try:
            if int(graph.nodes[user]["block_num"]) < 5 or graph.nodes[user]["type"] != "user function":
                return False
        except:
            return False
        re = [i[2]["rel_type"] for i in graph.in_edges(user, data=True)] + [i[2]["rel_type"] for i in
                                                                            graph.out_edges(user, data=True)]
        ret_g = set(self.rels.keys())
        if len(set(re).intersection(ret_g)) < 1 or len(re) <= 5:
            return False
        return True

    def get_graph_nodes(self, graph):
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

    def get_next_nodes(self, g, ns, edge_num, node_num, all_edges, all_nodes, pname):
        nodes = []
        edges = []
        all_es = all_edges
        for node in ns:
            if g.nodes[node]["type"] != "user function":
                continue
            for i in g.in_edges(node, data=True):
                in_node = i[0]
                if in_node in all_nodes:
                    continue
                if g.nodes[in_node]["type"] == "user function":
                    in_fname = pname + "|||" + in_node
                    if str(i) in all_es or i[2]['rel_type'] not in self.rels or in_fname not in self.funcs_embeddings:
                        continue
                    else:
                        all_es[str(i)] = 1
                    edges.append(i)
                    nodes.append(in_node)
                    if len(edges) + edge_num > self.max_edge_num or len(nodes) + node_num > self.max_node_num:
                        return nodes, edges, all_es
                pass
            for i in g.out_edges(node, data=True):
                out_node = i[1]
                if out_node in all_nodes:
                    continue
                if g.nodes[out_node]["type"] == "user function":
                    out_fname = pname + "|||" + out_node
                    if str(i) in all_es or i[2]['rel_type'] not in self.rels or out_fname not in self.funcs_embeddings:
                        continue
                    else:
                        all_es[str(i)] = 1
                    edges.append(i)
                    nodes.append(out_node)
                    if len(edges) + edge_num > self.max_edge_num or len(nodes) + node_num > self.max_node_num:
                        return nodes, edges, all_es
                if g.nodes[out_node]["type"] == "const strings":
                    edge = (i[1], i[0], i[2])
                    if str(edge) in all_es or i[2]['rel_type'] not in self.rels:
                        continue
                    else:
                        all_es[str(edge)] = 1
                    edges.append(edge)
                    nodes.append(out_node)
                    if len(edges) + edge_num > self.max_edge_num or len(nodes) + node_num > self.max_node_num:
                        return nodes, edges, all_es
        return nodes, edges, all_es
        pass

    def get_subgraph(self, g, funcName):
        users, strs = self.get_graph_nodes(g)
        node_n = funcName.split("|||")[-1]
        nodes = [node_n]
        edges = []
        all_edges = {}
        node2id = {}
        node2id[node_n] = 0
        nodes_embeds = [[]]
        pname = funcName.replace("|||" + node_n, "")
        for i in range(self.depth):
            a_nodes, a_edges, all_edges = self.get_next_nodes(g, nodes, len(edges), len(node2id), all_edges, node2id, pname)
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
        for edge in edges:
            start_edges.append(node2id[edge[0]])
            end_edges.append(node2id[edge[1]])
            relss.append(self.rels[edge[2]['rel_type']])
        if len(start_edges) == 0:
            return None
        
        for node, nids in node2id.items():
            start_edges.append(nids)
            end_edges.append(nids)
            relss.append(self.rels["self_loop"])
        g = dgl.graph((start_edges, end_edges))
        for node in node2id.keys():
            if node in users:
                fname = funcName.replace("|||" + node_n, "") + "|||" + node
                nodes_embeds[node2id[node]] = self.funcs_embeddings[fname]
            if node in strs:
                str_text = node[9:]
                nodes_embeds[node2id[node]] = self.strs_embeddings[str_text]
        g.ndata["feature"] = torch.tensor(nodes_embeds)
        g.edata["rel_type"] = torch.tensor(relss)
        return g

    def process(self):
        pass

    def load_sub_data(self):
        if os.path.exists(self.sub_graphs_path):
            gs, func_ids_lab = dgl.load_graphs(self.sub_graphs_path)
            func_ids = func_ids_lab["func_ids"].tolist()
            for g_i in range(len(gs)):
                self.subgraphs[func_ids[g_i]] = gs[g_i]
            with open(self.fid_name_path, "r") as f:
                self.fid2name = json.load(f)
            with open(self.func_classes_path, "r") as f:
                self.func_classes = json.load(f)["funcs"]

            return True
        return False

    def save_sub_data(self, subgraphs, func_ids):
        if not os.path.exists(self.sub_graphs_path):
            dgl.save_graphs(self.sub_graphs_path, subgraphs, {"func_ids": torch.tensor(func_ids)})
            with open(self.fid_name_path, "w") as f:
                json.dump(self.fid2name, f)
            funcs = {}
            funcs["funcs"] = self.func_classes
            with open(self.func_classes_path, "w") as f:
                json.dump(funcs, f)
        pass

    def multi_init_subgraphs(self):
        process_num = self.process_num
        random.shuffle(self.func_classes)
        fileList = self.func_classes
        p_list = []
        for i in range(process_num):
            files = fileList[int((i) / process_num * len(fileList)): int((i + 1) / process_num * len(fileList))]
            p = Process(target=self.init_subgraphs, args=(files, i))
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()
        print("begin save data...")
        subgraphs = []
        func_idss = []
        for pid in range(process_num):
            save_p = self.sub_graphs_path.replace(".bin", "_" + str(pid) + ".bin")
            if not os.path.exists(save_p):
                print(save_p + "is not exist...")
                continue
            gs, func_ids_lab = dgl.load_graphs(save_p)
            func_ids = func_ids_lab["func_ids"].tolist()
            for g_i in range(len(gs)):
                g = gs[g_i]
                # if g.num_nodes() == 1 or g.num_edges() == 1:
                #     continue
                self.subgraphs[func_ids[g_i]] = g
                subgraphs.append(g)
                func_idss.append(func_ids[g_i])
            os.remove(save_p)
        print("save data complete!")
        self.save_sub_data(subgraphs, func_idss)

    def init_subgraphs(self, func_cs, pid):
        tq = tqdm(func_cs, desc="it is init_subgraphs...", ncols=80)
        save_p = self.sub_graphs_path.replace(".bin", "_" + str(pid) + ".bin")
        subgraphs = []
        fun_ids = []
        for cs in tq:
            cs_len = len(cs)
            for func_index in range(cs_len):
                func_id = cs[func_index]
                graph = self.graphs[self.fid2gid[func_id]]
                funcName = self.fid2name[str(func_id)]
                subgraph = self.get_subgraph(graph, funcName)
                if subgraph == None:
                    continue
                subgraphs.append(subgraph)
                fun_ids.append(func_id)
        if len(subgraphs) != 0:
            dgl.save_graphs(save_p, subgraphs, {"func_ids": torch.tensor(fun_ids)})
        tq.set_description("Process-" + str(pid) + "is Over!")
        pass

    def positive_sample(self, cs_len, cs, c_id):
        flag = 0
        p_id = 0
        for i in range(10):
            p_id = cs[np.random.randint(cs_len)]
            if c_id != p_id and p_id in self.subgraphs:  # and self.check_is_suitable(graphs[p_id]):
                flag = 1
                break
        return flag, p_id
        pass

    def negative_sample_random(self, classes, anchor_func_id, cs, rand=1.0):
        ran = self.random.random()
        if ran <= rand:
            new_cs_set = set(cs)
            ans = []
            loop_max = 1
            while len(ans) == 0 and loop_max <= 3:
                anchor_func_embed = self.funcs_embeddings[self.fid2name[str(anchor_func_id)]]
                nns = self.annoy.get_nns_by_vector(anchor_func_embed, 10 ** loop_max)
                nns_set = set(nns)
                ans = nns_set - nns_set.intersection(new_cs_set)
                loop_max += 1
            if len(ans) == 0:
                n_id = self.negative_sample_select(classes, anchor_func_id)
            else:
                ans = list(ans)
                for a in ans:
                    if a not in self.subgraphs:
                        continue
                    self.rand_num += 1
                    n_id = list(ans)[0]
                    return n_id
                n_id = self.negative_sample_select(classes, anchor_func_id)
            # n_id = np.random.randint(len(nns))
            # while n_id in cs:
            #     n_id = np.random.randint(len(nns))
        else:
            n_id = self.negative_sample_select(classes, anchor_func_id)
        return n_id
        pass

    def negative_sample_select(self, classes, c_id):
        n_ids = np.random.randint(len(classes))
        while (len(classes[n_ids]) == 0) or (c_id in classes[n_ids]) or classes[n_ids][0] not in self.subgraphs:
            n_ids = np.random.randint(len(classes))
        nid = np.random.randint(len(classes[n_ids]))
        while classes[n_ids][nid] not in self.subgraphs:
            nid = np.random.randint(len(classes[n_ids]))
        n_id = classes[n_ids][nid]
        return n_id

    def shuffle(self):
        np.random.shuffle(self.func_classes)
        self.a_fids = []
        self.p_fids = []
        self.n_fids = []
        self.rand_num = 0
        self.init_pairs()
        print("selective negative sample num:" + str(self.rand_num) + "/" + str(len(self.a_fids)))

    def init_pairs(self):
        tq = tqdm(self.func_classes, desc="it is init pairs...", ncols=80)
        sample_num = 0
        for cs in tq:
            cs_len = len(cs)
            if (cs_len >= 2):
                for func_index in range(cs_len):
                    rand = self.random.random()
                    # if rand < 0.5:
                    #     continue
                    anchor_func_id = cs[func_index]
                    if anchor_func_id not in self.subgraphs:
                        continue
                    flag, p_id = self.positive_sample(cs_len, cs, anchor_func_id)
                    if flag == 1:
                        n_id = self.negative_sample_random(self.func_classes, anchor_func_id, cs,
                                                           rand=self.negative_rand)
                        self.a_fids.append(anchor_func_id)
                        self.p_fids.append(p_id)
                        self.n_fids.append(n_id)
                        sample_num += 1
                        if sample_num >= self.sample_max:
                            return

    def __getitem__(self, item):
        a_id = self.a_fids[item]
        p_id = self.p_fids[item]
        n_id = self.n_fids[item]
        a_g = self.subgraphs[a_id]
        a_n = self.fid2name[str(a_id)]
        p_g = self.subgraphs[p_id]
        p_n = self.fid2name[str(p_id)]
        n_g = self.subgraphs[n_id]
        n_n = self.fid2name[str(n_id)]
        return a_g, p_g, n_g, [a_n, p_n, n_n]
        pass

    def get_func_nums(self):
        return len(self.a_fids)

    def __len__(self):
        return len(self.a_fids)
        pass
