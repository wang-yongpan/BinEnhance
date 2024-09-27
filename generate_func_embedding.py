
import argparse
import os
import json

import torch
from tqdm import tqdm
from datasets import eesg_datasets
from predict import load_graph_data_neg
from predict import get_subgraph

def cli(fisname, funcDim):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--model-save", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--embedding_base", type=str, required=True, help="embedding base path of baselines")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--negative-slope', type=float, default=0.2)
    parser.add_argument('--name', type=str, default='dataset2')
    parser.add_argument('--modelname', type=str, default='BinEnhance')
    parser.add_argument("--funcDim", type=int, default=funcDim)
    parser.add_argument("--max-edge-num", type=int, default=500)
    parser.add_argument("--max-node-num", type=int, default=999999) 
    parser.add_argument("--sample-max", type=int, default=100000) 
    parser.add_argument("--negative-rand", type=float, default=0.)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--fis", type=str, default=fisname)
    args = parser.parse_args()
    print(args)

    model_name = 'r-gcn-' + str(args.modelname) + '-' + str(args.name) + '-' + str(args.max_edge_num) + '-' + str(
        args.max_node_num) + '-' + str(args.negative_rand) + '-' + str(args.lr) + '-' + str(
        args.num_layers) + '-' + str(args.sample_max) + "-" + str(args.batch_size)
    save_base = os.path.join(args.model_save, os.path.join(str(args.fis), model_name))
    model = torch.load(model).to(args.gpu)

    # strip = 1
    base_path = args.base_path
    data_base = os.path.join(base_path, "bidire_EESG")
   
    embedding_base = os.path.join(args.embedding_base, str(args.fis))
    save_res = args.output_dir
    if not os.path.exists(save_res):
        os.makedirs(save_res)
    func_embeddings_path = os.path.join(embedding_base, "test_function_embeddings_" + str(args.funcDim) + ".json")
    with open(func_embeddings_path, "r") as f:
        func_embeddings = json.load(f)
    strs_embeddings_path = ""
    
    strs_embeddings_path = os.path.join(os.path.join(base_path, "Strings"), "test_strs_embeddings_" + str(args.funcDim) + ".json")
    with open(strs_embeddings_path, "r") as f:
        strs_embeddings = json.load(f)
    data_path = os.path.join(data_base, "test")
    
    dataset = eesg_datasets(data_path, funcs_embeddings=func_embeddings, strs_embeddings=strs_embeddings,
                                    depth=args.num_layers, data_base=data_base, mode="test", args=args)
    save_res += "/test_function_embeddings.json"
    graphs = load_graph_data_neg(data_path, dataset.rels)
    subgraphs_embed = {}
    for fn, embeds in tqdm(func_embeddings.items(), desc="it is init subgraphs..."):
        fn_type = fn.split("|||")[0]
        if fn_type not in graphs:
            continue
        graph = graphs[fn_type]
        sub = get_subgraph(graph, fn, func_embeddings, strs_embeddings, args.num_layers, args.max_edge_num, relss_type=dataset.rels)
        if sub is None or sub.num_nodes() == 1 or sub.num_edges() == 1:
            continue
        embed = model.forward_once(sub).cpu().detach().numpy().tolist()
        subgraphs_embed[fn] = embed
    with open(save_res, "w") as f:
        json.dump(subgraphs_embed, f)

if __name__ == '__main__':
    cli()
