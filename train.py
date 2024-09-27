import argparse
import json
import os
from datetime import datetime
import time
import dgl
import tqdm
from dgl.dataloading import GraphDataLoader

from model import RGCN_Model
from datasets import eesg_datasets
import torch
import numpy as np
from sklearn.metrics import auc, roc_curve
import random
from predict import combine_eval



def set_random_seed(seed_value=1234):
    torch.manual_seed(seed_value)  # cpu  vars
    np.random.seed(seed_value)  # cpu vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

def collate_fn_mini(batch):
    #  batch is a list of tuples, each tuple is the result of dataset._getitem__
    batch = list(zip(*batch))
    a_graphs = dgl.batch(batch[0])
    b_graphs = dgl.batch(batch[1])
    c_graphs = dgl.batch(batch[2])
    funcNames = batch[3]
    del batch
    return a_graphs, b_graphs, c_graphs, funcNames

def contra_loss_show(net, dataLoader, DEVICE, bs, f_strings="", f_gv="", f_ef=""):
    loss_val = []
    tot_cos = []
    tot_truth = []
    tq = tqdm.tqdm(dataLoader, ncols=80)
    for batch_id, batch in enumerate(tq, 1):
        try:
           
            cos_p, cos_n, loss = net.forward(batch, bs, f_strings, f_gv,
                                                               f_ef) 

        except:
            print("valid or test process exist single...")
            continue
        cos_p = list(cos_p.cpu().detach().numpy())
        cos_n = list(cos_n.cpu().detach().numpy())
        tot_cos += cos_p
        tot_truth += [1] * len(cos_p)
        tot_cos += cos_n
        tot_truth += [-1] * len(cos_n)
        loss_val.append(loss.item())
        tq.set_description("Eval:[" + str(loss.item()) + "]")
    cos = np.array(tot_cos)
    truth = np.array(tot_truth)
    fpr, tpr, thres = roc_curve(truth, (1 + cos) / 2)
    model_auc = auc(fpr, tpr)
    return loss_val, model_auc, tpr


def run(args):
    # set the save path
    set_random_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    torch.cuda.set_device(args.gpu)
    base_path = args.base_path
    model_save = args.model_save
    data_base = os.path.join(base_path, "EESG")
    embedding_base = os.path.join(base_path, os.path.join("embeddings", str(args.fis)))
    model_name = 'r-gcn-' + str(args.modelname) + '-' + str(args.name) + '-' + str(args.max_edge_num) + '-' + str(
        args.max_node_num) + '-' + str(args.negative_rand) + '-' + str(args.lr) + '-' + str(
        args.num_layers) + '-' + str(args.sample_max) + "-" + str(args.batch_size)
    save_base = os.path.join(model_save, os.path.join(str(args.fis), model_name))
    train_func_embeddings_path = os.path.join(embedding_base,
                                              "train_function_embeddings_" + str(args.funcDim) + ".json")

    with open(train_func_embeddings_path, "r") as f:
        train_func_embeddings = json.load(f)
    test_func_embeddings_path = os.path.join(embedding_base, "test_function_embeddings_" + str(args.funcDim) + ".json")
    with open(test_func_embeddings_path, "r") as f:
        test_func_embeddings = json.load(f)
    valid_func_embeddings_path = os.path.join(embedding_base,
                                              "valid_function_embeddings_" + str(args.funcDim) + ".json")
    with open(valid_func_embeddings_path, "r") as f:
        valid_func_embeddings = json.load(f)

    test_strs_embeddings_path = ""

    
    train_strs_embeddings_path = os.path.join(os.path.join(base_path, "Strings"),
                                                "train_valid_strs_embeddings_" + str(args.funcDim) + ".json")
    with open(train_strs_embeddings_path, "r") as f:
        train_strs_embeddings = json.load(f)
    test_strs_embeddings_path = os.path.join(os.path.join(base_path, "Strings"),
                                                "test_strs_embeddings_" + str(args.funcDim) + ".json")
    with open(test_strs_embeddings_path, "r") as f:
        test_strs_embeddings = json.load(f)

    if not os.path.exists(save_base):
        os.makedirs(save_base)

    train_data_path = os.path.join(data_base, "train")
    valid_data_path = os.path.join(data_base, "valid")
    test_data_path = os.path.join(data_base, "test")


    f_strings_path = os.path.join(data_base, "all_strings.json")
    with open(f_strings_path, "r") as f:
        f_strings = json.load(f)
    f_ef_path = os.path.join(data_base, "all_external_funcs.json")
    with open(f_ef_path, "r") as f:
        f_ef = json.load(f)
    f_gv_path = os.path.join(data_base, "all_global_vars.json")
    with open(f_gv_path, "r") as f:
        f_gv = json.load(f)

    bs = args.batch_size
    train_dataset = eesg_datasets(train_data_path, funcs_embeddings=train_func_embeddings,
                                     strs_embeddings=train_strs_embeddings, depth=args.num_layers, data_base=data_base,
                                     mode="train", args=args)
    valid_dataset = eesg_datasets(valid_data_path, funcs_embeddings=valid_func_embeddings,
                                     strs_embeddings=train_strs_embeddings, depth=args.num_layers, data_base=data_base,
                                     mode="valid", args=args)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn_mini)
    test_dataset = eesg_datasets(test_data_path, funcs_embeddings=test_func_embeddings,
                                    strs_embeddings=test_strs_embeddings, depth=args.num_layers, data_base=data_base,
                                    mode="test", args=args)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn_mini)

    print("""----Data statistics------'
              #Train samples %d
              #Valid samples %d
              #Test samples %d""" %
          (train_dataset.get_func_nums(), valid_dataset.get_func_nums(),
           test_dataset.get_func_nums()))
    print("mode_save:" + str(save_base))
    print("rel_type:" + str(train_dataset.rels))
    # create model

   
    rels = train_dataset.rels
    funcDim = args.funcDim
    model = RGCN_Model(args, funcDim, funcDim, funcDim, len(rels))
    model.to(args.gpu)
    print(model)

    # use optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=args.weight_decay)

    SAVE_FREQ = 5
    best_loss = 99999
    gradsss = []
    epoch = args.epoch
    patience = 5
    for i in range(epoch):
        loss_val = []
        tot_cos = []
        tot_truth = []
        time_start = time.time()
        model.train()
        p_n_gap = []
        train_dataset.shuffle()
        train_dataloader = GraphDataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn_mini)
        tq = tqdm.tqdm(train_dataloader, position=0, ncols=80)
        for batch in tq:
            
            cos_p, cos_n, loss = model.forward(batch, bs, f_strings, f_gv, f_ef) 
            
                
            cos_p = cos_p.cpu().detach().numpy()
            cos_n = cos_n.cpu().detach().numpy()
            p_n_gap.append(np.mean(cos_p - cos_n))
            cos_p = list(cos_p)
            cos_n = list(cos_n)
            tot_cos += cos_p
            tot_truth += [1] * len(cos_p)
            tot_cos += cos_n
            tot_truth += [-1] * len(cos_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.cpu().detach().numpy().item()
            loss_val.append(loss)
            tq.set_description("Train:EPOCH" + str(i) + "[" + str(loss) + "]") 
        cos = np.array(tot_cos)
        truth = np.array(tot_truth)
        fpr, tpr, thres = roc_curve(truth, (1 + cos) / 2)
        model_auc = auc(fpr, tpr)
        print('Epoch: [%d]\tloss:%.4f\tp_n_gap:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
              (i, np.mean(loss_val), np.mean(p_n_gap), model_auc, datetime.now(), time.time() - time_start))
        model.eval()
        with torch.no_grad():
            time_start = time.time()
            
            loss_val, model_auc, tpr = contra_loss_show(model, valid_dataloader, args.gpu, bs, f_strings, f_gv,
                                                            f_ef) 
            
            print('Valid: [%d]\tloss:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
                  (i, np.mean(loss_val), model_auc, datetime.now(), time.time() - time_start))

            time_start = time.time()
            
            loss_test, test_model_auc, tpr = contra_loss_show(model, test_dataloader, args.gpu, bs, f_strings, f_gv,
                                                                  f_ef) 
            print("#" * 70)
            print('Test: [%d]\tloss:%.4f\tauc:%.4f\t@%s\ttime lapsed:\t%.2f s' %
                  (i, np.mean(loss_test), test_model_auc, datetime.now(), time.time() - time_start))
            print("#" * 70)

            patience -= 1
            if np.mean(loss_val) < best_loss:
                torch.save(model, save_base + "/BinEnhance-best.pt")
                best_loss = np.mean(loss_val)
                patience = 5

        if i % SAVE_FREQ == 0:
            torch.save(model, save_base + '/BinEnhance-' + str(i + 1) + ".pt")
        if patience <= 0:
            break
    
    save_res = os.path.join(base_path, os.path.join("results", args.fis))
    eval_p = os.path.join(base_path, "Eval_datas")
    if not os.path.exists(save_res):
        os.makedirs(save_res)

    save_res += model_name.split("/")[-1] + "_test_func_embeddings.json"
    ans = combine_eval(save_base, f_strings, f_gv, f_ef, hsg_data_path=test_data_path,
                    func_embedding_path=test_func_embeddings_path, relss_type=test_dataset.rels, args=args,
                    eval_type=0, str_embedding_path=test_strs_embeddings_path, model=args.fis, eval_p=eval_p, savepaths=save_res)
    print(ans)
    return ans


#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--base-path", type=str, required=True, help="the path of all features")
    parser.add_argument("--model-save", type=str, required=True, help="the path to save the model")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--negative-slope', type=float, default=0.2)
    parser.add_argument('--name', type=str, required=True, help="the dataset name")
    parser.add_argument('--modelname', type=str, default='BinEnhance')
    parser.add_argument("--funcDim", type=int, default=128)
    parser.add_argument("--max-edge-num", type=int, default=500)
    parser.add_argument("--max-node-num", type=int, default=999999) 
    parser.add_argument("--sample-max", type=int, default=100000) 
    parser.add_argument("--negative-rand", type=float, default=0.)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--fis", type=str, required=True, help="the baseline name")
    args = parser.parse_args()
    print(args)
    run(args)
    pass