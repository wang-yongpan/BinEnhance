import argparse
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
from tqdm import tqdm
import os


def set_random_seed(seed_value=85422):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

def get_func_classes(func_embeddings, MIN_BLOCKS):
    classes = []
    func2id = {}
    cop_func = []
    for funcN, fns in func_embeddings.items():
        func = funcN.split("|||")[0] + "|||" + funcN.split("|||")[1] + "|||" + funcN.split("|||")[-1]
        if int(fns["block_num"]) < MIN_BLOCKS:
            continue
        cop_func.append(funcN)
        if func not in func2id:
            func2id[func] = len(func2id)
    for func, id in func2id.items():
        classes.append([])
    for funcN in cop_func:
        func = funcN.split("|||")[0] + "|||" + funcN.split("|||")[1] + "|||" + funcN.split("|||")[-1]
        classes[func2id[func]].append(funcN)
    new_class = []
    for cl in classes:
        if len(cl) >= 2:
            new_class.append(cl)
    return new_class
    pass

def generate_eval_data(classes, poolsize=10000):
    sample_num = poolsize
    max_pos = 10
    eval_datas = []
    np.random.shuffle(classes)
    classes_eval = classes
    sl_data = {}
    for class_eval in tqdm(classes_eval):
        cl_index = classes_eval.index(class_eval)
        eval_data = []
        pos_num = 0
        for ce in class_eval[1:len(class_eval)]:
            pos_num += 1
            eval_data.append((class_eval[0], ce, 1))
            if pos_num >= max_pos:
                break
        if pos_num == 0:
            continue
        while len(eval_data) < sample_num:
            flag = 0
            for i in range(10):
                index1 = np.random.randint(0, len(classes_eval))
                if index1 != cl_index:
                    flag = 1
                    break
            if flag == 0:
                continue
            flag = 0
            for i in range(10):
                index2 = np.random.randint(0, len(classes_eval[index1]))
                s_d = (class_eval[0], classes_eval[index1][index2], 0)
                if str(s_d) not in sl_data:
                    sl_data[str(s_d)] = 0
                    flag = 1
                    break
            if flag != 1:
                continue
            else:
                eval_data.append((class_eval[0], classes_eval[index1][index2], 0))
        eval_datas.append(eval_data)
    return eval_datas
    pass

def eval_by_map(Gs_embed, eval_datas):
    lens = []
    sims = []
    labels = []
    for eds in tqdm(eval_datas):
        leng = 0
        label = []
        feat_s = []
        feat_e = []
        feat_s.append(Gs_embed[eds[0][0]]["embedding"])
        for ed in eds:
            feat_e.append(Gs_embed[ed[1]]["embedding"])
            label.append(str(ed[0]) + "||||" + str(ed[1]) + "||||" + str(ed[2]))
            leng += ed[2]
        sims.append(cosine_similarity(np.array(feat_s), np.array(feat_e)).reshape(-1))
        lens.append(leng)
        labels.append(label)
    print("it is calculating map score ...")
    ans = cal_map_metrics(sims, labels, lens)
    return ans
    pass

def cal_map_metrics(sims, labels, lens):
    q = len(sims)
    map = 0.0
    aps = []
    for i in tqdm(range(q)):
        ap = 0.0
        cos_num = 0.0
        sim = sims[i]
        label = labels[i]
        sorted_list = sorted([(si, li) for si, li in zip(sim, label)], reverse=True)
        label = [li for _, li in sorted_list]
        for j in range(10):
            if int(label[j].split("||||")[-1]) == 1:
                cos_num += 1.0
                ap = ap + cos_num / float(j + 1)
        ap = ap / lens[i]
        aps.append(ap)
        map += ap
    ans = map / float(q)
    return ans

def map_eval(block_size=5, embed_path="", poolsize=10000):
    set_random_seed()
    eval_func_embeddings_path = embed_path
    with open(eval_func_embeddings_path, "r") as f:
        func_embeddings = json.load(f)
    print(eval_func_embeddings_path)
    eval_data_path = embed_path.replace(embed_path.split("/")[-1], "") + "eval_data_" + str(block_size) + "_" + str(poolsize) + ".json"
    print("it is generating eval data ...")
    if not os.path.exists(eval_data_path):
        classes = get_func_classes(func_embeddings, block_size)
        eval_datas = generate_eval_data(classes, poolsize=poolsize)
        e = {}
        e["d"] = eval_datas
        with open(eval_data_path, "w") as f:
            json.dump(e, f)
    else:
        with open(eval_data_path, "r") as f:
            eval_datas = json.load(f)["d"]
    print("generate eval data completed ...")
    ans = eval_by_map(func_embeddings, eval_datas)
    return ans

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="OriginMethods",
                        help="the path of function embedding(including initial and BinEnhance).")
    args = parser.parse_args()
    print(args)
    data_path = args.data_path
    for file in os.listdir(data_path):
        base_path = os.path.join(data_path, file)
        old_path = os.path.join(base_path, "init_func_embeddings_new.json")
        new_path = os.path.join(base_path, "BinEnhance_func_embeddings_new.json")
        ps = [100, 1000, 10000]
        bs = [0, 5, 10]
        for b in bs:
            for p in ps:
                old_ans = map_eval(block_size=b, embed_path=old_path, poolsize=p)
                new_ans = map_eval(block_size=b, embed_path=new_path, poolsize=p)
                print("[Model-" + str(file) + " Block size-" + str(b) + " Pool size-" + str(p) + " Original]:" + str(old_ans))
                print("[Model-" + str(file) + " Block size-" + str(b) + " Pool size-" + str(p) + " Original]:" + str(new_ans))
    pass
