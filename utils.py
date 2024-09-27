import os 
import networkx as nx
import numpy as np
from tqdm import tqdm

def get_func_classes(func_embeddings, hsg_path="", MIN_BLOCKS=0):
    graphs = {}
    if hsg_path != "":
        for file in os.listdir(hsg_path):
            filename = file.split(".pkl")[0]
            filepath = os.path.join(hsg_path, file)
            graph = nx.read_gpickle(filepath)
            graphs[filename] = graph
    classes = []
    func2id = {}
    cop_func = []
    for funcN, embed in func_embeddings.items():
        func = funcN.split("_")[0] + "|||" + funcN.split("_")[5] + "|||" + funcN.split("|||")[-1]
        if MIN_BLOCKS != 0 and hsg_path != "":
            try:
                if int(graphs[funcN.replace("|||" + funcN.split("|||")[-1], "")].nodes[funcN.split("|||")[-1]]["block_num"]) < MIN_BLOCKS:
                    continue
            except:
                continue
        cop_func.append(funcN)
        if func not in func2id:
            func2id[func] = len(func2id)
    print("cop func:" + str(len(cop_func)) + "/" + str(len(func_embeddings)))
    for func, id in func2id.items():
        classes.append([])
    for funcN in cop_func:
        func = funcN.split("_")[0] + "|||" + funcN.split("_")[5] + "|||" + funcN.split("|||")[-1]
        classes[func2id[func]].append(funcN)
    new_class = []
    for cl in classes:
        if len(cl) >= 2:
            np.random.shuffle(cl)
            new_class.append(cl)
    np.random.shuffle(new_class)
    return new_class
    pass

def generate_eval_data(classes, poolsize=10000, yarchs="", yoptc="", tarchs="", toptc=""):
    sample_num = poolsize
    max_pos = 10 if poolsize > 20 else poolsize / 2
    max_sample = 5000
    eval_datas = []
    np.random.shuffle(classes)
    classes_eval = classes
    sl_data = {}
    for class_eval in tqdm(classes_eval, ncols=85):
        cl_index = classes_eval.index(class_eval)
        if yarchs != "":
            if yarchs not in class_eval[0] or yoptc not in class_eval[0]:
                continue
        eval_data = []
        pos_num = 0
        for ce in class_eval[1:len(class_eval)]:
            if tarchs != "":
                if tarchs not in ce or toptc not in ce:
                    continue
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
                # s_d = str(class_eval[0]) + str(classes_eval[index1][index2]) + str(0)
                s_d = (class_eval[0], classes_eval[index1][index2], 0)
                if str(s_d) not in sl_data and tarchs in classes_eval[index1][index2] and toptc in classes_eval[index1][index2]:
                    sl_data[str(s_d)] = 0
                    flag = 1
                    break
            if flag != 1:
                continue
            else:
                eval_data.append((class_eval[0], classes_eval[index1][index2], 0))
        eval_datas.append(eval_data)
        if len(eval_datas) >= max_sample:
            break
    return eval_datas