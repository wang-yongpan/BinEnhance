from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os
import json
from whitening_transformation import compute_kernel_bias, transform_and_normalize
import numpy as np

def sentence_bert(argparse):
    model_path = argparse.model_path
    dim = argparse.dimension
    output_dir = argparse.output_dir
    save_path = os.path.join(output_dir, "strs_embeddings_" + str(dim) + ".json")
    data_path = argparse.input_dir
    kernel_path = os.path.join(output_dir, "kernel_once_" + str(dim) + ".npy") # this can use the same kernal and bias with WT
    bias_path = os.path.join(output_dir, "bias_once_" + str(dim) + ".npy")
    strs_embeddings = {}
    model = SentenceTransformer(model_path)
    model.to(0)
    for file in os.listdir(data_path):
        filepath = os.path.join(data_path, file)
        with open(filepath, "r") as f:
            data = json.load(f)
        for fname, ts in data.items():
            for t in ts:
                if t not in strs_embeddings:
                    # delete the first 9 characters
                    strs_embeddings[t[9:]] = 0
    str_texts = list(strs_embeddings.keys())
    str_embedding = model.encode(str_texts)
    if not os.path.exists(kernel_path):
        kernel, bias = compute_kernel_bias(str_embedding, dim)
    else:
        kernel = np.load(kernel_path)
        bias = np.load(bias_path)
    str_embedding = transform_and_normalize(str_embedding, kernel=kernel, bias=bias).tolist()
    np.save(kernel_path, kernel)
    np.save(bias_path, bias)
    strs_embeddings = {}
    for i in range(len(str_texts)):
        strs_embeddings[str_texts[i]] = str_embedding[i]
    with open(save_path, "w") as f:
        json.dump(strs_embeddings, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True, help="the path of sentence bert model")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="the directory of input files")
    parser.add_argument("--dimension", "-d", type=str, required=True, help="the output dimension of sentence embeddings")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="the directory of output files")
    
    args = parser.parse_args()
    sentence_bert(args)