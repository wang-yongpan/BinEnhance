import numpy as np
import os
import json


def compute_kernel_bias(vecs, n_components=256):
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu

def transform_and_normalize(vecs, kernel=None, bias=None):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="the directory of function embedding files of baselines")
    parser.add_argument("--dimension", "-d", type=str, required=True, help="the output dimension of function embeddings")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="the directory of output files")
    
    args = parser.parse_args()
    dim = args.dimension
    path = args.input_dir
    output_path = args.output_dir
    kernel_path = os.path.join(output_path, "kernel_once_" + str(dim) + ".npy")
    bias_path = os.path.join(output_path, "bias_once_" + str(dim) + ".npy")
    
    with open(path, "r") as f:
        f_embeddings = json.load(f)
    s = []
    vs = []
    for k, v in f_embeddings.items():
        s.append(k)
        vs.append(v)
    vs = np.array(vs)
    if not os.path.exists(kernel_path):
        kernel, bias = compute_kernel_bias(vs, dim)
        np.save(kernel_path, kernel)
        np.save(bias_path, bias)
    else:
        kernel = np.load(kernel_path)
        bias = np.load(bias_path)
    vs = transform_and_normalize(vs, kernel=kernel, bias=bias).tolist()
    fs = {}
    for ss in s:
        fs[ss] = vs[s.index(ss)]
    with open(os.path.join(output_path, "function_embeddings_" + str(dim) + ".json"), "w") as f:
        json.dump(fs, f)