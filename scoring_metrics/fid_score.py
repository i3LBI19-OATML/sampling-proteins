"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
import subprocess
import pandas as pd
from .util import add_metric, identify_mutation
from pgen.utils import parse_fasta

import numpy as np
import torch
import tempfile
from scipy import linalg

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

def get_ESM1v_predictions(targets_fasta, device = 'cuda:0'):
    pred_arr = []
    if device=='cuda:0':
        torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory() as output_dir:
        outfile = output_dir + "/esm_results.tsv"
        try:
            proc = subprocess.run(['python', os.path.join(os.path.dirname(os.path.realpath(__file__)), "protein_gibbs_sampler/src/pgen/likelihood_esm.py"), "-i", targets_fasta, "-o", outfile, "--model", "esm1v", "--masking_off", "--score_name", "score", "--device", "gpu", "--use_repr"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            # print(proc.stdout)
            # print(proc.stderr)
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode('utf-8'))
            print(e.stderr.decode('utf-8'))
            raise e
            
        df = pd.read_table(outfile)
        print("done")
        for i, row in df.iterrows():
            p = row['score']
            pred_arr.append(p)
    # print(pred_arr)
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # sigma1 = np.nan_to_num(sigma1)
    # sigma2 = np.nan_to_num(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, orig_seq, device='cuda:0', num_workers=8):
    # act = get_activations(files, model, batch_size, dims, device, num_workers)
    # act = get_ESM1v_predictions(files, orig_seq, device)
    act = get_ESM1v_predictions(files, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, orig_seq, device, num_workers=1):
    if type(path) is list and len(path) > 0:
        m = np.mean(path, axis=0)
        s = np.cov(path, rowvar=False)
    else:
        m, s = calculate_activation_statistics(path, orig_seq, device, num_workers)

    return m, s


def calculate_fid_given_paths(target_files, reference_files, name, orig_seq, device='cuda:0', num_workers=8):
    """Calculates the FID of two paths"""
    # print(f'target_files:{target_files}, reference_files:{reference_files}')
    # Target statistics
    if type(target_files) is list and len(target_files) > 0:
        print(f'Using precomputed target statistics for {name}')
        m1, s1 = compute_statistics_of_path(target_files, device, num_workers)
    else:
        print(f'Calculating target statistics for {target_files} (Source: {target_files})')
        m1, s1 = calculate_activation_statistics(target_files, orig_seq, device, num_workers)

    # Reference statistics
    reference_name = pathlib.Path(name).stem
    cache_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"tmp/FID_reference_cache/{reference_name}.npy")
    if os.path.exists(cache_file):
        print(f"Loading reference statistics from {cache_file}")
        m2, s2 = np.load(cache_file, allow_pickle=True)
    else:
        print(f"Calculating reference statistics for {reference_name} (Source: {reference_files})")
        m2, s2 = compute_statistics_of_path(reference_files, orig_seq, device, num_workers)
        
        save_dir = os.path.dirname(cache_file)
        os.makedirs(save_dir, exist_ok=True)
        np.save(cache_file, (m2, s2))
        print(f"Saved reference statistics to {cache_file}")
    # Calculate FID
    fid_value = calculate_frechet_distance(m1, s1, m2, s2) * 1000

    # result = {"fid": fid_value}
    # df = pd.DataFrame.from_dict(result, orient="index")
    # df.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),"fid.csv"))
    # return 
    # for _, row in df.iterrows():
    #     results["FID"] = row["fid"]
    # del df
    # print("FID: ", fid_value)
    return fid_value
