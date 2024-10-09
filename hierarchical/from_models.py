import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def get_gamma(MODEL_NAME, device):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 torch_dtype=torch.float32,
                                                 device_map=device)

    gamma = model.get_output_embeddings().weight.detach()
    
    return gamma

def get_g(MODEL_NAME, device):
    gamma = get_gamma(MODEL_NAME, device)
    W, d = gamma.shape
    gamma_bar = torch.mean(gamma, dim = 0)
    centered_gamma = gamma - gamma_bar

    Cov_gamma = centered_gamma.T @ centered_gamma / W
    eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
    inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
    sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    g = centered_gamma @ inv_sqrt_Cov_gamma

    return g, inv_sqrt_Cov_gamma, sqrt_Cov_gamma

def get_vocab(MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_dict = tokenizer.get_vocab()
    vocab_list = [None] * (max(vocab_dict.values()) + 1)
    for word, index in vocab_dict.items():
        vocab_list[index] = word

    return vocab_dict, vocab_list


def compute_lambdas(texts, MODEL_NAME, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 torch_dtype=torch.float32,
                                                 device_map="auto")

    assert tokenizer.padding_side == "left", "The tokenizer padding side must be 'left'."

    
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        lambdas = outputs.hidden_states[-1][:, -1, :]

    return lambdas
