#%%
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import dotenv

config = dotenv.dotenv_values(".env")
model_name = config["MODEL_NAME"]
g_file_path = config["G_FILE_PATH"]

### load model ###
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.float32,
                                             device_map="auto")


### load unembdding vectors ###
gamma = model.get_output_embeddings().weight.detach()
W, d = gamma.shape
gamma_bar = torch.mean(gamma, dim = 0)
centered_gamma = gamma - gamma_bar

### compute Cov(gamma) and tranform gamma to g ###
Cov_gamma = centered_gamma.T @ centered_gamma / W
eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
g = centered_gamma @ inv_sqrt_Cov_gamma


## Use this PATH to load g in the notebooks
torch.save(g, g_file_path)
