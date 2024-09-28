import time
from transformers import AutoModel, AutoTokenizer
import os

import time
from transformers import AutoModel, AutoTokenizer
import os

from pdb import set_trace as st

def upload_model_and_tokenizer(model_path, repo_name, hf_token):
    """
    Uploads a model and tokenizer to the Hugging Face Hub under the specified organization.
    """
    print(f'-> Uploading mode at path {model_path} to {repo_name=} with your hf token.')
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf') # hardcoded cuz Brando forgot to save tokenizer

    # Upload model and tokenizer to the Hugging Face Hub as public
    model.push_to_hub(repo_name, use_auth_token=hf_token, private=False)
    tokenizer.push_to_hub(repo_name, use_auth_token=hf_token, private=False)
    print(f"Model and tokenizer successfully uploaded to: https://huggingface.co/{repo_name}")

def test_download_model(repo_name):
    """
    Downloads the model and tokenizer from the Hugging Face Hub to verify successful upload.
    """
    print("Testing download of the model...")
    model = AutoModel.from_pretrained(repo_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    print(f'{model=}')
    print(f'{tokenizer=}')
    print("Model and tokenizer downloaded successfully.")

def main():
    # List of checkpoint paths to upload
    checkpoint_paths = [
        # "/lfs/ampere9/0/brando9/data/results_2024-m02-d04-t02h_02m_02s",  # Checkpoint 1
        # "/lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_50m_22s",  # Checkpoint 2
        # "/lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_47m_30s",  # Checkpoint 3
        # "/lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_45m_48s",  # Checkpoint 4
        # "/lfs/ampere9/0/brando9/data/results_2024-m02-d03-t16h_34m_01s",  # Checkpoint 5
        # "/lfs/ampere9/0/brando9/data/results_2024-m01-d29-t16h_01m_30s",  # Checkpoint 6
        # "/lfs/ampere9/0/brando9/data/results_2024-m01-d29-t16h_00m_55s"   # Checkpoint 7
    ]

    # Organization and Hugging Face access token
    organization = "UDACA"  # The organization name on Hugging Face
    # hf_token = open(os.path.expanduser('~/keys/brandos_hf_token.txt')).read().strip()
    hf_token = open(os.path.expanduser('~/keys/token_made_for_div_coeff_llama2_pushes.txt')).read().strip()
    print(f'{hf_token=}')

    # Upload each checkpoint
    for i, ckpt_path in enumerate(checkpoint_paths):
        repo_name = f"{organization}/checkpoint-{i+1}"  # Creates a unique repo name for each checkpoint
        print(f"Uploading checkpoint {i+1} from {ckpt_path} to {repo_name}...")
        upload_model_and_tokenizer(ckpt_path, repo_name, hf_token)
        
        # Optionally, test downloading the model and tokenizer after upload
        # test_download_model(repo_name)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Done! Total time taken: {end_time - start_time:.2f} seconds")