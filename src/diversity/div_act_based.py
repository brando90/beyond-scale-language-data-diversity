"""
Problem:
I want to compute the CCA distance between two LLMs. 
The shape of any (intermediate) layer is [B, T, D], where B is batch size, T is sequence length, and D is the dimensionality of the layer.
The problem is that the CCA distance is only defined for two matrices of shape [B, D], where B is batch size and D is the dimensionality of the layer.
What is the right way to compute the CCA distance between two LLMs?

CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -nr | head -n 1 | awk -F ', ' '{print $1}')
echo CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES

ref: acts debate/conv: https://chat.openai.com/c/9aae0b31-689e-415c-ba40-73a790bb2e0d
ref: general acts code: https://chat.openai.com/g/g-KV0CvoH8Y-python-excellent-comments-doc-strings-types/c/d50783d2-f958-49d6-a729-2bc6cf28deb7
"""
from datasets import load_dataset
from transformers import GPT2Model, GPT2Tokenizer
import torch
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from anatome.similarity import pwcca_distance_choose_best_layer_matrix, svcca_distance, linear_cka_distance, orthogonal_procrustes_distance, temporal_cca

    # _default_backends = {'pwcca': partial(pwcca_distance_choose_best_layer_matrix, backend='svd', epsilon=1e-10),
    #                      'svcca': partial(svcca_distance, accept_rate=0.99, backend='svd'),
    #                      'lincka': partial(linear_cka_distance, reduce_bias=False),
    #                      "opd": orthogonal_procrustes_distance}

# Function to set all seeds for reproducibility
def set_random_seeds(seed_value=42):
    """
    This function sets the seed for randomness for reproducible results.
    
    Args:
    - seed_value (int): The value of the seed to be used for all libraries.
    """
    random.seed(seed_value)  # Python's built-in random library
    np.random.seed(seed_value)  # NumPy library
    torch.manual_seed(seed_value)  # PyTorch library

    # If you are using CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # If running on the GPU, also set the seed there
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def generate_same_token_sequence(token_value: int, 
                                sequence_length: int = 50, 
                                batch_size: int = 600, 
                                device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> torch.Tensor:
    """
    Generates a batch of token sequences where each token in the sequence is the same.

    Args:
    - token_value (int): The token value to be repeated in the sequence.
    - sequence_length (int, optional): The length of each token sequence. Defaults to 50.
    - batch_size (int, optional): The number of sequences in the batch. Defaults to 3.
    - device (torch.device): The device to perform computations on. Defaults to GPU if available, else CPU.

    Returns:
    - torch.Tensor: A tensor containing the batch of identical token sequences on the specified device.
    """
    # Create a single sequence of the same token
    single_sequence = [token_value] * sequence_length

    # Create a batch of identical sequences
    batch_sequences = [single_sequence for _ in range(batch_size)]

    # Convert the batch to a PyTorch tensor and move to the specified device
    token_tensor = torch.tensor(batch_sequences, dtype=torch.long).to(device)
    
    return token_tensor

def generate_semi_random_tokens_limited_vocab(tokenizer: GPT2Tokenizer, 
                           sequence_length: int = 50, 
                           batch_size: int = 600, 
                           device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                           percentange_vocab: float = 0.1,
                           ) -> torch.Tensor:
    """
    Generates a batch of semi-random token sequences compatible with GPT-2's tokenizer and moves them to the specified device.
    The randomness is reduced by limiting the selection to a subset of the tokenizer's vocabulary.

    Args:
    - tokenizer (GPT2Tokenizer): The tokenizer for GPT-2.
    - sequence_length (int, optional): The length of each random token sequence. Defaults to 50.
    - batch_size (int, optional): The number of sequences in the batch. Defaults to 1.
    - device (torch.device): The device to perform computations on.

    Returns:
    - torch.Tensor: A tensor containing the batch of semi-random token sequences on the specified device.
    """
    # Define a subset range of the tokenizer's vocabulary
    vocab_subset_range = int(tokenizer.vocab_size * percentange_vocab)  # For example, 10% of the total vocab size
    assert vocab_subset_range != 0, "The vocabulary subset range is 0!"

    # Generate batch of token sequences with tokens randomly selected from the subset range
    batch_random_tokens = [[random.randint(0, vocab_subset_range - 1) for _ in range(sequence_length)] for _ in range(batch_size)]
    
    token_tensor = torch.tensor(batch_random_tokens, dtype=torch.long).to(device)
    return token_tensor

def generate_random_tokens(tokenizer: GPT2Tokenizer, 
                           sequence_length: int = 50, 
                           batch_size: int = 600, 
                           device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                           ) -> torch.Tensor:
    """
    Generates a batch of random token sequences compatible with GPT-2's tokenizer and moves them to the specified device.

    Args:
    - tokenizer (GPT2Tokenizer): The tokenizer for GPT-2.
    - sequence_length (int, optional): The length of each random token sequence. Defaults to 50.
    - batch_size (int, optional): The number of sequences in the batch. Defaults to 1.
    - device (torch.device): The device to perform computations on.

    Returns:
    - torch.Tensor: A tensor containing the batch of random token sequences on the specified device.
    """
    batch_random_tokens = [[random.randint(0, tokenizer.vocab_size - 1) for _ in range(sequence_length)] for _ in range(batch_size)]
    token_tensor = torch.tensor(batch_random_tokens, dtype=torch.long).to(device)
    return token_tensor

def main():
    """
    Main function to load the GPT-2 model, generate random tokens, and compute activations.
    """
    # Determine if CUDA (GPU support) is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GPT-2 model and tokenizer
    model = GPT2Model.from_pretrained('gpt2')
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    token_value1: int = tokenizer.encode('the')[0]
    token_value2: int = tokenizer.encode('at')[0]
    print(f'{tokenizer.model_max_length=}')

    # Generate a random sequence of tokens
    # random_tokens1 = generate_random_tokens(tokenizer).to(device)
    # random_tokens2 = generate_random_tokens(tokenizer).to(device)
    random_tokens1 = generate_same_token_sequence(token_value1).to(device)
    random_tokens2 = generate_same_token_sequence(token_value2).to(device)
    random_tokens1 = generate_semi_random_tokens_limited_vocab(tokenizer).to(device)
    random_tokens2 = generate_semi_random_tokens_limited_vocab(tokenizer).to(device)
    assert random_tokens1.sum().item() != random_tokens2.sum().item(), "Two random sequences of tokens are the same!"
    print(f'{random_tokens1.shape=}')
    print(f'{random_tokens2.shape=}')
    print(f'{random_tokens1.sum()=}')
    print(f'{random_tokens2.sum()=}')

    # Compute the activations from the model
    activations1 = model(random_tokens1)
    activations2 = model(random_tokens2)
    # Extract the activations tensor
    activations1 = activations1[0]
    activations2 = activations2[0]
    # Reshape the activations tensor to the shape [B, T*D]
    # activations1 = activations1.view(activations1.size(0), -1)
    # activations2 = activations2.view(activations2.size(0), -1)
    # Reshape the activations tensor to the shape [B*T, D]
    activations1 = activations1.view(-1, activations1.size(-1))
    activations2 = activations2.view(-1, activations2.size(-1))

    # Print the shape of the activations tensor
    print(f"Shape of activations tensor: {activations1.shape}")
    print(f"Shape of activations tensor: {activations2.shape}")
    print(f'{activations1.sum()=}')
    print(f'{activations2.sum()=}')

    dist: torch.Tensor = svcca_distance(activations1, activations2)
    # dist: torch.Tensor = pwcca_distance_choose_best_layer_matrix(activations, activations, backend='svd', epsilon=1e-10)
    # dist, dists = temporal_cca(activations1, activations2)
    print(f'{dist=}')

def main2_percent_vs_avg_dist():
    """
    Main function to plot the relationship between percentage of vocabulary used in token generation
    and the average CCA distance between two sets of activations from a GPT-2 model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GPT-2 model and tokenizer
    model = GPT2Model.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    percentages = np.linspace(0.05, 1.0, 30)  # Range of percentages from 0.1 to 1.0
    avg_distances = []

    with torch.no_grad():
        for i, percentage in tqdm(enumerate(percentages)):
            print(f'{i=} percentage = {percentage}')
            torch.cuda.empty_cache()
            # Generate token sequences with the given percentage of the vocabulary
            random_tokens1 = generate_semi_random_tokens_limited_vocab(tokenizer, percentange_vocab=percentage, device=device)
            random_tokens2 = generate_semi_random_tokens_limited_vocab(tokenizer, percentange_vocab=percentage, device=device)
            torch.cuda.empty_cache()
            # Compute the activations from the model
            activations1 = model(random_tokens1)[0]
            activations2 = model(random_tokens2)[0]

            # Compute the activations
            # activations1 = activations1.view(random_tokens1.size(0), -1)
            # activations2 = activations2.view(random_tokens2.size(0), -1)
            # Reshape the activations tensor to the shape [B*T, D]
            activations1 = activations1.view(-1, activations1.size(-1))
            activations2 = activations2.view(-1, activations2.size(-1))
            torch.cuda.empty_cache()
            print(f'{activations1.shape=} {activations2.shape=}')

            # Calculate CCA distance
            # dist = svcca_distance(activations1, activations2)
            # dist = linear_cka_distance(activations1, activations2)
            dist = orthogonal_procrustes_distance(activations1, activations2)
            torch.cuda.empty_cache()
            div = dist.mean().item()
            print(f'{div=}')
            avg_distances.append(div)
            torch.cuda.empty_cache()

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, avg_distances, marker='o')
    plt.xlabel('Percentage of Vocabulary Used')
    plt.ylabel('Average CCA Distance')
    plt.title('Average CCA Distance vs. Vocabulary Usage Percentage')
    plt.grid(True)
    plt.show()
    # save plot as .png file to ~/beyond-scale-language-data-diversity
    plt.savefig(os.path.expanduser('~/beyond-scale-language-data-diversity/avg_cca_dist_vs_vocab_usage.png'))

def main3_percent_vs_avg_dist():
    """
    Main function to plot the relationship between percentage of vocabulary used in token generation
    and the average CCA distance between two sets of activations from a GPT-2 model,
    including 95% confidence intervals.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GPT-2 model and tokenizer
    model = GPT2Model.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f'{tokenizer.vocab_size=}')

    percentages = np.linspace(1.0/tokenizer.vocab_size, 1.0, 30)  # Range of percentages from 0.05 to 1.0
    # percentages = np.linspace(1.0/tokenizer.vocab_size, 0.02, 60)  # Range of percentages from 0.05 to 1.0
    # percentages = np.linspace(1.0/tokenizer.vocab_size, 0.001, 60)  # Range of percentages from 0.05 to 1.0
    avg_distances = []
    ci_values = []

    with torch.no_grad():
        for i, percentage in tqdm(enumerate(percentages)):
            print(f'{i=} percentage = {percentage}')
            torch.cuda.empty_cache()
            random_tokens1 = generate_semi_random_tokens_limited_vocab(tokenizer, percentange_vocab=percentage, device=device)
            random_tokens2 = generate_semi_random_tokens_limited_vocab(tokenizer, percentange_vocab=percentage, device=device)
            activations1 = model(random_tokens1)[0]
            activations2 = model(random_tokens2)[0]

            activations1 = activations1.view(-1, activations1.size(-1))
            activations2 = activations2.view(-1, activations2.size(-1))
            print(f'{activations1.shape=} {activations2.shape=}')

            dist = svcca_distance(activations1, activations2)
            # dist, _ = temporal_cca(activations1, activations2)
            dist_values = dist.view(-1).cpu().numpy()

            mean_dist = np.mean(dist_values)
            std_dist = np.std(dist_values)
            n_samples = len(dist_values)
            ci = 1.96 * (std_dist / np.sqrt(n_samples))
            div = mean_dist
            print(f'{div=} +- {ci}')

            avg_distances.append(mean_dist)
            ci_values.append(ci)

    # Plotting the results with 95% CI
    plt.figure(figsize=(10, 6))
    # plt.plot(percentages, avg_distances, marker='o')
    plt.errorbar(percentages, avg_distances, yerr=ci_values, fmt='-o', ecolor='lightgray', capsize=5)
    plt.xlabel('Percentage of Vocabulary Used')
    plt.ylabel('Average CCA Distance')
    plt.title('Average CCA Distance vs. Vocabulary Usage Percentage with 95% CI')
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.expanduser('~/beyond-scale-language-data-diversity/avg_cca_dist_vs_vocab_usage_with_ci.png'))

def main4_real_hf_dataset():
    """
    Main function to load the GPT-2 model, generate random tokens, and compute activations.
    """
    # set random seed
    set_random_seeds()

    # Determine if CUDA (GPU support) is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GPT-2 model and tokenizer
    model = GPT2Model.from_pretrained('gpt2')
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # print(f'{tokenizer.model_max_length=}')

    # Set the padding token in tokenizer to the EOS token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate a sequence of tokens from HF dataset
    dataset = load_dataset("c4", "en", split="train", streaming=True)
    # Preprocess the texts: encode them to input IDs (tokens)
    batch_size: int = 300
    # texts1: list[str] = [dataset[i]['text'] for i in range(batch_size)]  # Example: get batch_size texts from C4
    # texts2: list[str] = [dataset[i]['text'] for i in range(batch_size)]  # Example: get batch_size texts from C4
    # Get two batches of data from the dataset stream
    texts1 = list(dataset.take(2*batch_size))
    texts2 = texts1[batch_size:]
    texts1 = [example['text'] for example in texts1]
    texts2 = [example['text'] for example in texts2]
    # encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=model.config.n_positions)
    encodings1 = tokenizer(texts1, return_tensors='pt', padding=True, truncation=True, max_length=50)
    encodings2 = tokenizer(texts2, return_tensors='pt', padding=True, truncation=True, max_length=50)
    input_ids1 = encodings1['input_ids'].to(device)
    input_ids2 = encodings2['input_ids'].to(device)
    assert input_ids1.sum().item() != input_ids2.sum().item(), "Two  sequences of tokens are the same!"
    print(f'{input_ids1.shape=}')
    print(f'{input_ids2.shape=}')

    # Compute the activations from the model
    activations1 = model(input_ids1)
    activations2 = model(input_ids2)
    # Extract the activations tensor
    activations1 = activations1[0]
    activations2 = activations2[0]
    # Reshape the activations tensor to the shape [B, T*D]
    # activations1 = activations1.view(activations1.size(0), -1)
    # activations2 = activations2.view(activations2.size(0), -1)
    # Reshape the activations tensor to the shape [B*T, D]
    activations1 = activations1.view(-1, activations1.size(-1))
    activations2 = activations2.view(-1, activations2.size(-1))

    # Print the shape of the activations tensor
    print(f"Shape of activations tensor: {activations1.shape}")
    print(f"Shape of activations tensor: {activations2.shape}")
    print(f'{activations1.sum()=}')
    print(f'{activations2.sum()=}')

    dist: torch.Tensor = svcca_distance(activations1, activations2)
    # dist: torch.Tensor = pwcca_distance_choose_best_layer_matrix(activations, activations, backend='svd', epsilon=1e-10)
    # dist, dists = temporal_cca(activations1, activations2)
    print(f'{dist=}')

if __name__ == '__main__':
    import time
    start = time.time()
    # main()
    # main2_percent_vs_avg_dist()
    # main3_percent_vs_avg_dist()
    main4_real_hf_dataset()
    # print secs, mins, hours elapste one line
    print(f'Done!\a Time elapsed: {(time.time() - start):.2f}secs {((time.time() - start)/60):.2f}mins {((time.time() - start)/60/60):.2f}hours\a\a')