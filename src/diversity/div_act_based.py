"""
Problem:
I want to compute the CCA distance between two LLMs. 
The shape of any (intermediate) layer is [B, T, D], where B is batch size, T is sequence length, and D is the dimensionality of the layer.
The problem is that the CCA distance is only defined for two matrices of shape [B, D], where B is batch size and D is the dimensionality of the layer.
What is the right way to compute the CCA distance between two LLMs?
"""
from transformers import GPT2Model, GPT2Tokenizer
import torch
import random
import time

from anatome.similarity import pwcca_distance_choose_best_layer_matrix, svcca_distance, linear_cka_distance, orthogonal_procrustes_distance

    # _default_backends = {'pwcca': partial(pwcca_distance_choose_best_layer_matrix, backend='svd', epsilon=1e-10),
    #                      'svcca': partial(svcca_distance, accept_rate=0.99, backend='svd'),
    #                      'lincka': partial(linear_cka_distance, reduce_bias=False),
    #                      "opd": orthogonal_procrustes_distance}

def generate_random_tokens(tokenizer: GPT2Tokenizer, 
                           sequence_length: int = 50, 
                           batch_size: int = 3, 
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

    # Generate a random sequence of tokens
    random_tokens = generate_random_tokens(tokenizer).to(device)

    # Compute the activations from the model
    activations = model(random_tokens1)
    # Extract the activations tensor
    activations = activations[0]
    print(f"Shape of activations tensor: {activations.shape}")
    # Reshape the activations tensor to the shape [B, T*D]
    activations = activations.view(activations.size(0), -1)
    # Print the shape of the activations tensor
    print(f"Shape of activations tensor: {activations.shape}")

    dis: torch.Tensor = svcca_distance(activations, activations)
    # dis: torch.Tensor = pwcca_distance_choose_best_layer_matrix(activations, activations, backend='svd', epsilon=1e-10)
    print(f'{dis=}')

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    # print secs, mins, hours elapste one line
    print(f'Done!\a Time elapsed: {(time.time() - start):.2f}secs {((time.time() - start)/60):.2f}mins {((time.time() - start)/60/60):.2f}hours')