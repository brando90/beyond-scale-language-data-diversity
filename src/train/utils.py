"""
todo:
    - finish passing the HF block_size tokenization code here so its modular
    - add function to our train code train.py
    - print the sequence length of the data once we include this code
    - create a unit test here to test block size
    - use the re-init code smart ally & brando wrote
"""

from itertools import chain

from transformers.testing_utils import CaptureLogger

def tokenize_function(examples, tokenizer):
    """
    
    To use do:
    tokenizer = ...obtained from your model... 
    tokenize_function = lambda examples: tokenize_function(examples, tokenizer=tokenizer) 
    tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )
    """
    with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return output

def group_texts(examples, block_size: int = 4096):
    """
    tokenizer = ...obtained from your model... 
    tokenize_function = lambda examples: tokenize_function(examples, tokenizer=tokenizer) 
    tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map    
    """
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def _test_all_batches_are_size_block_size():
    ...

if __name__ == "__main__":
    pass