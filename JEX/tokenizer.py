import jax
import jax.numpy as jnp

from typing import NamedTuple
import sentencepiece as spm     # google's tokenizer for now

tk_path = r"/media/roy/TOSHIBA EXT/models/gemma-2b/tokenizer.model"

# TODO: separate later; this is an inference library

def load_tokenizer(path: str) -> spm.SentencePieceProcessor:
    """
    Loads a sentencepiece model from a path

    Args:
        path: path to the model to load
    """
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(path)
    return tokenizer


def encode_str(text: str, tokenizer: spm.SentencePieceProcessor) -> jnp.ndarray:
    """
    Encodes a text into a tokenized jnp.array

    Args:
        text: string to turn into token ids
        tokenizer: loaded sentencepiece model
    """

    # hardcode <bos> for now
    bos = [tokenizer.bos_id()]
    c = tokenizer.EncodeAsIds(text)
    return bos + tokenizer.encode(text)


def decode_tk(arr: jnp.array, tokenizer: spm.SentencePieceProcessor) -> str:
    """
    Decodes a tokenized jnp.array into a string

    Args:
        text: tokenized jnp.ndarray to turn into a string
        tokenizer: loaded sentencepiece model
    """
    return tokenizer.decode(arr)


if __name__ == "__main__":
    tokenizer = load_tokenizer(tk_path)
    print("Loaded tokenizer")

    test_str = "bonjour, comment ça va?"
    encoded = encode_str(test_str, tokenizer)
    decoded = decode_tk(encoded, tokenizer)

    print(f"expected: {test_str}")
    print(f"got: {decoded}")
    print(f"ids: {encoded}")

    # get memory usage
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print(f"Memory used: {process.memory_info().rss / 1024**2} MB")

    print(encode_str("Q:What's the capital of France?\nA:", tokenizer))
    print(encode_str("Paris", tokenizer))

    print(decode_tk([7127], tokenizer))