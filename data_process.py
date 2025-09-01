import hydra
from omegaconf import DictConfig
from typing import BinaryIO
import os
from tqdm import tqdm
import multiprocessing
import torch
import numpy as np

from cs336_basics.bpe_tokenizer import BPETokenizer


def worker(args):
    chunk_start, chunk_end, data_path, vocab_path, merges_path, special_tokens = args
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)
    with open(data_path, "rb") as f:
        f.seek(chunk_start)
        chunk = f.read(chunk_end - chunk_start)
        tokenized_data = tokenizer.encode(chunk.decode("utf-8", errors='ignore'))
    return tokenized_data

def find_chunk_boundaries_v2(input_file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    input_file.seek(0, os.SEEK_END)
    file_size = input_file.tell()
    file_size = file_size
    input_file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        end_position = chunk_boundaries[bi + 1] if bi + 1 < len(chunk_boundaries) else file_size
        input_file.seek(initial_position)  # Start at boundary guess
        while initial_position < end_position:
            mini_chunk = input_file.read(mini_chunk_size)  # Read a mini chunk
            
            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk, if it exists
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                # print(f"step {bi} Found special token at {initial_position + found_at + len(split_special_token)}")
                break
            initial_position += mini_chunk_size
            if initial_position > end_position:
                chunk_boundaries[bi] = 0

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

@hydra.main(config_path="conf", config_name="config.yaml")
def preprocess(cfg: DictConfig):
    print("Preprocessing data...")
    with open(cfg.data.path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries_v2(f, 100, cfg.tokenizer.special_tokens[0].encode("utf-8"))    

    chunks = list(zip(chunk_boundaries[:-1], chunk_boundaries[1:]))

    args = [
        (start, end, cfg.data.path, cfg.tokenizer.vocab_path, cfg.tokenizer.merges_path, cfg.tokenizer.special_tokens)
        for start, end in chunks
    ]
    
    all_tokenized_data = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for tokenized_data in tqdm(pool.imap(worker, args), total=len(args), desc="Pretokenizing words"):
            all_tokenized_data.extend(tokenized_data)
    
    print(f"Finished pretokenizing all chunks. Total tokens: {len(all_tokenized_data)}")

    # Save to a binary file
    arr = np.array(all_tokenized_data, dtype='uint16')
    arr.tofile(cfg.data.tokenized_path)
    print(f"Tokenized data saved to {cfg.data.tokenized_path}")


if __name__ == "__main__":
    preprocess()
