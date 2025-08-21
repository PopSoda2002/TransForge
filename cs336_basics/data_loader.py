import numpy as np
import torch
from typing import Iterator, Tuple


def get_batch_data(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    # The highest possible starting index is len(x) - context_length - 1,
    # because we need a full sequence of context_length + 1 tokens.
    # numpy.random.randint's `high` parameter is exclusive, so we use len(x) - context_length.
    max_start_index = len(x) - context_length
    while True:
        # 1. Generate a batch of random starting indices
        start_indices = np.random.randint(0, max_start_index, size=(batch_size,))

        # 2. Create the input and target sequences using these indices
        x_sequences = [x[i : i + context_length] for i in start_indices]
        y_sequences = [x[i + 1 : i + context_length + 1] for i in start_indices]

        # 3. Convert lists of numpy arrays to a single numpy array, then to a tensor.
        # This is more efficient and avoids a PyTorch warning.
        x_batch = torch.tensor(np.array(x_sequences), device=device)
        y_batch = torch.tensor(np.array(y_sequences), device=device)

        yield x_batch, y_batch