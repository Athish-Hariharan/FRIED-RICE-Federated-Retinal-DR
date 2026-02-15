import numpy as np
from torch.utils.data import Subset

def create_non_iid_partitions(dataset, num_clients=3):

    labels = np.array(dataset.labels)
    indices = np.arange(len(dataset))

    client_indices = [[] for _ in range(num_clients)]

    # Sort by label
    sorted_indices = indices[np.argsort(labels)]

    # Split sorted labels into chunks
    shards = np.array_split(sorted_indices, num_clients)

    for i in range(num_clients):
        client_indices[i] = shards[i]

    return [Subset(dataset, idxs) for idxs in client_indices]
