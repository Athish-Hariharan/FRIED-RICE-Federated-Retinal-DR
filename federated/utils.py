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

import numpy as np
from torch.utils.data import Subset

def create_dirichlet_partitions(dataset, num_clients=3, alpha=0.5, seed=42):
    """
    Dirichlet-based non-IID partitioning.

    alpha controls heterogeneity:
        small alpha → strong non-IID
        large alpha → near IID
    """

    np.random.seed(seed)

    labels = np.array(dataset.targets)  # Make sure AptosDataset has .targets
    num_classes = len(np.unique(labels))

    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):

        # Sample proportions for each client
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # Scale proportions by number of samples in class
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]

        split_indices = np.split(class_indices[c], proportions)

        for client_id in range(num_clients):
            client_indices[client_id].extend(split_indices[client_id])

    # Create Subsets
    client_datasets = [
        Subset(dataset, indices) for indices in client_indices
    ]

    return client_datasets
