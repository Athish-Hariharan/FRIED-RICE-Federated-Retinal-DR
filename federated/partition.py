import numpy as np
from torch.utils.data import Subset

def dirichlet_partition(dataset, num_clients=10, alpha=0.5):

    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):

        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions / proportions.sum())

        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        split_indices = np.split(idx_c, splits)

        for i in range(num_clients):
            client_indices[i].extend(split_indices[i])

    return [Subset(dataset, idxs) for idxs in client_indices]
