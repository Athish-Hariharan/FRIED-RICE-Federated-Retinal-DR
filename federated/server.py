import torch

def fed_avg(global_model, client_weights, client_sizes):

    total_samples = sum(client_sizes)
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        weighted_sum = 0

        for i in range(len(client_weights)):
            weight = client_sizes[i] / total_samples
            weighted_sum += client_weights[i][key].float() * weight

        global_dict[key] = weighted_sum

    global_model.load_state_dict(global_dict)
    return global_model

