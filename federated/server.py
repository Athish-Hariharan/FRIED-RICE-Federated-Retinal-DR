import torch

def fed_avg(global_model, client_weights):
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_weights[i][key].float() for i in range(len(client_weights))],
            dim=0
        ).mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model
