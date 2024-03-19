import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/') # need to add the import folder to system path for python to know where to look for
import torch
from Input.config import dropout


def dropout_network(network, dropout_percentage=dropout):
    if dropout_percentage is not None and 0 < dropout_percentage < 1:
        # Perform the dropout
        for name, param in network.named_parameters():
            if 'up_samples.2.1.deconv.weight' in name:
                # Flatten the parameter tensor to 1D
                flattened_params = param.view(-1)
                total_weights = flattened_params.numel()

                # Calculate the number of weights to dropout
                weights_to_dropout = int(dropout_percentage * total_weights)

                # Generate random indices for dropout
                indices_to_dropout = torch.randperm(total_weights)[:weights_to_dropout]

                # Set the selected weights to zero
                with torch.no_grad():  # Ensure these operations are not tracked by autograd
                    flattened_params[indices_to_dropout] = flattened_params.min()
                # The original param tensor is modified due to the in-place operation on the view.
    return network
