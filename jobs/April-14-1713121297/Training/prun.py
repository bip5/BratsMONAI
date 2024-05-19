import sys
sys.path.append('/scratch/a.bip5/BraTS/scripts/') # need to add the import folder to system path for python to know where to look for
import torch
from Input.config import PRUNE_PERCENTAGE

def prune_network(network):
    if PRUNE_PERCENTAGE is not None:
        # Perform the pruning
        for name, param in network.named_parameters():
            if 'weight' in name:
                # Flatten the parameter tensor to 1D and get the indices of the elements in sorted order
                flattened_params = param.view(-1)
                _, indices_array = torch.sort(flattened_params)

                # Determine how many weights to delete based on the pruning percentage
                weights_to_delete = int(PRUNE_PERCENTAGE * flattened_params.numel())

                # Set the smallest 'weights_to_delete' number of parameters to zero
                with torch.no_grad():  # In-place operations should not be tracked by autograd
                    if weights_to_delete<0:
                        print('max before', flattened_params.max())
                        flattened_params[indices_array[weights_to_delete:]]=0
                        print('max after', flattened_params.max())
                    else:
                        print('max before', flattened_params.max())
                        flattened_params[indices_array[:weights_to_delete]] = 0
                        print('max after', flattened_params.max())
                # The original param tensor is modified due to the in-place operation on the view.
    return network