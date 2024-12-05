'''Script to refactor training.py into a new structure'''

def partial_training(layers_to_unfreeze, model):
                # Step 2: Freeze all layers
                for param in model.parameters():
                    param.requires_grad = False
                
                for name, param in model.named_parameters():
                    if name in layers_to_unfreeze:
                        param.requires_grad = True
                print(f'only training {unfreeze_layers}')
                return model
                
                
def get_shared_memory_list(self, length=0):
    mp.current_process().authkey = np.arange(32, dtype=np.uint8).tobytes()
    shl0 = mp.Manager().list([None] * length)

    if self.distributed:
        # to support multi-node training, we need check for a local process group
        is_multinode = False

        if dist_launched():
            local_world_size = int(os.getenv("LOCAL_WORLD_SIZE"))
            world_size = int(os.getenv("WORLD_SIZE"))
            group_rank = int(os.getenv("GROUP_RANK"))
            if world_size > local_world_size:
                is_multinode = True
                # we're in multi-node, get local world sizes
                lw = torch.tensor(local_world_size, dtype=torch.int, device=self.device)
                lw_sizes = [torch.zeros_like(lw) for _ in range(world_size)]
                dist.all_gather(tensor_list=lw_sizes, tensor=lw)

                src = g_rank = 0
                while src < world_size:
                    # create sub-groups local to a node, to share memory only within a node
                    # and broadcast shared list within a node
                    group = dist.new_group(ranks=list(range(src, src + local_world_size)))
                    if group_rank == g_rank:
                        shl_list = [shl0]
                        dist.broadcast_object_list(shl_list, src=src, group=group, device=self.device)
                        shl = shl_list[0]
                    dist.destroy_process_group(group)
                    src = src + lw_sizes[src].item()  # rank of first process in the next node
                    g_rank += 1

        if not is_multinode:
            shl_list = [shl0]
            dist.broadcast_object_list(shl_list, src=0, device=self.device)
            shl = shl_list[0]

    else:
        shl = shl0

    return shl