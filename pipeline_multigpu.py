import torch.multiprocessing as mp
import os
from icefall_utils.dist import cleanup_dist, setup_dist
from torch.nn.parallel import DistributedDataParallel as DDP


WORLD_SIZE = int(os.environ.get("WORLD_SIZE"))

def run(rank, world_size, args):
    if world_size > 1:
        setup_dist(rank, world_size, master_port=None)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)