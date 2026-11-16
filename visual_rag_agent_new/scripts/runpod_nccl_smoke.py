#!/usr/bin/env python3
"""On the target Pod: torchrun --standalone --nproc_per_node=4 scripts/runpod_nccl_smoke.py.

Checks one-node, four-GPU NCCL communication; this does not validate model training.
"""
from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 4, "Expected exactly four processes"
    assert torch.cuda.device_count() == 4, "Expected exactly four visible GPUs"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        value = torch.full((1024,), rank + 1.0, device=f"cuda:{rank}")
        dist.all_reduce(value)
        torch.testing.assert_close(value, torch.full_like(value, 10.0))
        dist.barrier()
        if rank == 0:
            print(json.dumps({"world_size": 4, "nccl_all_reduce_correct": True,
                              "training_validated": False}))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
