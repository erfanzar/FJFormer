from datasets import load_dataset
from typing import Any, Callable, Optional
from jax import numpy as jnp
from torch.utils.data import DataLoader


def get_dataloader(dataset_or_huggingface_dataset_hub_id: Any, batch_size: int, num_epochs: int,
                   select_hf_dataset_field='train',
                   max_steps: int = None, max_length: int = 4096, dataset_hf_kwargs: dict = {},
                   collate_fn: Callable = None, shuffle: Optional[bool] = None,
                   sampler=None,
                   batch_sampler=None,
                   num_workers: int = 0,
                   pin_memory: bool = False, drop_last: bool = False,
                   timeout: float = 0, worker_init_fn=None,
                   multiprocessing_context=None, generator=None,
                   *, prefetch_factor: Optional[int] = None,
                   persistent_workers: bool = False,
                   pin_memory_device: str = ""):
    if collate_fn is None:
        def collate_fn(batch):
            rs = {}
            for key in batch[0].keys():
                ssp = [jnp.array(f[key])[..., -max_length:] for f in batch]
                rs[key] = jnp.stack(ssp).reshape(-1, ssp[0].shape[-1])
            return rs
    if isinstance(dataset_or_huggingface_dataset_hub_id, str):

        dataset = load_dataset(dataset_or_huggingface_dataset_hub_id, **dataset_hf_kwargs)[select_hf_dataset_field]
    else:
        dataset = dataset_or_huggingface_dataset_hub_id

    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        shuffle=shuffle,
        timeout=timeout,
        sampler=sampler, batch_sampler=batch_sampler,
        drop_last=drop_last,
        generator=generator, persistent_workers=persistent_workers,
        pin_memory_device=pin_memory_device,
        multiprocessing_context=multiprocessing_context, worker_init_fn=worker_init_fn

    )
    max_steps = num_epochs * len(dataloader) if max_steps is None else max_steps
    return dataloader, max_steps
