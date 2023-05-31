from jax import numpy as jnp
from transformers import AutoConfig, FlaxAutoModelForCausalLM, FlaxAutoModelForMaskedLM, FlaxAutoModel
from typing import Union
import os
from huggingface_hub import snapshot_download
from fjutils.checkpointing import StreamingCheckpointer

TRANSFORMERS_CLS: Union = [
    FlaxAutoModel,
    FlaxAutoModelForMaskedLM,
    FlaxAutoModelForCausalLM
]


def load_pretrained_model(repo_id: str, do_init=False, trust_remote_code: bool = True, dtype: jnp.dtype = jnp.float16,
                          transformers_cls: TRANSFORMERS_CLS = FlaxAutoModelForCausalLM,
                          cache_dir: str = 'model', model_msgpack_name: str = 'mlxu_model_checkpoints',
                          target: str = 'params', disallow_trainstate: bool = False,
                          **kwargs):
    try:
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=trust_remote_code, dtype=dtype, **kwargs)
    except OSError:
        raise ValueError(f'{repo_id} Not found in huggingface ! make sure this model exists or login to your account')

    model = transformers_cls.from_config(config, trust_remote_code=trust_remote_code, dtype=dtype,
                                         _do_init=do_init)
    print('sending req')

    snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
    path = cache_dir + '/models--' + repo_id.replace('/', '--') + '/' + 'snapshots'
    shot = [s for s in os.listdir(path) if os.path.exists(os.path.join(path, s))][0]
    path_shot = path + f'/{shot}'
    msgpack_path = path_shot + f'/{model_msgpack_name}'
    data = StreamingCheckpointer.load_trainstate_checkpoint(f"{target}::{msgpack_path}",
                                                            disallow_trainstate=disallow_trainstate)
    return model, data, config
