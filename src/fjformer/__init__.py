from .checkpoint import (
    CheckpointManager as CheckpointManager,
    load_and_convert_checkpoint_to_torch as load_and_convert_checkpoint_to_torch,
    float_tensor_to_dtype as float_tensor_to_dtype,
    read_ckpt as read_ckpt,
    save_ckpt as save_ckpt,
    get_dtype as get_dtype,
)

from .partition_utils import (
    get_jax_mesh as get_jax_mesh,
    names_in_current_mesh as names_in_current_mesh,
    get_names_from_partition_spec as get_names_from_partition_spec,
    match_partition_rules as match_partition_rules,
    flatten_tree as flatten_tree,
    get_metrics as get_metrics,
    tree_apply as tree_apply,
    get_weight_decay_mask as get_weight_decay_mask,
    named_tree_map as named_tree_map,
    tree_path_to_string as tree_path_to_string,
    make_shard_and_gather_fns as make_shard_and_gather_fns,
    with_sharding_constraint as with_sharding_constraint,
    wrap_function_with_rng as wrap_function_with_rng,
    create_mesh as create_mesh
)

from .monitor import (
    run as smi_run,
    get_memory_information as get_memory_information,
    is_notebook as is_notebook,
    threaded_log as smi_threaded_log,
    initialise_tracking as smi_initialise_tracking
)

from .func import (
    average_metrics as average_metrics,
    global_norm as global_norm,
    transpose as transpose,
    auxiliary_load_balancing_loss_func as auxiliary_load_balancing_loss_func,
    get_loss_normalizing_factor_and_weights as get_loss_normalizing_factor_and_weights,
    convert_special_loss_normalizing_factor_to_enum as convert_special_loss_normalizing_factor_to_enum,
    SpecialLossNormalizingFactor as SpecialLossNormalizingFactor,
    cross_entropy_loss_and_accuracy as cross_entropy_loss_and_accuracy,
    fused_cross_entropy_loss_and_accuracy as fused_cross_entropy_loss_and_accuracy,
    compute_weighted_cross_entropy_and_accuracy as compute_weighted_cross_entropy_and_accuracy,
    compute_weighted_cross_entropy as compute_weighted_cross_entropy,
    binary_cross_entropy_onehot as binary_cross_entropy_onehot,
    binary_cross_entropy as binary_cross_entropy,
    cross_entropy as cross_entropy,
    cross_entropy_with_logits as cross_entropy_with_logits,
    cross_entropy_onehot as cross_entropy_onehot,
    l2 as l2,
    hinge as hinge,
    mae as mae,
    nll as nll,
    mse_loss as mse_loss,
    mse as mse,
)

from .utils import (
    JaxRNG as JaxRNG,
    GenerateRNG as GenerateRNG,
    init_rng as init_rng,
    next_rng as next_rng,
)

from . import pallas_operations as pallas_operations
from . import optimizers as optimizers
from . import linen as linen

__version__ = "0.0.58"

__all__ = (
    # Loss and extra function

    "transpose",
    "global_norm",
    "average_metrics",
    "auxiliary_load_balancing_loss_func",
    "get_loss_normalizing_factor_and_weights",
    "convert_special_loss_normalizing_factor_to_enum",
    "SpecialLossNormalizingFactor",
    "mse_loss",
    "mse",
    "cross_entropy_loss_and_accuracy",
    "fused_cross_entropy_loss_and_accuracy",
    "compute_weighted_cross_entropy_and_accuracy",
    "compute_weighted_cross_entropy",
    "binary_cross_entropy_onehot",
    "binary_cross_entropy",
    "cross_entropy",
    "cross_entropy_with_logits",
    "cross_entropy_onehot",
    "l2",
    "hinge",
    "mae",
    "nll",

    # RNG Utils

    "JaxRNG",
    "GenerateRNG",
    "init_rng",
    "next_rng",

    # Monitor Functions

    "smi_run",
    "get_memory_information",
    "is_notebook",
    "smi_threaded_log",
    "smi_initialise_tracking",

    # Partition Utils
    "get_jax_mesh",
    "names_in_current_mesh",
    "get_names_from_partition_spec",
    "match_partition_rules",
    "flatten_tree",
    "get_metrics",
    "tree_apply",
    "get_weight_decay_mask",
    "named_tree_map",
    "tree_path_to_string",
    "make_shard_and_gather_fns",
    "with_sharding_constraint",
    "wrap_function_with_rng",
    "create_mesh",

    # Checkpointing Utils
    "CheckpointManager",
    "load_and_convert_checkpoint_to_torch",
    "float_tensor_to_dtype",
    "read_ckpt",
    "save_ckpt",
    "get_dtype",

    # Pallas Operations
    "pallas_operations",

    # Optimizers
    "optimizers",

    # Linen
    "linen"
)
