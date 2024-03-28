from .adamw import (
    get_adamw_with_cosine_scheduler as get_adamw_with_cosine_scheduler,
    get_adamw_with_warm_up_cosine_scheduler as get_adamw_with_warm_up_cosine_scheduler,
    get_adamw_with_warmup_linear_scheduler as get_adamw_with_warmup_linear_scheduler,
    get_adamw_with_linear_scheduler as get_adamw_with_linear_scheduler
)
from .lion import (
    get_lion_with_cosine_scheduler as get_lion_with_cosine_scheduler,
    get_lion_with_with_warmup_linear_scheduler as get_lion_with_with_warmup_linear_scheduler,
    get_lion_with_warm_up_cosine_scheduler as get_lion_with_warm_up_cosine_scheduler,
    get_lion_with_linear_scheduler as get_lion_with_linear_scheduler
)
from .adafactor import (
    get_adafactor_with_cosine_scheduler as get_adafactor_with_cosine_scheduler,
    get_adafactor_with_warm_up_cosine_scheduler as get_adafactor_with_warm_up_cosine_scheduler,
    get_adafactor_with_warmup_linear_scheduler as get_adafactor_with_warmup_linear_scheduler,
    get_adafactor_with_linear_scheduler as get_adafactor_with_linear_scheduler
)

from .rmsprop import (
    get_rmsprop_with_cosine_scheduler as get_rmsprop_with_cosine_scheduler,
    get_rmsprop_with_linear_scheduler as get_rmsprop_with_linear_scheduler,
    get_rmsprop_with_warmup_linear_scheduler as get_rmsprop_with_warmup_linear_scheduler,
    get_rmsprop_with_warm_up_cosine_scheduler as get_rmsprop_with_warm_up_cosine_scheduler
)

from .optimizer_utils import (
    optax_add_scheduled_weight_decay as optax_add_scheduled_weight_decay
)

__all__ = (
    # RMS Prop
    "get_rmsprop_with_cosine_scheduler",
    "get_rmsprop_with_linear_scheduler",
    "get_rmsprop_with_warmup_linear_scheduler",
    "get_rmsprop_with_warm_up_cosine_scheduler",
    # Ada Factor
    "get_adafactor_with_cosine_scheduler",
    "get_adafactor_with_warm_up_cosine_scheduler",
    "get_adafactor_with_warmup_linear_scheduler",
    "get_adafactor_with_linear_scheduler",
    # Lion
    "get_lion_with_cosine_scheduler",
    "get_lion_with_with_warmup_linear_scheduler",
    "get_lion_with_warm_up_cosine_scheduler",
    "get_lion_with_linear_scheduler",
    # AdamW
    "get_adamw_with_cosine_scheduler",
    "get_adamw_with_warm_up_cosine_scheduler",
    "get_adamw_with_warmup_linear_scheduler",
    "get_adamw_with_linear_scheduler",

    # Non-Optimizers

    "optax_add_scheduled_weight_decay"
)
