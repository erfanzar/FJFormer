import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__),
        ),
        "../src",
    )
)


def main():
    start_time = time.time()
    import fjformer as fjformer
    from fjformer import checkpoint as checkpoint
    from fjformer import functions as functions
    from fjformer import linen as linen
    from fjformer import lora as lora
    from fjformer import monitor as monitor
    from fjformer import optimizers as optimizers
    from fjformer import pallas_operations as pallas_operations
    from fjformer import sharding as sharding
    from fjformer import utils as utils
    from fjformer.pallas_operations.efficient_attention import (
        efficient_attention as efficient_attention,
    )
    from fjformer.pallas_operations.gpu.flash_attention import (
        flash_attention as flash_attention,
    )
    from fjformer.pallas_operations.gpu.layer_norm import layer_norm as layer_norm
    from fjformer.pallas_operations.gpu.rms_norm import rms_norm as rms_norm
    from fjformer.pallas_operations.gpu.softmax import softmax as softmax
    from fjformer.pallas_operations.pallas_attention import (
        flash_attention as _flash_attention,
    )

    del _flash_attention
    from fjformer.pallas_operations.tpu.flash_attention import (
        flash_attention as _flash_attention,
    )

    del _flash_attention
    from fjformer.pallas_operations.tpu.paged_attention import (
        paged_attention as paged_attention,
    )
    from fjformer.pallas_operations.tpu.ring_attention import (
        ring_attention as ring_attention,
    )
    from fjformer.pallas_operations.tpu.splash_attention import (
        make_splash_mha as make_splash_mha,
    )

    end_time = time.time()
    print(f"Import time {end_time - start_time} sec")


if __name__ == "__main__":
    main()
