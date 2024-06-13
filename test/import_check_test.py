import time


def main():
    start_time = time.time()

    from src.fjformer import checkpoint
    from src.fjformer import functions
    from src.fjformer import linen
    from src.fjformer import monitor
    from src.fjformer import optimizers
    from src.fjformer import pallas_operations
    from src.fjformer import sharding
    from src.fjformer import xrapture
    from src.fjformer import utils
    from src.fjformer.pallas_operations.pallas_attention import flash_attention
    from src.fjformer.pallas_operations.efficient_attention import efficient_attention

    from src.fjformer.pallas_operations.gpu.rms_norm import rms_norm
    from src.fjformer.pallas_operations.gpu.softmax import softmax
    from src.fjformer.pallas_operations.gpu.layer_norm import layer_norm
    from src.fjformer.pallas_operations.gpu.flash_attention import flash_attention

    from src.fjformer.pallas_operations.tpu.paged_attention import paged_attention
    from src.fjformer.pallas_operations.tpu.ring_attention import ring_attention
    from src.fjformer.pallas_operations.tpu.splash_attention import make_splash_mha
    from src.fjformer.pallas_operations.tpu.flash_attention import flash_attention

    end_time = time.time()
    print(f"Import time {end_time - start_time} sec")


if __name__ == "__main__":
    main()
