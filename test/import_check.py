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
	from fjformer import lora as lora
	from fjformer import monitor as monitor
	from fjformer import optimizers as optimizers 
	from fjformer import sharding as sharding
	from fjformer import utils as utils
	from fjformer import kernels as kernels

	end_time = time.time()
	print(f"Import time {end_time - start_time} sec")


if __name__ == "__main__":
	main()
