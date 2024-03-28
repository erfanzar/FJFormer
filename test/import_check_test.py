import time


def main():
    start_time = time.time()
    from src import fjformer
    end_time = time.time()
    print(f"Import time {end_time - start_time} sec")


if __name__ == "__main__":
    main()
