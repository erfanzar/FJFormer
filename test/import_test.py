from time import time


def main():
    start = time()
    import src.fjformer as fjformer
    print(fjformer)
    print(time() - start)


if __name__ == '__main__':
    main()
