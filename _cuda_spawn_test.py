import multiprocessing as mp
import os


def worker():
    import torch

    print("child pid", os.getpid())
    print("is_available", torch.cuda.is_available())
    try:
        torch.cuda.set_device(0)
        x = torch.zeros(1, device="cuda")
        print("child cuda ok", x)
    except Exception as e:
        print("child cuda FAIL", type(e), e)


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=worker)
    p.start()
    p.join()
    print("exit", p.exitcode)
