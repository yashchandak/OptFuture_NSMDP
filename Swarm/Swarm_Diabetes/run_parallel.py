from multiprocessing import Pool as ThreadPool
from subprocess import call
from Src.run_NS import main as myfunction
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--base", default=490, help="Base counter for Hyper-param search", type=int)
parser.add_argument("--inc", default=0, help="Increment counter for Hyper-param search", type=int)
parser.add_argument("--hyper", default='Diabetes0', help="Which Hyper param settings")
args = parser.parse_args()

sequential = 1
parallel = 2

errors = []
# Sequential processing:
for seq in range(sequential):

    # Parallel processing
    pool = ThreadPool(parallel)
    my_array = []
    for par in range(parallel):
        my_array.append((True, args.inc*sequential*parallel + seq*parallel + par, args.hyper, args.base))

    try:
        results = pool.starmap(myfunction, my_array)
        # close the pool and wait for the work to finish
        pool.close()
        pool.join()

    except Exception as e:
        errors.append(e)
        idx = args.inc*sequential*parallel + seq*parallel
        print('Problem: {}. Skipping following threads during parallel execution: {},{}'.format(e, idx, idx + 1))

if len(errors) > 0:
    raise RuntimeError(errors)