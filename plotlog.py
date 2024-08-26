
from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("logfile", type=str)
ap.add_argument("--logscale", action="store_true")
ap.add_argument("--fs", type=int, default=14)
args = ap.parse_args()


import pylab as plt


lines = open(args.logfile, 'r')
lines = [l for l in lines if "Done" in l]
train_lines = [float(l.strip().split()[-1]) for l in lines if "train loss" in l]
test_lines = [float(l.strip().split()[-1]) for l in lines if "test loss" in l]

plt.plot( train_lines, 'o', label="train")
plt.plot(test_lines, 'x', color="tomato", label="test")
if args.logscale:
    plt.gca().set_yscale('log')
plt.legend(prop={"size":args.fs})
plt.gca().tick_params(labelsize=args.fs, length=0)
plt.ylabel("Loss", fontsize=args.fs)
plt.xlabel("epoch", fontsize=args.fs )
plt.gca().grid(ls="--")
plt.show()



