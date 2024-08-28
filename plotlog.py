
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
test_lines = [float(l.strip().split("loss=")[1].split(";")[0]) for l in lines if "test loss" in l]
try:
    acc = [float(l.strip().split("accuracy=")[1].split("%")[0]) for l in lines if "test loss" in l]
except:
    acc = None

ax1 = plt.gca()

ax1.plot( train_lines, 'o', label="train")
ax1.plot(test_lines, 'x', color="tomato", label="test")
if args.logscale:
    ax1.set_yscale('log')
ax1.tick_params(labelsize=args.fs, length=2)
ax1.set_ylabel("Loss", fontsize=args.fs)
ax1.set_xlabel("epoch", fontsize=args.fs )

ax1.legend(prop={"size":args.fs}, loc=8)

if acc is not None:
    ax2 = plt.gca().twinx()
    ax2.plot(acc, 's', color="C2", label="test acc")
    ax2.set_ylabel("accuracy", color="C2", fontsize=args.fs, rotation=270, labelpad=15)
    ax2.tick_params(axis='y', labelcolor="C2", labelsize=args.fs)
    ax2.legend(prop={"size":args.fs}, loc=9)

#ax1.grid(ls="--")

plt.subplots_adjust(right=0.9)
#leg = plt.
plt.show()



