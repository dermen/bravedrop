Algorithms and scripts for analyzing droplet images

#### The current model

```
wget https://smb.slac.stanford.edu/~dermen/model_epoch_130.net
```

That model was trained using

```
python DataLoader.py /data/brave/MARCO/MS/bravedrop/Final.log --resnet 34 --savepath /data/brave/MARCO/MS --devID 0
```

Here is the training log from that command:

```
wget https://smb.slac.stanford.edu/~dermen/Final.log
```

