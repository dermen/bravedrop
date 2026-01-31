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

#### Running the inference server

```
nohup python marco_server.py > server_log.txt 2>&1 &
```

Then interact with it via `marco_score.py`, 

```
python brave/marco_score.py -i /path/to/crystal_plate_image.jpeg

python brave/marco_score.py -i /path/to/crystal_plate_image.png
```

Or do an installation


```
pip install -e .
marcoscore -h
```

