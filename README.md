Algorithms and scripts for analyzing droplet images


#### Getting th MARCO training data

```
# get terf for unpacking imgs 

wget https://github.com/ubccr/terf/releases/download/v0.0.3/terf-0.0.3-0-gde97e3c-linux-amd64.zip

unzip terf-0.0.3-0-gde97e3c-linux-amd64.zip 

# get MARCO data
wget --recursive --no-parent https://marco.ccr.buffalo.edu/data/

# extract jpegs
cd marco.ccr.buffalo.edu/data/archive/

tar -xvf train-jpg-tfrecords.tar 
tar -xvf test-jpg-tfrecords.tar 

../../../terf-0.0.3-0-gde97e3c-linux-amd64/terf  -d extract --input train-jpg -o train_out
../../../terf-0.0.3-0-gde97e3c-linux-amd64/terf  -d extract --input test-jpg -o test_out
```

The model below was trained by reading the CSV files

```
training_file = 'marco.ccr.buffalo.edu/data/archive/train_out/info.csv'
testing_file = 'marco.ccr.buffalo.edu/data/archive/test_out/info.csv'
```

which are obtained upon issuring the above commands...


#### The current model

```
wget https://smb.slac.stanford.edu/~dermen/model_epoch_130.net
```

That model was trained using

```
python DataLoader.py /data/brave/MARCO/MS/bravedrop/Final.log --resnet 34 --savepath /data/brave/MARCO/MS --devID 0
```

Be sure to update the info.csv paths which are hard-coded in the `DataLoader.py` script. 

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

