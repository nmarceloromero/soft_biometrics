#!/bin/bash

# Start virtualenv
#source /home/users/marcelo/labic2/env/bin/activate ;

# Check the index of the label, if none was passed, the label is x=0 (0_Gender)
args=("$@")
if [ $# != 0 ] 
then
	x=${args[0]}
else
	x=0
fi

python train_lstm_model.py -i $x -n 64 -s 2
python train_lstm_model.py -i $x -n 64 -s 3
python train_lstm_model.py -i $x -n 64 -s 4
python train_lstm_model.py -i $x -n 64 -s 5

python train_lstm_model.py -i $x -n 128 -s 2
python train_lstm_model.py -i $x -n 128 -s 3
python train_lstm_model.py -i $x -n 128 -s 4
python train_lstm_model.py -i $x -n 128 -s 5

python train_lstm_model.py -i $x -n 256 -s 2
python train_lstm_model.py -i $x -n 256 -s 3
python train_lstm_model.py -i $x -n 256 -s 4
python train_lstm_model.py -i $x -n 256 -s 5

python train_lstm_model.py -i $x -n 512 -s 2
python train_lstm_model.py -i $x -n 512 -s 3
python train_lstm_model.py -i $x -n 512 -s 4
python train_lstm_model.py -i $x -n 512 -s 5

python train_lstm_model.py -i $x -n 1024 -s 2
python train_lstm_model.py -i $x -n 1024 -s 3
python train_lstm_model.py -i $x -n 1024 -s 4
python train_lstm_model.py -i $x -n 1024 -s 5

python train_lstm_model.py -i $x -n 2048 -s 2
python train_lstm_model.py -i $x -n 2048 -s 3
python train_lstm_model.py -i $x -n 2048 -s 4
python train_lstm_model.py -i $x -n 2048 -s 5
