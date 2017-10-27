# Deeplearning
An introduction on deeplearning(Chainer)

## Installation:
1. Install chainer: 
`pip install --user chainer==2.0.0`
 
2. Install cupy:
`pip install --user cupy==1.0.0.1`

## Prep data
* [Download the dataset from here](https://drive.google.com/file/d/0By0A8jnpSd8lQWcxcDY0cWViN1E/view?usp=sharing)
* Generate input txt for each dataset using gen.py
`python gen.py -d <path to extracted data>`

## Training
* Select the model file and update it in main.py
* Change other parameter(optional):
  * Learning rate
  * Optimizer(Other meta params depending upon the optimizer selected).
  * Number of epoch
  * batch size
* Run `python main.py -h` to learn about configurable params
* Start training `python main.py -o <path to save net files> -d <dataset txt file>`
  
