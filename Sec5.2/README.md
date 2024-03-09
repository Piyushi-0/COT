## Cell-Perturbation Experiment

This folder contains code for running the cell perturbation experiment

### Dataset:

The dataset should be named as 'hvg.h5ad' and placed in the `./datasets` directory relative to this folder.

The dataset can be downloaded through this [public link](https://polybox.ethz.ch/index.php/s/RAykIMfDl0qCJaM).

### Training:

Run `python3 main.py`. If you want to change hyperparams, you can specify the hyperparams and change accordingly.

Some example hyperparameters are:

`--lda`: Lambda for regularization  
`--lr`: Learning Rate  
`--noise_dim`: The dimension of the noise passed to the implicit model  

For training outsample models, specify the holdout parameter with the dose:

`python3 main.py --holdout 10`

### Eval:

For in-sample evaluation, please run:

`python eval.py --num_repeats 50`

**TODO**: Make this a flag to supply filename  
**Note**: Current filename is 'model_trained_1000.pth'. This needs to be present in the same directory as this file.  

This will repeat the experiment 50 times.  
It outputs the average per-dose and per-drug perturbation signatures, MMD, and Wasserstein Distances between the predicted and target distributions.

For out sample distributions, please run:

`python3 eval_outsample.py --num_repeats 50`

The outsample directory must exist in the same folder and must have a model file corresponding to each dose in the format model_{dose}.pth. 

