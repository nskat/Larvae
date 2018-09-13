# Larvae

Small programmes to load, process and classify larvae data in an unsupervised fashion.

## Prerequisites
Python 3.x \
numpy v.1.14.0 \
pandas v.0.23.0 \
tensorflow-gpu v.1.7.0 \
keras v.2.1.6 \
sklearn v.0.19.1 \
tables v.3.4.3 \
matplotlib v.2.2.2 \
scipy v.1.1.0 \
progress v.1.4 \
mpi4py v.3.0.0 \
OpenMPI v.3.1.1 \
Singularity v.2.5.2 \
CUDA v.9.0 \
cuDNN v.7.0 \



## Loading

First, we need to load and preprocess larvae data. Three different loaders exist: 
- loader.py : to load and process data using a window of any length 
- loader_lwft.py : to load and process data using a layered window Fourier transform 
- loader_mpi.py : same as loader.py, compatible with MPI 

All of these store data in .hdf5 files.

Usage :  
For loader.py :
```
python loader.py --path=/path/to/the/screen --save_dir=/path/to/save/dir --lines=line1,line2,...lineN --window=1 --screen=t15
```

For loader_lwft.py :
```
python loader_lwft.py --path=/path/to/the/screen --features=all --save_dir=/path/to/save/dir --lines=line1,line2,...lineN --window=1 --screen=t15
```

For loader_mpi.py :
```
mpiexec -np 4 python loader_mpi.py --path=/path/to/the/screen --save_dir=/path/to/save/dir --lines=line1,line2,...lineN --window=1 --screen=t15
```

## IDEC
The main model we use for our computations is based on the IDEC algorithm (Improved Deep Embedded Clustering with Local Structure Preservation by Guo et al.).

Usage :
With data already transformed and stored in a hdf5 file:
```
python IDEC_GPU.py --dataset=/path/to/data.hdf5 --n_clusters=x --maxiter=200000 --update_interval=500 --save_dir=/path/to/save_dir --epochs=100
```

If you also have pretrained weights for the autoencoder:
```
python IDEC_GPU.py --dataset=/path/to/data.hdf5 --n_clusters=x --maxiter=200000 --update_interval=500 --save_dir=/path/to/save_dir --ae_weights=path/to/pretrained/ae.h5
```

To resume training :
```
python IDEC_GPU.py --dataset=/path/to/data.hdf5 --n_clusters=x --maxiter=200000 --update_interval=500 --save_dir=/path/to/save_dir --idec_weights=path/to/weights.h5
```

This implementation is based on the one of Guo et al. (see https://github.com/XifengGuo/IDEC)


## Singularity
To allow to perform computations on a HPC cluster, we use encapsulated environments using Singularity. 
We provide here two recipes from which you can create environments that are fully compatible with our programmes.

To create a new sandbox environment with tensorflow-gpu :
```
singularity build --sandbox tensorflow_gpu.img tensorflow_gpu
```

Once the container built and this repo imported in it, you can perform computations.
For example, on TARS, the Pasteur cluster, we use:
```
singularity exec --writable --nv -H $HOME:/home/$USER tensorflow_gpu.img/ python /Larvae/IDEC/IDEC_GPU.py --dataset=/Larvae/path/to/dataset.hdf5 --n_clusters=6 --maxiter=800000 --update_interval=500 --save_dir=/Larvae/save_dir --epochs=50
```
On a pretty large line (typically GMR_72F11), we determined empirically that 500,000 < maxiter < 800,000, epochs=50 and update_interval=500 are satisfying values to obtain convergence.

However, on smaller lines, the update_interval may be reduced to 140, which is the value used in the paper.
For more information on the arguments:
```
singularity exec --writable --nv -H $HOME:/home/$USER tensorflow_gpu.img/ python /Larvae/IDEC/IDEC_GPU.py -h 
```

The other recipe allows the use MPI thanks to mpi4py to perform parallel computations on several nodes and speed up loading and processing of data. 
Usage example:
```
mpiexec -np 2 singularity exec --writable -H $HOME:/home/$USER -B /pasteur/projets/policy02/Larva-Screen/screens/:/screen centos7_mpi4py.img/ python /Larvae/utils/loader_mpi.py /screen --save_dir=/Larvae/test/ --lines=GMR_75G10_AE_01@UAS_TNT_2_0003,FCF_attP2_1500062@UAS_TNT_2_0003 --window=1 --screen=t5
```
For this to work, please make sure the OpenMPI version in the path is 3.1.1.

For more information on Singularity, please refer to https://www.sylabs.io/docs/

## Compute and save metrics

We can easily compute the time distributions, transitions and probabilities of the different clusters using:
```
python utils/metrics.py --path=/screen --n_clusters=6 --line=line --window=1 --screen=t15 --idec_weight=/path/to/weights.h5
```
The line should here include the experimental process (ex: GMR_72F11_AE_01@UAS_Chrimson_Venus_X_0070/r_LED100_30s2x15s30s) in order to remain coherent with the hit analysis.
The metrics are saved in .npz files supported by numpy, and plots are generated in .png format.

## Generate TRX files for visualisation of the classification
Depending on the model you trained, use the corresponding utils/plot_classification file.
Usage :

```
python visualize_classification_idec.py --path=path/to/trx/folder/ --n_clusters=6 --idec_weights=path/to/weights.h5 --window=1
```

Then simply run MATLAB_visualisation/Visualize_Larva_Clustering.m in MATLAB. 
You may need to resize the window by launching Visualize_Larva_Clustering.fig and modifying the settings.

## VaDE
We also made our own implementation of the VaDE algorithm (Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering by Jiang et al.).
The usage is similar to IDEC:
```
python VaDE_tensorflow.py --dataset=path/to/dataset.hdf5 --epochs=5000 --epochs_pretrain=50
```
For more info, please use
```
python VaDE_tensorflow.py -h
```

## Authors

Nicolas Skatchkovsky, Decision and Bayesian Computation, Institut Pasteur \
Jean-Baptiste Masson, Decision and Bayesian Computation, Institut Pasteur
