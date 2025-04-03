# Training UNISON 2025
Exercises for training given at Unison 2025


You need to download the exercises in the Unison cluster

```bash
git pull https://github.com/bvillasen/training_unison_2025.git
```

# Create a conda environment

First you need to load the conda module

```bash 
module load conda
```

Then, we create a conda environment, in the example below the environment will be named `my_conda_env`, you can use another name.  

```bash
conda create -n my_conda_env python=3.10
```

After the environment is created, you need to activate it every time you plan to use it

```bash
conda activate my_conda_env
```

# Pytorch

## Install Pytorch

Load and activate your conda environment

```bash
module load conda
conda activate my_conda_env
```

After you have created and activated your conda environment, install Pytorch using pip. Look for the Pytorch install command in the Pytorch website [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
```

Check that the Pytorch installation was successful

```bash
python -c "import torch; print(f'Pytorch version: {torch.__version__}')"
```

## Run Pytorch examples

Go to the directory where there are some Pytorch examples

```bash
cd pytroch_examples
```

Run a very simple example

```bash
python pytorch_simple.py
```

Now run a more complicated example that trains a neural network based on the CIFAR-10 dataset

```bash
python pytorch_train_cifar.py
```


# Tensorflow

## Install Tensorflow


```bash
module load conda
conda activate my_conda_env
```

After you have created and activated your conda environment, install TensorFlow using pip. The instruction below was taken from here [https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html#using-a-wheels-package](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html#using-a-wheels-package) 


```bash
pip install tensorflow-rocm==2.16.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2/ --upgrade
```

Check that the Tensorflow installation was successful

```bash
python -c 'import tensorflow' 2> /dev/null && echo ‘Success’ || echo ‘Failure’
```


## Exercises suite

We have a full suite of exercises you can work on here: [https://github.com/amd/HPCTrainingExamples/tree/main](https://github.com/amd/HPCTrainingExamples/tree/main)

## Other useful resources 

