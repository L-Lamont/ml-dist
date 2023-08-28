# TensorFlow
This details the process of distributing TensorFlow to multiple GPUs on a slurm 
managed cluster. It also assumes a module system is available for loading some 
software required in the installation. Please note that this distributes an 
mnist example so performance scaling can not be judged as the overhead of 
training is much higher than the compute required for training.

## Installation
### Apptainer
Pull the TensorFlow container from dockerhub and place it somewhere that will 
be visible to your jobscript. By default apptainer uses a cache directory which 
is placed in your home directory, some sites have limits to the size of your 
home directories so updating the `APPTAINER_CACHEDIR` environment variable to a 
path not in your home directory can prevent some issues caused by your home 
directory filling up. The following 2 commands should be all that is necessary 
to pull the container and update your cache location. It is important to pull a 
GPU version of the TensorFlow container the default `tensorflow/tensorflow` is 
not and will cause scripts to fail because they can't see any GPUs

```
APPTAINER_CACHEDIR="/not/my/home/directory"
apptainer pull docker://tensorflow/tensorflow:latest-gpu
```

### Virtual Environment
To build a python virtual environment update the `venv_build.job` script located 
at `ml-dist/tensorflow/venv_build.job`. The SBATCH parameters, proxy variables 
(if required), module versions loaded and some variables such as `ML_DIST_PATH` 
need to be updated before use.

Building the vitual environment on the servers you plan to run the training or 
inference on can help reduce the number of errors that occur because the 
software is built for a different environment, because of this the example 
script is a jobscript which can be submitted to slurm and run on a targeted 
architecture. Similiar errors can also arise when the modules loaded at built 
time are different to the modules loaded at runtime. Purging all modules using 
`module purge` and then loading in the specific versions required is helpful to 
minimise these errors.

## Jobscript
There are a set of example jobscript for launching TensorFlow on a slurm 
managed cluster in `ml-dist/tensorflow`. All scripts use the `USE_CONTAINER` 
variable to toggle between launching in a virtual environment and in a 
container. To run on a single GPU use the `base_tensorflow.job`, to use the 
multi worker mirrored strategy to distribute to multiple GPUs or servers use 
`mwm_tensorflow.job` and to use horovod use `horovod_tensorflow.job` (horovod
will be discussed more later).

### Environment Variables
When distributing using the multi worker mirrored strategy the `http_proxy` and
`https_proxy` variables must be unset, if your cluster uses proxy servers you 
will need to pre download datasets and load them from disk at runtime instead 
of downloading them at runtime. The examples shown use XLA a optmizing compiler 
to optimize training. If you are attempting to run non-distributed training on 
server with multiple GPUs you will have to restrict the visible GPUs using 
`CUDA_VISIBLE_DEVICES=0` in the jobscript, otherwise XLA will attempt to 
distribute training to all GPUs, this is also useful to note if you just want 
to distribute training to multiple GPUs on a single server, without any code 
changes. You may also need to set the 
`XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_PATH}` if your cuda is installed in a 
non-standard directory path, this problem could arrise if you use a module 
system to load cuda, this solution assumes `CUDA_PATH` is set by the cuda module.

### Venv
For mutliworker mirrored strategy `srun` is used to start mutliple versions 
of the training script on every server, each script then uses a unique GPU to 
run on. `srun` is used and not `mpirun` because we have used the slurm cluster 
to distribute training and it relies on slurm environment variables that are 
not always set when using `mpirun`, for example `SLURM_STEP_NUM_TASKS`. In 
mwm_training.py we also use `SLURM_LOCALID` to pin each process to a unique GPU 
(using set_visible_devices function from `tensorflow.config`).

### Apptainer
Apptainer adds one layer of complexity, instead of `srun` directly running the 
training script it now launches `apptainer run` which starts each rank inside 
of a container.

### Usage
The follwoing specifies the changes to `ml-dist/tensorflow/mwm_tensorflow.job`
that are required before it will work.

SBATCH Parameters
- #SBATCH --nodes = number of servers
- #SBATCH --job-name = name of the job (used in output naming in example script)
- #SBATCH --partition = the queue or partition to use
- #SBATCH --constraint = the constraint to target a specific machine
- #SBATCH --tasks-per-node = the number of gpus per server

The module versions loaded need to be added (`module load python/3.11.2`). The
`ML_DIST_PATH` variable need to be changed to reflect the correct path. If 
internet access is via a proxy server the `http_proxy` and `https_proxy` 
environment variables need to be updated, remeber if you are using the multi 
worker mirrored strategy to distribute training you need to pre-download data 
and unset these variables. There are several variables that could be updated 
like the `CONTAINER_NAME` and `TRAIN_SCRIPT`. It is important also to check that 
all paths are correct, `mwm_tensorflow.job` is written to run the example job 
which writes output to and reads data from known places if you are modifying 
these locations then the paths will have to be updated (be sure that they are 
available in the container by updating the bindpath in apptainerArgs too).

## Changes to training.py
There are no changes to the training script if you are using a virtual 
environment or a container.

A full example of a multi worker mirrored strategy script can be found at 
`ml-dist/tensorflow/mwm_training.py`, an example of a single GPU training script
can be found at `ml-dist/tensorflow/base_training.py`. The changes required to 
implement the multi worker mirrored strategy are outlined below. One issue we 
ran into was using NCCL as the CommunicationImplementation, on our system 
training stalled and only worked if we used TensorFlow's RING implementation.

```
# Set each process to see a unique GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[int(os.environ['SLURM_LOCALID'])]
tf.config.set_visible_devices(gpu, 'GPU')

# Allow memory growth on the GPU
tf.config.experimental.set_memory_growth(gpu, True)

# Choose a communication implementation
communication_options = tf.distribute.experimental.CommunicationOptions(
    # implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
    implementation=tf.distribute.experimental.CommunicationImplementation.RING
)

# Use the slurm cluster resolver to find other processes
cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=communication_options,
    cluster_resolver=cluster_resolver
)

# The batchsize argument now sets the batch_size local to 1 gpu so the 
# effective global batchsize will be the number of gpus * batch_size

# With the larger batchsize you can scale the learning rate so the "steps"
# are larger and training is faster
args.lr = args.lr * int(os.environ["SLURM_NTASKS"])

# Wrap the model definition and compilation in strategy.scope()
with strategy.scope():
    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        metrics=['accuracy']
    )

# Use SLURM_PROCID to run certain commands only once for example certain log
# messages and saving the model
if args.save_model and os.environ['SLURM_PROCID'] == 0:
    tf.keras.Sequential.save(model, filepath="{}/mnist_model".format(args.output_dir))
```

# Horovod
TensorFlow mutli worker mirroredstrategy is not the only option for 
distributing TensorFlow training you can also use horovod.

Horovod uses MPI to distribute training to multiple GPUs rather than the 
inbuilt mutli worker mirrored strategy. It still uses data parallelism to 
distribute training though.

## Installation
### Apptainer
Similiarly to TensorFlow mutli worker mirrored strategy you can pull a 
pre-built container from dockerhub. The same issues with cache directories 
presist so be sure to update the `APPTAINER_CACHEDIR` environment variable.

## Virtual Environment
The example script `ml-dist/tensorflow/venv_build.job` includes an example of how 
to build horovod it requries a few extra environment variables, update the 
`INSTALL_HOROVOD` variable to 1 to create a virtual environment with horovod you 
will also need to update the `NCCL_PATH` variable which is passed to horovod 
during the build stage.

## Jobscript
When using horovod to distribute training you can choose between `mpirun` and 
`srun` to launch processes. Horovod also provides `horovodrun` which is a wrapper 
around `mpirun` however we have encountered issues during multi server training 
using it. A source of errors we encountered were slurm environment variables 
being incorrectly set when using `mpirun`, `SLURM_PROCID` was always set to 0 for
every process. This is easily overcome by remembering to use `hvd.local_rank()`, 
`hvd.rank()`, `hvd.local_size` and `hvd.size()` instead of environment variables.

## Usage
When distributing using horovod communication uses MPI because of this you no 
longer need to unset the `http_proxy` and `https_proxy` environment variables.

## Changes
```
# import horovod
import horovod.tensorflow.keras as hvd

# Use horovod.local_rank() so each process sees a unique GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[int(hvd.local_rank())]
tf.config.set_visible_devices(gpu, 'GPU')

# Allow memory growth on the GPU
tf.config.experimental.set_memory_growth(gpu, True)

# Scale the learning rate
args.lr = args.lr * int(os.environ["SLURM_NTASKS"])

# Wrap your optimizer in hvd.DistributedOptimizer
opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
opt = hvd.DistributedOptimizer(opt)

# Use horovod.rank() to run certain commands only once for example certain log
# messages and saving the model
if args.save_model and hvd.rank() == 0:
    tf.keras.Sequential.save(model, filepath="{}/mnist_model".format(args.output_dir))
```
