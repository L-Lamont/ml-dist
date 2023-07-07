# PyTorch
This details the process of distributing pytorch to multiple nodes on a 
slurm managed cluster. It also assumes a module system is availble for loading
some software required in installation.

## Installation
### Apptainer
Pull the pytorch container from dockerhub and place it somewhere that will be 
visible to your jobscript. By default apptainer uses a cache directory which is 
placed in your home directory, some sites have limits to the size of your home 
directories so updating the APPTAINER_CACHEDIR environment variable so the limit 
is not increased is useful. The following 2 commands should be all that is 
necessary to pull the container and update your cache location.

```
APPTAINER_CACHEDIR="/not/my/home/directory"
apptainer pull docker://pytorch/pytorch
```

### Virtual Environment
To build a python virtual environment update the venv_build.job script located 
at ml-dist/pytorch/venv_build.job. The SBATCH parameters, proxy variables (if
required), module versions loaded and some variables will need to be updated 
before use.

Building on the nodes you plan to run the training on can help reduce the 
number of errors that occur because software is built for a different 
environment, because of this the example script is a jobscript which can be 
submitted to slurm and on a targeted architecture. Similiar errors can also 
arise when the modules loaded at build time are not the same as the modules 
loaded at runtime. Purging all modules using `module purge` and then loading in 
the specific versions required is helpful to minimise these errors.


## Jobscript
There is an example jobscript for launching using a container and using a 
virutal environment located at ml-dist/pytorch/ddp_pytorch.job. It uses the 
USE_CONTAINER variable to toggle between launching in a virtual environment and 
a container. The methods used to start training are very similiar using the 
container just adds one level of complexity to the launch process.

### Environment Variables
`MASTER_ADDR` should be set to one the address of one of the nodes being used a 
random port should be appended to it, this is then past to the rndzv_endpoint 
argument of `torchrun`. Utilising scontrol is a simple way to get the name of 
node when the addresses could be changing depending on the nodes you are 
allocated. In multi server training you may also need to set and pass to MPI 
the `NCCL_IB_GID_INDEX`, it controls the global ID used in RoCE mode and had to 
be set to 3 for multi server training to work (it may be set automatically by 
slurm so it is not always required)

### Venv
`torchrun` is used to start multiple versions of the training script on a single 
node, each script then uses a unique GPU to run on and communication between 
them is controlled by the `rdzv_endpoint` and `rdzv_backend` torchrun arguments. 
To launch for multiple nodes we prepend the `mpirun` command to that which then 
launches a copy of torchrun per node and training continues the same as before.
An important thing to remember is that the number of tasks per node (controlled 
by `#SBATCH --ntasks-per-node`) is always 1 and the number of GPUs used per node 
is controlled via the torchrun arguments.

### Apptainer
If you are using Apptainer you now need to launch torchrun inside of a 
container, to do this use the add the `apptainer run pytorch_latest.sif` command 
before `torchrun`. It requires 2 arguments `nv` to allow the container to 
use cuda gpus and `B` to set the bindpath it is easiest to bind in the root 
directory of your project so you do not have to change other arguments (This is 
what is done in ml-dist/pytorch/ddp_pytorch.job).jj

### Usage
The following specifies the changes to ml-dist/pytorch/ddp_pytorch.job that 
are required before they will work.

SBATCH Parameters
- #SBATCH --nodes = number of nodes
- #SBATCH --job-name = name of the job (used in output naming in example script)
- #SBATCH --partition = the queue or partition to use
- #SBATCH --constraint = the constraint to target a specific machine

The module versions loaded needed to be added (`module load python/3.11.2`). The
`ML_DIST_PATH` and `GPUS_PER_NODE` variable need to be changed to reflect the paths
and number of gpus the servers being targeted have. If you need proxy variables 
set you will also need to set the `http_proxy` and `https_proxy` environment 
variables, but these are not always required. There are several variables that
could be updated like the `CONTAINER_NAME` and `TRAIN_SCRIPT`. It is important 
also to check that all paths are correct, `ddp_pytorch.job` is written to run the
example jobs which place output and data in known places if you are modifying 
these locations then the paths will have to be updated (be sure that they are
available in the container by updating the bindpath in apptainerArgs too).

## Changes to training.py
There are no changes to the training script if you are using a virtual 
environment or a container. 

A full example of a distributed data parallel script can be found at 
ml-dist/pytorch/ddp_example.py, an example of a single GPU training script 
can be found at ml-dist/pytorch/base_training.py. The changes required to 
implement distributed data parallel are outlined below.
```
# Add the specific imports
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from types import SimpleNamespace

def init_dist()
  # Gather information about the distributed training using the environment 
  # variables torchrun sets
  size = int(os.environ["WORLD_SIZE"])
  rank = int(os.environ["RANK"])
  local_size = int(os.environ["LOCAL_WORLD_SIZE"])
  local_rank = int(os.environ["LOCAL_RANK"])

  # Initialise the distribution there are other options but 'nccl' works well 
  # for training on gpus
  dist.init_process_group('nccl')

  # Create dist_info for easier reference to the variables
  return = SimpleNamespace(
    rank=rank, size=size,
    local_size=local_size, local_rank=local_rank
  )

# Set CUDA_VISIBLE_DEVICES and create the device
# - this limits the process to one specific gpu so processes don't fight over a 
#   single gpu
dist = init_dist()
os.environ["CUDA_VISIBLE_DEVICES"] = str(dist.local_rank)
device = torch.device('cuda:{}'.format(dist.local_rank))

# Scale the learning rate by the size
lr = lr * dist_info.size

# Wrap the train and test samplers in DistributedSampler
train_sampler = DistributedSampler(train_dataset, shuffle=True)
test_sampler = DistributedSampler(test_dataset, shuffle=True)

# Wrap the model in DDP
model = DDP(model, device_ids=[device])

# Use dist.rank to run certain commands only once for example certain log 
# messages and saving the model
if dist.rank == 0:
  torch.save(model.state_dict(), "{}/mnist_cnn.pt".format(args.output_dir))
  
# cleanup the process, pytorch sometimes fails to clean up the process group 
# properly so we need to explicitly cleanup the process group
dist.destroy_process_group()
```

### Other Distribution Strategies
PyTorch distributed data parallel is not the only option for distributing 
PyTorch training, you can also use horovod which was originally developed by 
Uber, Accelerate developed by HuggingFace.

## Horovod
Horovod uses MPI to distribute training to multiple GPUs rather than the 
inbuilt distributed data parallel. It stil uses data parallelism to distribute 
training though.

### Installation
#### Containers
Similiarly to PyTorch Distributed Data Parallel you can pull a pre-built 
container from dockerhub. The same issues with cache directories persist so be 
sure to update the APPTAINER_CACHEDIR environment variable.

#### Venv
The example script ml-dist/pytorch/venv_build.job includes an example of how to
build horovod it requires a few extra environment variables, update the 
`INSTALL_HOROVOD` variable to 1 to create a virtual environment with horovod 
you will also need to update the `NCCL_PATH` variable which is passed to 
horovod during the build stage.

#### Jobscript
An example jobscript is provided at ml-dist/pytorch/horovod_pytorch.job When 
using horovod torchrun is no longer required instead mpirun can be used to 
launch all the processes. This means the sbatch parameter ntasks-per-node can 
now be used to control the number of tasks to launch per node. The `np` 
argument will now be `$SLURM_NTASKS` instead of `$SLURM_JOB_NUM_NODES` and the 
torchrun will be emitted leaving new submission commands like the following 

Virtual Environment
`mpirun $mpiArgs python $TRAIN_SCRIPT $trainArgs`

Container
```
CONTAINER=horovod_latest.sif # The path to the container
apptainerArgs="
  --nv
  -B ${ML_DIST_ROOT}:${ML_DIST_ROOT}
  ${CONTAINER}
  python $TRAIN_SCRIPT $trainArgs
"
mpirun $mpiArgs apptainer run $apptainerArgs
```

#### Usage
The only difference to slurm parameters between horovod and pytorch ddp is 
the `--ntasks-per-node` SBATCH parameter is set to the number of GPUs per node 
when using horovod, as mpirun launches all processes.
