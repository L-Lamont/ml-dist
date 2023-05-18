# ml-dist
A collection of distributed and non-distributed ml examples and an example 
jobscript for running them on a slurm managed cluster.

## Creating a virtual environment
- Use the scripts/venv-build.job to create the required venv
  - Before use update the SBATCH parameters (it is important to build the venv on a compute node)
  - nccl_path must be set if --horovod is being used
  - mxnet examples require mxnet 2

## Modifying an example jobscript
- Update the sbatch parameters
- Update the modules loaded
- Update the paths specified
- Ensure $DATA exists
- Ensure the parent of $OUTPUT exists
- Add specific environment variables e.g. proxy variables
