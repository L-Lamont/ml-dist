# ml-dist
A collection of distributed and non-distributed ml examples and example 
jobscripts for running them on a slurm managed cluster using virtual 
environments and using containers. The README.md in each frameworks directory 
gives a more detailed overview of the changes required and the jobscript for 
each package.

## Modifying an example jobscript
- Update the sbatch parameters
- Update the modules loaded
- Update the paths specified
- Ensure $DATA exists
- Ensure the parent of $OUTPUT exists
- Add specific environment variables e.g. proxy variables
- Download required container or build required virtual environment

## Future Work
- Elastic training examples
- Model parallel examples
- More distribution strategies (accelerate etc.)
- More launching options (deepspeed)
- Better checkpointing and logging examples (weights & biases)
