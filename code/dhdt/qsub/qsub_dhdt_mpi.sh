#!/bin/sh
# simple shell script to be used with SGE for launching dhdt

# this script is specific to eddie3.ecdf.ed.ac.uk

# tell SGE to change into current working directory and use current 
# environment
#$-cwd

. /etc/profile.d/modules.sh

module load anaconda
module load openmpi
#source activate /exports/eddie/scratch/mhagdorn/test2
source activate /exports/csce/eddie/geos/groups/geos_EO/cryotop/dhdtPy/python/dhdtPy

CMD="$*"

mpirun -n $NSLOTS $CMD

