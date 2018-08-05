#!/bin/sh
# simple shell script to be used with SGE for launching dhdt

# this script is specific to eddie3.ecdf.ed.ac.uk

# tell SGE to change into current working directory and use current 
# environment
#$-cwd

. /etc/profile.d/modules.sh

module load anaconda
#source activate /exports/eddie/scratch/mhagdorn/test2
source activate /exports/csce/eddie/geos/groups/geos_EO/cryotop/dhdtPy/python/dhdtPy

# check if we are an array job, ie when SGE_TASK_ID is not undefined
if [ -z "$SGE_TASK_ID" -o x"$SGE_TASK_ID"x = xundefinedx ]; then
CMD="$*"
else
let p=SGE_TASK_ID-1
CMD="$* -p $p"
fi

$CMD
