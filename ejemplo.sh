#!/bin/sh

## Request n processor on m node 
#PBS -l nodes=1:ppn=1

## Request time hh:mm:ss of walltime
#PBS -l walltime=10000:00:00

## Request 8 gigabyte of memory per process
## PBS -l pmem=8gb

## set name of job
#PBS -N jchen

## Mail alert at (b)eginning, (e)nd and (a)bortion of execution
#PBS -m bea

## Send mail to the following address
#PBS -M jchen168@hawk.iit.edu

## Use submission environment
#PBS -V

## Output files are given by cron. No need to use these.
##PBS -e job.err
##PBS -o job.out


####################################
# Starting from here put any code that you want to run
cd /home/jchen168/heat
# /share/apps/matlab/R2017a/bin/matlab -nodisplay -nosplash -r advection_dispersion
# /share/apps/matlab/R2017a/bin/matlab -nodisplay -nosplash -r space_filling
/share/apps/matlab/R2017a/bin/matlab -nodisplay -nosplash -r D_optimal
