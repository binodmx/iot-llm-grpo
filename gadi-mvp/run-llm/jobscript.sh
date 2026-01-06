#!/bin/bash

#PBS -P wd04
#PBS -q dgxa100
#PBS -l ncpus=16
#PBS -l ngpus=1
#PBS -l mem=256GB
#PBS -l jobfs=1GB
#PBS -l walltime=00:30:00
#PBS -l storage=scratch/wd04
#PBS -l wd
#PBS -M s4025371@student.rmit.edu.au
#PBS -m be
#PBS -o /scratch/wd04/bk2508/repositories/iot-llm-grpo/gadi/run-llm/logs/
#PBS -e /scratch/wd04/bk2508/repositories/iot-llm-grpo/gadi/run-llm/logs/

module load python3/3.10.4
source /scratch/wd04/bk2508/venvs/llm-env/bin/activate
python3 main.py $PBS_JOBID "unsloth/gemma-3-4b-it" > /scratch/wd04/bk2508/repositories/iot-llm-grpo/gadi/run-llm/logs/$PBS_JOBID.log
deactivate
