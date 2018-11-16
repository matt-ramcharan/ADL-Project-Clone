#!/bin/bash                                                                                                                                          
sbatch -t 0-02:00 # Runtime in D-HH:MM
sbatch -p gpu # Partition to submit to
sbatch --gres=gpu:1
sbatch --mem=15000
sbatch --account=comsm0018       # use the course account
sbatch -J  testing_tensorflow    # name
sbatch -o hostname_%j.out # File to which STDOUT will be written
sbatch -e hostname_%j.err # File to which STDERR will be written
sbatch --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
sbatch --mail-user=mr15009@bristol.ac.uk # Email to which notifications will be sent

module add libs/tensorflow/1.2

srun python cifar_hyperparam.py
wait
