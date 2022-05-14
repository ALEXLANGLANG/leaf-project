#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=vis_act
=======
#SBATCH --job-name=vis3.0
>>>>>>> new-name
#SBATCH -t 14-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=90g
#SBATCH --partition=compsci-gpu
#SBATCH --constraint=p100
source /usr/project/xtmp/xs75/anaconda3/etc/profile.d/conda.sh
<<<<<<< HEAD
conda activate py3
cd /home/users/xs75/Xian/leaves/code/run
python3 visualize.py -c='../../configs/visualize.yml'  


=======
conda activate leaf
cd /home/users/xs75/Xian/leaf-damage-new/leaves/code/run
python3 visualize.py -c='../../configs/visualize_tile.yml'  
>>>>>>> new-name
"$@"
