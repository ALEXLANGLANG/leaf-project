#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=s2
=======
#SBATCH --job-name=t3.0
>>>>>>> new-name
#SBATCH -t 14-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=65g
#SBATCH --partition=compsci-gpu
#SBATCH --constraint=p100
source /usr/project/xtmp/xs75/anaconda3/etc/profile.d/conda.sh
<<<<<<< HEAD
conda activate py3
cd /home/users/xs75/Xian/leaves/code/run
python3 train.py -c='../../configs/train_2stage.yml' -n=s2_al0.99b1ta2 


=======
conda activate leaf
cd /home/users/xs75/Xian/leaf-damage-new/leaves/code/run
# python3 train.py -c='../../configs/train_img_separate_hole.yml' -n=qb_hole_img_nonWhite
python3 train.py -c='../../configs/train_tile.yml' -n=tile_3.0_train_with_testing_validation
# python3 train.py -c='../../configs/train_2stage.yml' -n=os_hole
# python3 train.py -r='/usr/xtmp/xs75/leaves/exps/US_shift_rot30_al0.9g2_20211202_205648'
>>>>>>> new-name
"$@"
