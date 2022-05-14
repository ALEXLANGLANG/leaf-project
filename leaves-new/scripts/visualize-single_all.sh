#!/bin/bash
#SBATCH --job-name=vis_act
#SBATCH -t 14-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=90g
#SBATCH --partition=compsci-gpu
#SBATCH --constraint=p100
source /usr/project/xtmp/xs75/anaconda3/etc/profile.d/conda.sh
conda activate leaf
cd /home/users/xs75/Xian/leaf-damage-new/leaves/code/run
# os_hole os_margin os_skel qb_hole qb_margin ql_hole ql_margin ql_skel
for name in qb_skel;
do python3 visualize.py -c="../../configs/single_damage_single_species/visualize_$name.yml";
done;
"$@"
