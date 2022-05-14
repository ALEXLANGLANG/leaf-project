import numpy as np
from path import Path
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import os
import sys
sys.path.append('../')
import random
from tool.data_io import get_data_paths,read_image, read_json, save_image, try_create_dir,save_json
from tool.yaml_io import write_to_yaml,read_from_yaml
from tool.plt_utils import plt_samples
def set_labels(label_file):
    labels = read_json(label_file)
    label_ids = set()
    label_names = set()
    for species in labels.keys():
        for (idx, item) in labels[species].items():
            label_ids.add(item['number'])
            label_names.add(item['label'])
    return labels

phases = ['validation', 'testing']
dtypes = ['skel','hole'] #'margin',
species_types = ['ql'] #'qb', 'os', 
# iter_ = 20
for phase in phases:
    for d_type in dtypes:
        for species_type in species_types:
            print(species_type, d_type)
    # phase='validation' # validation testing
    # d_type= 'margin'
    # species_type = 'qb'

            save_=True
            save_root = Path("/usr/xtmp/xs75/leaves/datasets/vis/single_damge_single_species/")
    #         save_=False
            data_type = f'{species_type}_{d_type}' #'ep300_ql_sk' #f'e{epoch}_al05' #f'e{epoch}_s1r95p05_mask', 'ep1_a001_b11_beta2_mar01'
            partition = read_json(f'../../stats/partition_{species_type}_{d_type}.json')  #'../../stats/partition_os_hole.json'
            root_leaves = Path('/usr/xtmp/xs75/leaves/datasets/leaves')
            root_acts = Path('/usr/xtmp/xs75/leaves/datasets/acts/image')
            labels_map = set_labels(f'../../stats/label_{d_type}.json' )# ../../stats/label_file_common_binary.json' ../../stats/label_skel.json
            if save_:
                print(f'********saving to{species_type}_{d_type}********')
            root=root_acts
            pos_acts = []
            neg_acts = []
            id2paths = partition[phase]['images']
            targets= [] #['quercus-bicolor-herbivory/00000389_4249426','quercus-bicolor-herbivory/00000495_26585']
            # targets=['quercus-lobata/00000293_1619758','quercus-lobata/00000131_1314762',
            #          'quercus-lobata/00000181_1314822','quercus-lobata/00000302_1657887',
            #          'quercus-lobata/00000384_3819210','quercus-lobata/00000102_1299701']
            # targets =[ 'quercus-lobata/00000131_1314762']
            tot_tp = tot_fp = tot_fn = 0
            i = 0

            for path in tqdm(glob(root/phase/f'*/{data_type}/*.npy')):
                id_ = os.path.join(path.split('/')[-3:][0], (path.split('/')[-4:][-1]).split('.')[0] )
            #     if id_ not in targets:
            #         continue
                if id_ not in id2paths.keys(): 
                    continue
    #             if id_ not in qb_margin_samples_val:
    #                 continue
                print(id_)
                label, pl_label = read_image(root_leaves/id2paths[id_]['label'])
                image,_ = read_image(root_leaves/id2paths[id_]['image'])
                species,name = id_.split('/')
                for key, value in labels_map[species].items():
                    label[label == int(key)] = value['number']

                leaf_mask, _ = read_image(root_leaves/id2paths[id_]['leaf_mask'])
                act = np.load(path)
                pos_acts = (act[np.logical_and(label!=0,leaf_mask)])
                neg_acts = (act[np.logical_and(label==0,leaf_mask)])

                thres = -0
                index_neg_1 = np.logical_and(np.logical_and(label==0, act>=thres), leaf_mask)
                index_pos_1 = np.logical_and(np.logical_and(label!=0, act>=thres), leaf_mask)
                index_pos_0 = np.logical_and(np.logical_and(label!=0, act<thres),leaf_mask)

                tot_tp += np.count_nonzero(index_pos_1)
                tot_fp += np.count_nonzero(index_pos_0)
                tot_fn += np.count_nonzero(index_neg_1)

                img_red = image.copy()
                img_red[index_neg_1]=[255,0,0] #FP
            #     img_red[index_pos_1]=[255,0,0]
                img_red[index_pos_0]=[0,0,255] #FN
                img_red[index_pos_1]=[0,255,0] #TP
                if save_:
                    dir_ = save_root/Path(f'{data_type}/{phase}/{name}')
                    try_create_dir(dir_)
                    save_image(dir_/'img_original.jpeg', image)
                    save_image(dir_/'label_common.png', label, pl_label)
                    label, pl_label = read_image(root_leaves/id2paths[id_]['label'])
                    save_image(dir_/'label_original.png', label, pl_label)
                    save_image(dir_/'img_painted.jpeg', img_red)
                else:
                    plt_samples([image,label,img_red], 1,3,figsize=(20,30),cmap='gray')
                    plt.show()
    #             if i>iter_:
    #                 break
                i+=1