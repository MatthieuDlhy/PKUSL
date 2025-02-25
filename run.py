# -*- coding: utf-8 -*-
"""
Créé le : Dimanche 3 avril 2022, 19:45:49

@author: Lenovo
"""
'''
22/04/2022 : Ajout de l'entraînement sur des échantillons de données de courant anodique de Chalco (Chinalco).
'''

from main import Main
from sklearn.model_selection import GridSearchCV
from get_data import get_data_ucr, get_data_ACS

# Signal de courant
# path_zhonglv = r'E:\Python\Classification_ACS_20211027\shapelet_learning_ACS\data\zhong_lv'
# path_yangxin = r'E:\Python\Classification_ACS_20211027\shapelet_learning_ACS\data\total_all_origin'
path_zhonglv8310 = r'E:\Code\Data\2min_AE_zhonglv\ACS\yu_zhong_lv8310'
path_zhonglv8311 = r'E:\Code\Data\2min_AE_zhonglv\ACS\yu_zhong_lv8311'
path_zhonglv8312 = r'E:\Code\Data\2min_AE_zhonglv\ACS\yu_zhong_lv8312'
path_zhonglv8331 = r'E:\Code\Data\2min_AE_zhonglv\ACS\yu_zhong_lv8331'
path_zhonglv8331min = r'E:\Code\Data\2min_AE_zhonglv\ACS\yu_zhong_lv8331_min'  # Petit échantillon
path_zhonglv8331_20_5 = r'E:\Code\Data\2min_AE_zhonglv\ACS\zhong_lv8331_20__5'
path_zhonglv8331_20_5_multi_class = r'E:\Python\Data\2min_AE_zhonglv\ACS\zhong_lv8331_20__5_multi_class_normal_100_100'

# Signal de tension
path_zhonglv8331_volt_20_5 = r'E:\code\Data\2min_AE_zhonglv\volt\zhong_lv8331_volt'
path_zhonglv_knowledge = r'E:\code\Data\2min_AE_zhonglv\volt\knowledge\FAE_SAE_nor'  # Contient des connaissances
path_zhonglv_knowledge2 = r'E:\code\Data\2min_AE_zhonglv\volt\knowledge\FAE_nor'    # Contient des connaissances
path_zhonglv_knowledge3 = r'E:\code\Data\2min_AE_zhonglv\volt\knowledge\AE_FAE_SAE_nor'  # Considérer AE comme SAE

path_folder_ACSF1 = '/home/administrateur/Documents/Code/Times series models exploration/PKUSL/data/'#ACSF1_TRAIN.ts'
'''
path : chemin, path_zhonglv
q : plus q est petit, plus la tendance à contenir des connaissances est élevée.
'''

main = Main(ucr_dataset_name='ACSF1',
            # ucr_dataset_base_folder=path_zhonglv_knowledge,
            ucr_dataset_base_folder=path_folder_ACSF1,
            K=0.15,
            Lmin=0.3,
            learning_rate=0.1,
            epoch=10,
            batch_size=250,
            t1=0.001,
            t2=200,
            t3=40,
            t4=0.01,
            shapelet_epoch=5,
            R=1,
            q=0.2,
            show_visualization=False)

record_data_plot = main.train_model()
