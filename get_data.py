# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:05:30 2022

@author: Lenovo
"""

from os import path
from numpy import genfromtxt
from utils import normalize_data,normalize_dim2,Znormalize_dim2
from knowledge_variable_deal import Change_Knowledge_Variable
from sktime.datasets._readers_writers.ts import load_from_tsfile
import os
import pandas as pd
import numpy as np
import torch


### Revoir la normalisation : quelle méthode ? bien adaptée à nos données ?

class get_data_ucr():
    print("get_data_ucr() running... ")
    def __init__(self, ucr_dataset_name, ucr_dataset_base_folder):
        self.ucr_dataset_name = ucr_dataset_name
        self.ucr_dataset_base_folder = ucr_dataset_base_folder

    def load_ts_file(self, file_path):
        X, y = load_from_tsfile(file_path, replace_missing_vals_with='NaN', return_y=True,
                                return_data_type='numpy3d', encoding='utf-8')
        y = np.array(y, dtype=int)
        return X, y

    def load_dataset(self):
        dataset_path = os.path.join(self.ucr_dataset_base_folder, self.ucr_dataset_name)
        train_file_path = os.path.join(dataset_path, '{}_TRAIN.ts'.format(self.ucr_dataset_name))
        test_file_path = os.path.join(dataset_path, '{}_TEST.ts'.format(self.ucr_dataset_name))

        # Si les fichiers sont au format .ts, on utilise load_from_tsfile
        if train_file_path.endswith('.ts'):
            print("train_file_path :", train_file_path)
            print("test_file_path :", test_file_path)
            X_train, y_train = self.load_ts_file(train_file_path)
            X_test, y_test = self.load_ts_file(test_file_path)
        else:
            # Code existant pour CSV (Github initial)
            train_raw_arr = genfromtxt(train_file_path, delimiter=',')
            X_train = train_raw_arr[:, 1:]
            y_train = train_raw_arr[:, 0] - 1
            test_raw_arr = genfromtxt(test_file_path, delimiter=',')
            X_test = test_raw_arr[:, 1:]
            y_test = test_raw_arr[:, 0] - 1
        
        return X_train, y_train, X_test, y_test
    
    def main(self):
        X_train, y_train, X_test, y_test = self.load_dataset()
        # #print(X_train.shape,type(X_train),y_train.shape,type(y_train),'X_train.shape,type(X_train),y_train.shape,type(y_train)')
        # y_train=y_train-min(y_train)
        # X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1])
        # X_test=X_test.reshape(X_test.shape[0],1,X_test.shape[1])
        # y_test=y_test-min(y_test)
        X_train, scaler = normalize_data(X_train)
        X_test, _ = normalize_data(X_test,scaler)
        return X_train, y_train, X_test, y_test


# class get_data_ucr():
#     def __init__(self, ucr_dataset_name,ucr_dataset_base_folder):
#         self.ucr_dataset_name=ucr_dataset_name
#         self.ucr_dataset_base_folder=ucr_dataset_base_folder


#     def load_dataset(self):
#         #if self.ucr_dataset_base_floder!='zhonglv':
#         dataset_path = path.join(self.ucr_dataset_base_folder, self.ucr_dataset_name)
#         train_file_path = path.join(dataset_path, '{}_TRAIN'.format(self.ucr_dataset_name))
#         test_file_path = path.join(dataset_path, '{}_TEST'.format(self.ucr_dataset_name))
#         #print('train_file_path',train_file_path)
        
#         # training data
#         train_raw_arr = genfromtxt(train_file_path, delimiter=',')
#         train_data = train_raw_arr[:, 1:]
#         train_labels = train_raw_arr[:, 0] - 1
#         # one was subtracted to change the labels to 0 and 1 instead of 1 and 2
        
#         # test_data
#         test_raw_arr = genfromtxt(test_file_path, delimiter=',')
#         test_data = test_raw_arr[:, 1:]
#         test_labels = test_raw_arr[:, 0] - 1
#         #else:
#         #print(train_data.shape,train_labels.shape,test_data.shape,test_labels.shape,'*******************')
#         return train_data, train_labels, test_data, test_labels
    
#     def main(self):
#         X_train, y_train, X_test, y_test = self.load_dataset()
#         #print(X_train.shape,type(X_train),y_train.shape,type(y_train),'X_train.shape,type(X_train),y_train.shape,type(y_train)')
#         y_train=y_train-min(y_train)
#         X_train=X_train.reshape(X_train.shape[0],1,X_train.shape[1])
#         X_test=X_test.reshape(X_test.shape[0],1,X_test.shape[1])
#         y_test=y_test-min(y_test)
#         X_train, scaler = normalize_data(X_train)
#         X_test,_=normalize_data(X_test,scaler)
#         return X_train,y_train,X_test,y_test
    
class get_data_ACS():
    print("get_data_ACS running... ")
    # def __init__(self,path_total):
    #     """
    #     Args:
    #         slef.path_total:输入数据路径
    #         self.path_adj:邻接矩阵路径
    #     """
    #     self.path_total=path_total

    def __init__(self, ucr_dataset_name, ucr_dataset_base_folder):
        self.ucr_dataset_name = ucr_dataset_name
        self.ucr_dataset_base_folder = ucr_dataset_base_folder
        
    # def csv_to_excel(self,path_total):
    #     """
    #     csv_to_excel:
    #         将数据中csv的数据转换为excel
    #     Args:
    #         path_total:包含所有文件路径的列表
    #     """
    #     for i in path_total:
    #         if i[-4:]=='.csv':
    #             csv=pd.read_csv(i,encoding='utf-8',header=None,engine='python')
    #             csv.to_excel(i.replace('.csv','.xlsx'),encoding='utf-8')
    #             os.remove(i)
    #     return 
    
    # def del_excess_columns_indexs(self,path_total):
    #     """
    #     del_excess_columns_indexs:
    #         如果数据包含多余的行和列就将多余的行列删除，并写入新的excel中
    #     Args:
    #         path_total:包含所有文件路径的列表
    #     """
    #     for i in path_total:
    #         if i[-4:]=='.csv':
    #             csv=pd.read_csv(i,header=None,engine='python')
    #         else:
    #             csv=pd.read_excel(i,header=None)
    #         if pd.isnull(csv.iloc[0,0]):
    #             print(i)
    #             csv=csv.drop(csv.index[[0]])
    #             csv=csv.drop(csv.columns[[0]],axis=1)
    #             csv.to_excel(i,header=None,index=False)
    #     return
    
    # def data_label(self,files):
    #     """
    #     data_label:
    #         获取数据和标签
    #     Args:
    #         total_path:用于记录所有文件的路径
    #         label_y:用于记录每个文件的标签
    #     """
    #     label_y=[]
    #     total_path=[]
    #     for filenames,dirnames,files in os.walk(files):
    #         for name in files:
    #             total_path+=[filenames+'/'+name]
    #             label_y.append(int(filenames[-1]))
    #     return total_path,label_y
    
    # def change_data_zhong_lv(self):
    #     """
    #     change_data：
    #          对数据进行处理,将csv数据转为excel数据,删除出现多余第一行和第一列的数据
    #     Args:
    #         total_path_：处理完之后的数据的路径列表
    #         y_total：所有的标签列表
    #     """
    #     #total_path_,y_total=self.data_label(self.path_total)
    #     #self.csv_to_excel(total_path_)
    #     total_path_,y_total=self.data_label(self.path_total)
    #     self.del_excess_columns_indexs(total_path_)
    #     return total_path_,y_total
    
    # def change_data_total_all(self):
    #     """
    #     change_data：
    #          对数据进行处理,将csv数据转为excel数据,删除出现多余第一行和第一列的数据
    #     Args:
    #         total_path_：处理完之后的数据的路径列表
    #         y_total：所有的标签列表
    #     """
    #     total_path_,y_total=self.data_label(self.path_total)
    #     self.csv_to_excel(total_path_)
    #     total_path_,y_train=self.data_label(self.path_total)
    #     self.del_excess_columns_indexs(total_path_)
    #     return total_path_,y_total
    
    # def preprocessing_standard0(self,L):
    #     """
    #        preprocessing_standard: 对一行数据进行数据标准化,L[i]=(L[i]-L.min)/(L.max-L.min)
    #     """
    #     datamax=max(L)
    #     datamin=min(L)
    #     L1=[]
    #     for index,row in enumerate(L):
    #         if datamax-datamin!=0:
    #             m=(row-datamin)/float((datamax-datamin))
    #             L1.append(round(m,4))
    #         else:
    #             L1.append(0)
    #    #        matlabshow(row,index=str(index)+'_')    
    #     return L1
    
    # def preprocess0(self,dataframe):
    #     dataframe_new=pd.DataFrame(columns=dataframe.columns)
    #     for i in dataframe.columns:
    #         dataframe_new[i]=self.preprocessing_standard0(dataframe[i])
    #         #dataframe_new[i]=pywt_pro_0(list(dataframe_new[i]))
    #     return dataframe_new
    

    def load_ts_file(self, file_path):
        X, y = load_from_tsfile(file_path, replace_missing_vals_with='NaN', return_y=True,
                                return_data_type='numpy3d', encoding='utf-8')
        y = np.array(y, dtype=int)
        return X, y

    def load_dataset(self):
        dataset_path = os.path.join(self.ucr_dataset_base_folder, self.ucr_dataset_name)
        train_file_path = os.path.join(dataset_path, '{}_TRAIN.ts'.format(self.ucr_dataset_name))
        test_file_path = os.path.join(dataset_path, '{}_TEST.ts'.format(self.ucr_dataset_name))

        # Si les fichiers sont au format .ts, on utilise load_from_tsfile
        # if train_file_path.endswith('.ts'):
        print("train_file_path :", train_file_path)
        print("test_file_path :", test_file_path)
        X_train, y_train = self.load_ts_file(train_file_path)
        X_test, y_test = self.load_ts_file(test_file_path)
        # else:
        #     # Code existant pour CSV (Github initial)
        #     train_raw_arr = genfromtxt(train_file_path, delimiter=',')
        #     X_train = train_raw_arr[:, 1:]
        #     y_train = train_raw_arr[:, 0] - 1
        #     test_raw_arr = genfromtxt(test_file_path, delimiter=',')
        #     X_test = test_raw_arr[:, 1:]
        #     y_test = test_raw_arr[:, 0] - 1
        
        return X_train, y_train, X_test, y_test
    
    
    def main(self):
        # if self.path_total==r'E:\code\Classification_ACS_20211027\shapelet_learning_ACS\data\zhong_lv': 
        #     print("Case 1")
        #     total_path_,y_total=self.change_data_zhong_lv()
        #     total=np.array([np.array(self.preprocess0(pd.read_csv(total_path,header=None,engine='c')).T) for total_path in total_path_])
        # if 'volt' in self.path_total:
        # print("self.path_total : ", self.path_total)
        X_train, y_train, X_test, y_test = self.load_dataset()
        X_train, scaler = normalize_data(X_train)
        X_test, _ = normalize_data(X_test,scaler)
        # return X_train, y_train, X_test, y_test
        knowledge_bool = True
        if knowledge_bool:
            print("Case 2.1")
            # total_path_,y_total=self.change_data_total_all()
            # total=np.array([np.array(self.preprocess0(pd.read_excel(total_path)[['potVolt']]).T) for total_path in total_path_])
            # knowledge=np.array([np.array(Change_Knowledge_Variable(pd.read_excel(total_path)).change_knowledge_variable()) for total_path in total_path_])
            knowledge_train=pd.read_csv("/home/administrateur/Documents/Code/Times series models exploration/PKUSL/data/ACSF1/knowledge_features_train.csv")
            # knowledge_test=pd.read_csv("/home/administrateur/Documents/Code/Times series models exploration/PKUSL/data/ACSF1/knowledge_features_test.csv")
        else:
            print("Case 2.2")
            # total_path_,y_total=self.change_data_total_all()
            # total=np.array([np.array(self.preprocess0(pd.read_excel(total_path,header=None)).T) for total_path in total_path_])
        # else:
        #     print("Case 3")
        #     total_path_,y_total=self.change_data_total_all()
        #     #total=np.array([np.array(self.preprocess0(pd.read_excel(total_path,header=None))[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]].T) for total_path in total_path_])
        #     total=np.array([np.array(pd.read_excel(total_path,header=None)[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]].T) for total_path in total_path_])
        #将顺序打乱

        ##### Bien vérifier si shuffle fait correspondre même index entre X et knowledge (train/test)

        index_train=[i for i in range(len(y_train))]
        np.random.shuffle(index_train)

        X_train=X_train[index_train]
        knowledge=knowledge_train.iloc[index_train]
        # train_num=int(total.shape[0]*0.8)

        # index_test=[i for i in range(len(y_test))]
        # np.random.shuffle(index_test)

        # X_test_shuff=X_test[index_test]
        # knowledge_test=knowledge_test[index_test]
        
        # print("X_train : ", X_train)
        # print("knowledge : ", knowledge)
        
        # X_train=total[:train_num,:,:]
        # X_test=total[train_num:,:,:]
        # if 'knowledge' in self.path_total:
        if knowledge_bool:
            # knowledge=knowledge[:train_num,:]
            knowledge = torch.tensor(knowledge.to_numpy())
            knowledge = knowledge.float()
            if len(knowledge.shape) >2:
                knowledge = knowledge.squeeze(1)
        # knowledge = normalize_dim2(knowledge, 1)
            knowledge = Znormalize_dim2(knowledge, 1)
            
        #数据标签 
        # y_total=np.array(y_total)
        # y_total=y_total[index]
        # y_train=y_total[:train_num]
        # y_test=y_total[train_num:]
        #X_train, scaler = normalize_data(X_train)
        #X_test,_=normalize_data(X_test,scaler)
        if knowledge_bool:
            return X_train,y_train,X_test,y_test,knowledge
        else:
            return X_train,y_train,X_test,y_test
    



    # def main(self):
    #     if self.path_total==r'E:\code\Classification_ACS_20211027\shapelet_learning_ACS\data\zhong_lv': 
    #         print("Case 1")
    #         total_path_,y_total=self.change_data_zhong_lv()
    #         total=np.array([np.array(self.preprocess0(pd.read_csv(total_path,header=None,engine='c')).T) for total_path in total_path_])
    #     elif 'volt' in self.path_total:
    #         print("self.path_total : ", self.path_total)
    #         if 'knowledge' in self.path_total:
    #             print("Case 2.1")
    #             # total_path_,y_total=self.change_data_total_all()
    #             # total=np.array([np.array(self.preprocess0(pd.read_excel(total_path)[['potVolt']]).T) for total_path in total_path_])
    #             knowledge=np.array([np.array(Change_Knowledge_Variable(pd.read_excel(total_path)).change_knowledge_variable()) for total_path in total_path_])
    #         else:
    #             print("Case 2.2")
    #             total_path_,y_total=self.change_data_total_all()
    #             total=np.array([np.array(self.preprocess0(pd.read_excel(total_path,header=None)).T) for total_path in total_path_])
    #     else:
    #         print("Case 3")
    #         total_path_,y_total=self.change_data_total_all()
    #         #total=np.array([np.array(self.preprocess0(pd.read_excel(total_path,header=None))[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]].T) for total_path in total_path_])
    #         total=np.array([np.array(pd.read_excel(total_path,header=None)[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]].T) for total_path in total_path_])
    #     #将顺序打乱
    #     index=[i for i in range(len(total))]
    #     np.random.shuffle(index)

    #     total=total[index]
    #     knowledge=knowledge[index]
    #     train_num=int(total.shape[0]*0.8)
        
    #     print("total : ", total)
    #     print("knowledge : ", knowledge)
        
    #     X_train=total[:train_num,:,:]
    #     X_test=total[train_num:,:,:]
    #     if 'knowledge' in self.path_total:
    #         knowledge=knowledge[:train_num,:]
    #         knowledge = torch.tensor(knowledge)
    #         knowledge = knowledge.float()
    #         if len(knowledge.shape) >2:
    #             knowledge = knowledge.squeeze(1)
    #        # knowledge = normalize_dim2(knowledge, 1)
    #         knowledge = Znormalize_dim2(knowledge, 1)
            
    #     #数据标签 
    #     y_total=np.array(y_total)
    #     y_total=y_total[index]
    #     y_train=y_total[:train_num]
    #     y_test=y_total[train_num:]
    #     #X_train, scaler = normalize_data(X_train)
    #     #X_test,_=normalize_data(X_test,scaler)
    #     if 'knowledge' in self.path_total:
    #         return X_train,y_train,X_test,y_test,knowledge
    #     else:
    #         return X_train,y_train,X_test,y_test

    
    
    