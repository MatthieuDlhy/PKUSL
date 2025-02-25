# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:07:21 2022

@author: Lenovo
"""
import pandas as pd
import numpy as np
import torch

class Change_Knowledge_Variable():
    def __init__(self,input_):
        '''
        input_type : DataFrame
        input_size : [d, k]
             d : longueur de la série temporelle
             k : nombre de variables de connaissance
        Signification des variables de connaissance :
            potVolt          : tension du creuset (pot voltage)
            filterResist     : résistance de filtrage
            smoothResist     : résistance de lissage
            slopeData        : pente
            sumSlopeData     : pente cumulée
            fluctDelta       : vibration type "aiguille" (variation de fluctuation)
            wavingDelta      : oscillation (ou balancement)
            settingVoltMax   : tension réglée maximale
            settingVoltMin   : tension réglée minimale
            anodeChangeToNow : temps écoulé depuis le dernier changement d'anode
        '''
        self.input_=input_
        self.output_=pd.DataFrame({'feature0':0,'feature1':0},index=[0])
        
    def change_knowledge_variable(self):
        # feature0 : si le maximum de "fluctDelta" dépasse 15, on renvoie 0.8, sinon 0.2
        self.output_['feature0'] = np.where(self.input_['fluctDelta'].max() > 15, 0.8, 0.2)
        
        # Alternative commentée : on pourrait aussi définir feature1 selon le dépassement de 15 (0.1 vs 0.8)
        # self.output_['feature1'] = np.where(self.input_['fluctDelta'].max() > 15, 0.1, 0.8)
        
        # Alternative commentée : on pourrait simplement utiliser le maximum de "fluctDelta"
        # self.output_['feature0'] = self.input_['fluctDelta'].max()
        
        # Alternative commentée : on pourrait définir feature3 comme 1 si le maximum de "wavingDelta" dépasse 6, sinon 0
        # self.output_['feature3'] = np.where(self.input_['wavingDelta'].max() > 6, 1, 0)
        
        # feature1 : le maximum de "wavingDelta"
        self.output_['feature1'] = self.input_['wavingDelta'].max()
        
        # feature2 : la variance de "sumSlopeData" (pente cumulée)
        self.output_['feature2'] = self.input_['sumSlopeData'].var()
        
        # feature3 : la variance de "slopeData" (pente)
        self.output_['feature3'] = self.input_['slopeData'].var()
        
        # Alternative commentée : on pourrait définir feature2 comme le rapport du maximum de "potVolt" sur sa variance
        # self.output_['feature2'] = self.input_['potVolt'].max() / self.input_['potVolt'].var()
        return self.output_

        
    