# -*- coding: utf-8 -*-
"""
Created on Mon Jun  13  2018

@author: hubingqing
"""
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.preprocessing
import pandas as pd
from class_optimizer import one_sample, get_chunk
import sys

# import stocks daily excess return
excess, _ = pd.read_pickle('clean_data.pkl')

# import 000905 components
weight0 = pd.read_pickle('000905_weight.pkl')
num_comp = sum(weight0.iloc[0]>0)

# citic_1
industry_name, industry = pd.read_pickle('citic_1.pkl')
#print(industry_name)

# import BP value
factor_val = pd.read_pickle('BP.pkl')
# some data cleaning
factor_BP = np.array(factor_val.fillna(0))
factor_BP = sklearn.preprocessing.scale(factor_BP,axis=1)
factor_BP = pd.DataFrame(factor_BP,index=factor_val.index,columns=factor_val.columns)

# import return_1m
factor_val = pd.read_pickle('return_1m.pkl')
# some data cleaning
factor_return = -1 * np.array(factor_val.fillna(0))
factor_return[factor_return>100]=0
factor_return = sklearn.preprocessing.scale(factor_return,axis=1)
factor_return = pd.DataFrame(factor_return,index=factor_val.index,columns=factor_val.columns)

if __name__ == '__main__':

    #log_file = open("result_2017.log",'w')
    #sys.stdout = log_file

    BP_list = [-10]
    for BP_min in BP_list:
        print(BP_min)
        arg_dict = {'num_final_stocks': 50, 'te_limit': 0.08, 'industry_deviation': 0.01, 'turn_over_limit': 0.5, \
        'weight_deviation': 0.03, 'weight_max': 0.1, 'BP_min': BP_min}
        timesteps = 100
        future_time = 20
        time_point = list(excess.index).index(20110105) - timesteps
        ex_return_list = [0]
        l2 = 0
        fe_sum = 0
        while time_point < list(excess.index).index(20111207) - timesteps:
            net = one_sample(num_input = num_comp, timesteps = timesteps, time_point = time_point, future_time = future_time, \
            learning_rate = 0.00001, training_steps = 2000, display_step = 100, num_industry = len(industry_name), **arg_dict)
            net.define_graph(weight_dim=num_comp)
            print(excess.index[time_point+timesteps])
            weight0,latest_weight,training_x,test_y,factor_BP,factor_return,industry_class,index_industry,stocks = net.run()
            net.define_graph(weight_dim=arg_dict['num_final_stocks'])
            net.second_run(weight0,latest_weight,training_x,test_y,factor_BP,factor_return,industry_class,index_industry,stocks)
            print("This month excess return: "+str(net.ex_return_end))
            ex_return_list.append(net.ex_return_end+ex_return_list[-1])
            l2 = l2 + net.l2_end
            fe_sum = fe_sum + net.fe_end
            time_point = time_point + future_time
        
        ex = ex_return_list[-1]
        te = math.sqrt(l2)
        IR = ex/te
        fe_ave = fe_sum/12
        print("TE: " + str(te)[0:6] + ", EX: " + str(ex)[0:6] + ", IR: " + str(IR)[0:5] + ", fe_ave: " + str(fe_ave)[0:5])
        
        plt.figure()
        plt.plot(ex_return_list)
        output = 'back_test_BP_min='+str(BP_min)+'.jpg'
        plt.savefig(output)
    
    #log_file.close()