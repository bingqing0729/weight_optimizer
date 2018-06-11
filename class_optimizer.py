# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:38:45 2018

@author: hubingqing
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.preprocessing
import pandas as pd

# import stocks daily excess return
pkl_file = open('clean_data.pkl','rb')
excess, _ = pickle.load(pkl_file)

# import 000905 components
pkl_file = open('000905_weight.pkl','rb')
weight0 = pickle.load(pkl_file)
num_comp = sum(weight0.iloc[0]>0)

# citic_1
pkl_file = open('citic_1.pkl','rb')
industry_name, industry = pickle.load(pkl_file)
#print(industry_name)

# import BP value
pkl_file = open('BP.pkl','rb')
factor_val = pickle.load(pkl_file)
# some data cleaning
factor_BP = np.array(factor_val.fillna(0))
factor_BP = sklearn.preprocessing.scale(factor_BP,axis=1)
factor_BP = pd.DataFrame(factor_BP,index=factor_val.index,columns=factor_val.columns)

#
pkl_file = open('return_1m.pkl','rb')
factor_val = pickle.load(pkl_file)
# some data cleaning
factor_return = np.array(factor_val.fillna(0))
factor_return[factor_return<-100]=0
factor_return = sklearn.preprocessing.scale(factor_return,axis=1)
factor_return = pd.DataFrame(factor_return,index=factor_val.index,columns=factor_val.columns)

# get data samples (when n >1, can get multiple samples)
def get_chunk(timesteps,num_input,n=1,factor_BP=factor_BP,factor_return=factor_return):
    x = np.zeros([timesteps,num_input])
    i = 0
    while i < 1:
        # pick a day
        samples = np.random.randint(0,len(list(excess.index[0:-timesteps])),1)
        # current time point of index
        current_comp_index = len(weight0.index[weight0.index<excess.index[samples+timesteps][0]])-1
        # components and weights
        comp = weight0.iloc[current_comp_index,:][weight0.iloc[current_comp_index,:]>0]
        # industry distribution
        current_industry = industry.iloc[current_comp_index,:]
        index_industry = [0]*len(industry_name)

        for j in range(0,len(industry_name)):
            current_j_industry = current_industry.loc[current_industry==j].index
            index_industry[j] = sum(comp[current_j_industry].dropna())
        index_industry = np.array(index_industry)/sum(index_industry)

        # pick num_input stocks(loc)
        stocks = random.sample(range(0,num_comp),num_input)
        # stocks and weights
        stocks = comp.index[stocks]
        weights = np.array(list(comp[stocks]))/100.0
        # history excess return
        x = excess.loc[excess.index[samples][0]:excess.index[samples+timesteps-1][0],stocks]
        # factor
        factor_BP = factor_BP.loc[excess.index[samples+timesteps-1][0],stocks]
        factor_return = factor_return.loc[excess.index[samples+timesteps-1][0],stocks]

        #if sum(sum(np.isnan(x)))>0:
            #i = i-1
        i = i+1

    return(x,weights,np.array(factor_BP),np.array(factor_return),np.array(current_industry[stocks]).astype('int32'),index_industry,stocks)

#
class one_sample:

    def __init__(self, num_input, timesteps, learning_rate, \
    training_steps,display_step, num_industry, **arg):
        self.name = one_sample
        self.num_input = num_input
        self.num_final_stocks = arg['num_final_stocks']
        self.te_limit = arg['te_limit']
        self.industry_deviation = arg['industry_deviation']
        self.turn_over_limit = arg['turn_over_limit']
        self.weight_deviation = arg['weight_deviation']
        self.weight_max = arg['weight_max']
        self.return_max = arg['return_max']
        self.num_industry = num_industry
        self.timesteps = timesteps
        self.display_step = display_step
        
        # Hyper parameters
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        
        # Graph related
        self.graph = tf.Graph()


    def define_graph(self,weight_dim):

        with self.graph.as_default():

            self.tf_train_samples = tf.placeholder("float",None)
            self.latest_weight = tf.placeholder("float",None)
            self.factor_BP = tf.placeholder('float',None)
            self.factor_return = tf.placeholder('float',None)
            self.index_industry = tf.placeholder('float',None)
            self.industry_class = tf.placeholder('int32',None)
        
            weights = tf.Variable(tf.random_normal([1,weight_dim],0,0.01))
            
            def model():
                
                return weights + tf.expand_dims(self.latest_weight,0) 
                
            # calculate loss
            def cal_loss(output,input_samples,factor_BP,factor_return):
                
                self.prediction = tf.abs(output)
                # the output stock weights after normalization
                self.prediction = tf.expand_dims(tf.divide(self.prediction,tf.expand_dims( \
                tf.reduce_sum(self.prediction,1),1)),1)
                self.prediction_final = tf.reduce_sum(tf.multiply(tf.expand_dims(input_samples,1),self.prediction),2)
                # factor_BP
                factor_BP = tf.expand_dims(factor_BP,0)
                self.fe_BP = tf.reduce_sum(tf.reduce_sum(tf.multiply( \
                tf.expand_dims(factor_BP,0),self.prediction),2),1)
                # factor return_1m
                factor_return = tf.expand_dims(factor_return,0)
                self.fe_return = tf.reduce_sum(tf.reduce_sum(tf.multiply( \
                tf.expand_dims(factor_return,0),self.prediction),2),1)
                # tracking error
                self.te = tf.sqrt(tf.nn.l2_loss(self.prediction_final)*2/self.timesteps*250)
                # industry proportion
                one_hot = tf.one_hot(self.industry_class,self.num_industry,1,0)
                self.real_industry_deviation = tf.reduce_mean(tf.abs(tf.subtract(tf.matmul( \
                self.prediction[0],tf.cast(one_hot,tf.float32)),self.index_industry)))
                real_weight_deviation = tf.reduce_mean(tf.abs(tf.subtract(self.prediction[0][0],self.latest_weight)))
                self.turn_over = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(self.prediction[0][0], \
                self.latest_weight)),self.latest_weight))
                loss = -1000*self.fe_BP + 100000*tf.nn.relu(self.te-self.te_limit) + \
                100000*tf.nn.relu(self.real_industry_deviation-self.industry_deviation) + \
                100000*tf.nn.relu(real_weight_deviation-self.weight_deviation) + \
                100000*tf.nn.relu(self.turn_over-self.turn_over_limit) + \
                100000*tf.nn.relu(tf.reduce_mean(self.prediction[0][0])-self.weight_max) + \
                100000*tf.nn.relu(self.fe_return-self.return_max)
                self.ex_return = tf.reduce_sum(self.prediction_final)/self.timesteps*250
                return loss
        
            output =  model()
            self.loss = cal_loss(output=output,input_samples=self.tf_train_samples,factor_BP=self.factor_BP,factor_return=self.factor_return)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

    
    # first training
    def run(self):
        
        self.session = tf.Session(graph=self.graph)
        
        with self.session as sess:
            
            training_x, weight0, factor_BP, factor_return, industry_class, index_industry, stocks = get_chunk(self.timesteps,self.num_input)
            #latest_weight = np.array([1.0/self.num_input]*self.num_input)
            latest_weight = weight0/sum(weight0)
            tf.global_variables_initializer().run()
            print('Start Training:')
            l = np.zeros(self.training_steps)
            ex_return = np.zeros(self.training_steps)
            te = np.zeros(self.training_steps)
            fe = np.zeros(self.training_steps)
            w = np.zeros([self.training_steps,self.num_input])
            turn_over = np.zeros(self.training_steps)
            id_dev = np.zeros(self.training_steps)
            for step in range(0, self.training_steps):
            # Run optimization op (backprop)
                _, l[step], ex_return[step], te[step], fe[step], w[step], turn_over[step], id_dev[step] = \
                sess.run([self.optimizer,self.loss,self.ex_return,self.te,self.fe_BP,self.prediction,\
                self.turn_over,self.real_industry_deviation], \
                                        feed_dict={self.tf_train_samples: training_x, \
                                        self.latest_weight:  latest_weight, self.factor_BP: factor_BP, self.factor_return: \
                                        factor_return, self.industry_class: industry_class, self.index_industry: index_industry, \
                                        })       
                if step % self.display_step == 0:
                    print("Step {0:4s}, Loss = {1:12.3f}, Turn_over = {2:6.2f}, TE = {3:.3f}, Industry_dev = {4:.3f}, FE = {5:.3f}".format(str(step),l[step],turn_over[step],\
                    te[step],id_dev[step],fe[step]))

            print("First Optimization Finished!")

            result = pd.DataFrame([w[-1],latest_weight,industry_class,factor_BP],index=['final_weight','initial_weight', \
            'industry','BP'],columns=stocks).T
            print(result.sort_values(by='final_weight'))

            # choose final stocks
            stocks = result.sort_values(by='final_weight',ascending=False).iloc[0:self.num_final_stocks].index
            d = dict(list(zip(result.index,list(range(0,result.shape[0])))))
            stocks_location = [d.get(s) for s in stocks]


            # new inputs
            weight0 = weight0[stocks_location]
            latest_weight = weight0/sum(weight0)
            training_x = pd.DataFrame(training_x)
            training_x = training_x.iloc[:,stocks_location]
            factor_BP = factor_BP[stocks_location]
            factor_return = factor_return[stocks_location]
            industry_class = industry_class[stocks_location]

            return(weight0,latest_weight,training_x,factor_BP,factor_return,industry_class,index_industry,stocks)
        
    # second training
    def second_run(self,weight0,latest_weight,training_x,factor_BP,factor_return,industry_class,index_industry,stocks):
        
        self.session = tf.Session(graph=self.graph)

        with self.session as sess:

            tf.global_variables_initializer().run()
            print('Start Training:')
            l = np.zeros(self.training_steps)
            ex_return = np.zeros(self.training_steps)
            te = np.zeros(self.training_steps)
            fe = np.zeros(self.training_steps)
            w = np.zeros([self.training_steps,self.num_final_stocks])
            turn_over = np.zeros(self.training_steps)
            id_dev = np.zeros(self.training_steps)
            for step in range(0, self.training_steps):
            # Run optimization op (backprop)
                _, l[step], ex_return[step], te[step], fe[step], w[step], turn_over[step], id_dev[step] = \
                sess.run([self.optimizer,self.loss,self.ex_return,self.te,self.fe_BP,self.prediction,\
                self.turn_over,self.real_industry_deviation], \
                                        feed_dict={self.tf_train_samples: training_x, \
                                        self.latest_weight:  latest_weight, self.factor_BP: factor_BP, self.factor_return: \
                                        factor_return, self.industry_class: industry_class, self.index_industry: index_industry, \
                                        })       
                if step % self.display_step == 0:
                    print("Step {0:4s}, Loss = {1:12.3f}, Turn_over = {2:6.2f}, TE = {3:.3f}, Industry_dev = {4:.3f}, FE = {5:.3f}".format(str(step),l[step],turn_over[step],\
                    te[step],id_dev[step],fe[step]))

            print("Second Optimization Finished!")

            self.fe_end = fe[-1]
            result = pd.DataFrame([w[-1],latest_weight,industry_class,factor_BP],index=['final_weight','initial_weight', \
            'industry','BP'],columns=stocks).T
            print(result)
            

if __name__ == '__main__':
    
    arg_dict = {'num_final_stocks': 100, 'te_limit': 0.05, 'industry_deviation': 0.01, 'turn_over_limit': 0.5, \
    'weight_deviation': 0.03, 'weight_max': 0.1, 'return_max': 0.5}
    net = one_sample(num_input = num_comp, timesteps = 100, learning_rate = 0.00001, \
    training_steps = 2000, display_step = 100, num_industry = len(industry_name), **arg_dict)
    net.define_graph(weight_dim=num_comp)
    weight0,latest_weight,training_x,factor_BP,factor_return,industry_class,index_industry,stocks = net.run()
    net.define_graph(weight_dim=arg_dict['num_final_stocks'])
    net.second_run(weight0,latest_weight,training_x,factor_BP,factor_return,industry_class,index_industry,stocks)