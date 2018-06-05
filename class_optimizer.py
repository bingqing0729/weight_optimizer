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

# citic_1
pkl_file = open('citic_1.pkl','rb')
industry_class, industry = pickle.load(pkl_file)

# import BP value
pkl_file = open('BP.pkl','rb')
factor_mom = pickle.load(pkl_file)
# some data cleaning
factor = np.array(factor_mom.fillna(0))
factor[factor<-10]=0
factor = sklearn.preprocessing.scale(factor,axis=1)
factor = pd.DataFrame(factor,index=factor_mom.index,columns=factor_mom.columns)

# get data samples (when n >1, can get multiple samples)
def get_chunk(timesteps,num_input,n=1,factor=factor):
    x = np.zeros([n,timesteps,num_input])
    i = 0
    while i < n:
        # pick a day
        samples = np.random.randint(0,len(list(excess.index[0:-timesteps])),1)
        # current time point of index
        current_comp_index = len(weight0.index[weight0.index<excess.index[samples+timesteps][0]])-1
        # components and weights
        comp = weight0.iloc[current_comp_index,:][weight0.iloc[current_comp_index,:]>0]
        # industry distribution
        current_industry = industry.iloc[current_comp_index,:]
        index_industry = [0]*len(industry_class)
        for j in range(0,len(industry_class)):
            current_j_industry = current_industry.loc[current_industry==j].index
            index_industry[j] = sum(comp[current_j_industry].dropna())
        index_industry = np.array(index_industry)/sum(index_industry)
        # pick num_input stocks(loc)
        stocks = random.sample(range(0,500),num_input)
        # stocks and weights
        stocks = comp.index[stocks]
        weights = np.array(list(comp[stocks]))/100.0
        # history excess return
        x[i] = excess.loc[excess.index[samples][0]:excess.index[samples+timesteps-1][0],stocks]
        # history factor
        factor_h = factor.loc[excess.index[samples+timesteps-1][0],stocks]


        if sum(sum(np.isnan(x[i])))>0:
            i = i-1
        i = i+1

    return(x,weights,np.array(factor_h),np.array(current_industry[stocks]).astype('int32'),index_industry)

#
class one_sample:

    def __init__(self, num_input, from_index, timesteps, learning_rate, \
    training_steps,display_step, te_limit, industry_deviation, \
    turn_over_limit, weight_deviation, num_industry):
        
        self.num_input = num_input
        self.from_index = from_index
        self.te_limit = te_limit
        self.industry_deviation = industry_deviation
        self.turn_over_limit = turn_over_limit
        self.weight_deviation = weight_deviation
        self.num_industry = num_industry
        self.timesteps = timesteps
        self.display_step = display_step
        
        # Hyper parameters
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        
        # Graph related
        self.graph = tf.Graph()
        self.tf_train_samples = None
        self.weight0 = None
        self.latest_weight = None
        self.factor = None
        self.index_industry = None
        self.industry_class = None


    def define_graph(self):
        
        with self.graph.as_default():
            self.tf_train_samples = tf.placeholder("float", [None, self.timesteps, self.num_input])
            self.weight0 = tf.placeholder("float",self.num_input)
            self.latest_weight = tf.placeholder("float",self.num_input)
            self.factor = tf.placeholder('float',self.num_input)
            self.index_industry = tf.placeholder('float',self.num_industry)
            self.industry_class = tf.placeholder('int32',self.num_input)



            weights = tf.Variable(tf.random_normal([1,self.num_input],0,0.01))
            

            def model(reuse=None):

                return weights + tf.expand_dims(self.weight0,0) 

            # calculate loss
            def cal_loss(output,input_samples,factor):
                self.prediction = tf.abs(output)
                # the output stock weights after normalization
                self.prediction = tf.expand_dims(tf.divide(self.prediction,tf.expand_dims(\
                tf.reduce_sum(self.prediction,1),1)),1)
                self.prediction_final = tf.reduce_sum(tf.multiply(input_samples,self.prediction),2)
                factor = tf.expand_dims(factor,0)
                self.fe = tf.reduce_sum(tf.reduce_sum(tf.multiply(\
                tf.expand_dims(factor,0),self.prediction),2),1)
                # tracking error
                self.te = tf.sqrt(tf.nn.l2_loss(self.prediction_final)*2/self.timesteps*250)
                # industry proportion
                one_hot = tf.one_hot(self.industry_class,self.num_industry,1,0)
                self.real_industry_deviation = tf.reduce_mean(tf.abs(tf.subtract(tf.matmul(\
                self.prediction[0],tf.cast(one_hot,tf.float32)),self.index_industry)))
                real_weight_deviation = tf.reduce_mean(tf.abs(tf.subtract(self.prediction[0][0],self.weight0))) \
                if self.from_index == True else 0
                self.turn_over = tf.reduce_mean(tf.divide(tf.abs(tf.subtract(self.prediction[0][0],\
                self.latest_weight)),self.latest_weight))
                loss = -self.fe + 100000*tf.nn.relu(self.te-self.te_limit) + \
                100000*tf.nn.relu(self.real_industry_deviation-self.industry_deviation) + \
                100000*tf.nn.relu(real_weight_deviation-self.weight_deviation) + \
                100000*tf.nn.relu(self.turn_over-self.turn_over_limit) + \
                100000*tf.nn.relu(tf.reduce_mean(self.prediction[0][0])-0.1)
                self.ex_return = tf.reduce_sum(self.prediction_final)/self.timesteps*250
                return loss
        
            output =  model(self.tf_train_samples)
            self.loss = cal_loss(output=output,input_samples=self.tf_train_samples,factor=self.factor)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            

    def run(self):
        
        self.session = tf.Session(graph=self.graph)
        
        with self.session as sess:
            
            # training
            training_x, weight0, factor_h, industry_class, index_industry = get_chunk(self.timesteps,self.num_input)
            latest_weight = np.array([1.0/self.num_input]*self.num_input)
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
                sess.run([self.optimizer,self.loss,self.ex_return,self.te,self.fe,self.prediction,\
                self.turn_over,self.real_industry_deviation], \
                                        feed_dict={self.tf_train_samples: training_x, self.weight0: weight0, \
                                        self.latest_weight:  latest_weight, self.factor: factor_h,\
                                        self.industry_class: industry_class, self.index_industry: index_industry})       
                if step % self.display_step == 0:
                    print("Step {0:4s}, Loss = {1:12.3f}, Turn_over = {2:6.2f}, TE = {3:.3f}, Industry_dev = {4:.3f}, FE = {5:.3f}".format(str(step),l[step],turn_over[step],\
                    te[step],id_dev[step],fe[step]))

            print("Optimization Finished!")

            self.fe_end = fe[-1]
            print(w[-1])
            
            

if __name__ == '__main__':
    num_exp = 1
    net = one_sample(num_input = 200, from_index=True, timesteps = 100, learning_rate = 0.001, \
    training_steps = 8000, display_step = 200, te_limit = 0.05, industry_deviation = 0.01, \
    turn_over_limit = 0.1, weight_deviation = 0.03, num_industry = len(industry_class))
    net.define_graph()
    fe_end = 0
    for i in range(0,num_exp):
        print(i)
        net.run()
        fe_end = fe_end + net.fe_end
    
    fe_end = fe_end/num_exp
    print(fe_end)
