import sys
import os                                                                                                
import random
import numpy as np                ##Calculate the inner product
import pandas as pd

#from multiprocessing import Pool
from datetime import datetime, timedelta, date
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
'''
Ceated by Pierre Lee, 2021/10/18
Python version: 3.5
library: numpy, pandas, sklearn
Global K-means algorithm
Global K-means, Fast Global K-means, Mix Global K-means
'''

##Limit the thread number in numpy calculation
#os.environ['OPENBLAS_NUM_THREADS'] = '4'
#os.environ['NUMEXPR_NUM_THREADS'] = '4'
#os.environ['MKL_NUM_THREADS'] = '4'

#Scikit-Learn perform K-means++ algorithm
def cluster_by_kmeans(data_for_cluster_series_run, n_clusters):
    #model = KMeans(n_clusters=n_clusters, random_state=0).fit(data_for_cluster_series_run)
    model = KMeans(n_clusters=n_clusters, n_init = 1).fit(data_for_cluster_series_run)
    labels = model.labels_
    cluster_center = model.cluster_centers_
    inertia = model.inertia_
    #print(f'Ave Inertia: {inertia / len(data_for_cluster_series_run)}')
    print(f'Inertia: {inertia}')
    print(f'Iteration: {model.n_iter_}')

    return labels, cluster_center, inertia

class Global_Kmeans:
    
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    ##Directly perform global kmeans followed the version in the paper
    def global_original(self, list_data):
        print('Perform global kmeans algorithm')
        n_clusters = self.n_clusters
        data_for_cluster = np.array(list_data)
        ##Search the 1st cluster center = mean(x)
        cluster_center = np.array([np.mean(data_for_cluster, axis = 0)])
        min_cluster_center = cluster_center
        min_inertia = np.sum((data_for_cluster - cluster_center[0]) ** 2)
        min_labels = np.array([])
        for i in range(2, n_clusters + 1):
            #print(f'Add {i}th center')
            ##Kmean for each data point
            for ith_center in data_for_cluster:
                temp_cluster_center = np.append(cluster_center, [ith_center], axis = 0)
                model = KMeans(n_clusters=i, init = temp_cluster_center, n_init=1, ).fit(data_for_cluster)
                if model.inertia_ < min_inertia:
                    min_inertia = model.inertia_
                    min_cluster_center = model.cluster_centers_
                    min_labels = model.labels_
            print(f'min inertia in {i}th center: {min_inertia}')
            cluster_center = min_cluster_center
            #print(cluster_center)

        print(f'min inertia:{min_inertia}')
        return min_labels, cluster_center, min_inertia

    ##Input: data points, cluster_center
    ##Output: initial_inertia, final_inertia, cluster_center
    def cal_inertia_kmeans(self, x):
        data_for_cluster = x[0]
        cluster_center = x[1]
        result_argmin = pairwise_distances_argmin_min(data_for_cluster, cluster_center)
        model = KMeans(n_clusters=len(cluster_center), init = cluster_center, n_init=1, ).fit(data_for_cluster)
        return [np.sum(result_argmin[1] ** 2), model.inertia_, model.cluster_centers_]
    
    ##Perform global kmeans with storing the ranks of inertia
    def global_store_rank(self, list_data):
        print('Perform global kmeans algorithm and store the inertia rank data')
        n_clusters = self.n_clusters
        data_for_cluster = np.array(list_data)
        
        ##Skip the duplicated data point
        list_data.sort()
        data_unique = np.array([list_data[i] for i in range(len(list_data)) if i == 0 or list_data[i] != list_data[i-1]]) 
        print(f'Number of iterated in each adding step {len(data_unique)}')
        
        ##Search the 1st cluster center = mean(x)
        cluster_center = np.array([np.mean(data_for_cluster, axis = 0)])
        #min_cluster_center = cluster_center
        #min_inertia = np.sum((data_for_cluster - cluster_center[0]) ** 2)
        #min_labels = np.array([])
        pd_rank_inertia = pd.DataFrame([])
        for i in range(2, n_clusters + 1):
            #print(f'Add {i}th center')
            ##Kmean for each data point
            inertias = [self.cal_inertia_kmeans([data_for_cluster, np.append(cluster_center, [ith_center], axis = 0)]) for ith_center in data_unique]
            
            pd_inertias = pd.DataFrame(inertias, columns= ['Inertia_i', 'Inertia_f', 'cluster_center'])
            pd_inertias['rank_f'] = pd_inertias['Inertia_f'].rank(method = 'dense')
            pd_rank_inertia[i] = pd_inertias.sort_values(by = ['Inertia_i'])['rank_f'].tolist()
            min_inertia = min(pd_inertias['Inertia_f'].tolist())
            print(f'{i}th inertia after kmeans: {min_inertia}')
            #cluster_center = pd_inertias.loc[pd_inertias['Inertia_f'] == min_inertia]['cluster_center'][0]
            i_cluster_center = pd_inertias.sort_values(by = ['Inertia_f'])['cluster_center'].keys()[0]
            cluster_center = pd_inertias['cluster_center'][i_cluster_center]
            result_argmin = pairwise_distances_argmin_min(data_for_cluster, cluster_center)
            #print(f'Inertia Check: {np.sum(result_argmin[1] ** 2)}')
            #print(cluster_center)
            #print('')
        #result_argmin = pairwise_distances_argmin_min(data_for_cluster, cluster_center)
        #print(f'Inertia after global Kmeans: {np.sum(result_argmin[1] ** 2)}')
        model = KMeans(n_clusters=n_clusters, max_iter = 100, init = cluster_center, n_init=1, ).fit(data_for_cluster)
        labels = model.labels_
        cluster_center = model.cluster_centers_
        min_inertia = model.inertia

        path_rank_inertia = 'inertia_rank.csv'
        pd_rank_inertia.to_csv(path_rank_inertia)
        print(f'Min inertia after global Kmeans: {min_inertia}')
        print(f'Write the rank of inertia to {path_rank_inertia}')
        return labels, cluster_center, min_inertia

    def global_fast(self, list_data):
        print('Perform fast global kmeans algorithm')
        n_clusters = self.n_clusters
        data_for_cluster = np.array(list_data)
        
        ##Skip the duplicated data point
        list_data.sort()
        data_unique = np.array([list_data[i] for i in range(len(list_data)) if i == 0 or list_data[i] != list_data[i-1]]) 
        
        ##Search the 1st cluster center = mean(x)
        cluster_center = np.array([np.mean(data_for_cluster, axis = 0)])
        min_cluster_center = cluster_center
        min_inertia = np.sum((data_for_cluster - cluster_center[0]) ** 2)
        labels = np.array([])
        for i in range(2, n_clusters + 1):
            #print(f'Add {i}th center')
            ##Kmean for the minimum inertia
            for ith_center in data_unique:
                temp_cluster_center = np.append(cluster_center, [ith_center], axis = 0)
                result_argmin = pairwise_distances_argmin_min(data_for_cluster, temp_cluster_center)
                ##result[0] = labels for each data points
                ##result[1] = distance to closest cluster center for each data points
                inertia = np.sum(result_argmin[1] ** 2)
                if inertia < min_inertia:
                    min_inertia = inertia
                    min_cluster_center = temp_cluster_center
                    #print(f'iterated inertia:{min_inertia}')
            
            model = KMeans(n_clusters=i, init = min_cluster_center, n_init=1, ).fit(data_for_cluster)
            labels = model.labels_
            cluster_center = model.cluster_centers_
            min_inertia = model.inertia_
            print(f'{i}th inertia after kmeans: {min_inertia}')

        print(f'Inertia after fast global kmeans:{min_inertia}')
        return labels, cluster_center, min_inertia

    ##Modify the process in each step of adding cluster center:
    ##Store the distance between data point and cluster center once before searching the minimum initial inertia
    def global_fast_v2(self, list_data):
        print('Perform fast global kmeans algorithm')
        n_clusters = self.n_clusters
        data_for_cluster = np.array(list_data)
        
        ##Skip the duplicated data point
        list_data.sort()
        data_unique = np.array([list_data[i] for i in range(len(list_data)) if i == 0 or np.all(list_data[i] == list_data[i-1]) == False]) 
        
        ##Search the 1st cluster center = mean(x)
        cluster_center = np.array([np.mean(data_for_cluster, axis = 0)])
        min_cluster_center = cluster_center
        min_inertia = np.sum((data_for_cluster - cluster_center[0]) ** 2)
        labels = np.array([])
        print(f'1st min inertia: {min_inertia}')
        for i in range(2, n_clusters + 1):
            ##Store the distance between data points and the cluster centers of 1~(M-1)
            min_dis2_centers = pairwise_distances_argmin_min(data_for_cluster, cluster_center)[1] ** 2
            ##result[0] = labels for each data points
            ##result[1] = distance to closest cluster center for each data points
            ##Kmean for the minimum inertia
            for ith_center in data_unique:
                dis2_center = np.sum((data_for_cluster - ith_center) ** 2, axis = 1)
                out = np.where(min_dis2_centers < dis2_center, min_dis2_centers, dis2_center) 
                inertia = np.sum(out)
                if inertia < min_inertia:
                    min_inertia = inertia
                    min_cluster_center = np.append(cluster_center, [ith_center], axis = 0)
            
            model = KMeans(n_clusters=i, init = min_cluster_center, n_init=1, ).fit(data_for_cluster)
            labels = model.labels_
            cluster_center = model.cluster_centers_
            min_inertia = model.inertia_
            print(f'{i}th inertia after kmeans: {min_inertia}')

        print(f'Inertia after fast global kmeans:{min_inertia}')
        return labels, cluster_center, min_inertia
    

    ##Mix the global kmeans and fast global kmeans, version 1
    ##Set the ratio of sample(top list in lower initial inertia) to perform kmeans in each adding step
    ##Ex: ratio = 0.1, n_sample = 100 there are top 10 sample with lower initial inertia to perform kmeans.
    def global_fast_mix1(self, list_data, global_ratio):
        n_clusters = self.n_clusters
        data_for_cluster = np.array(list_data)
        
        ##Skip the duplicated data point
        list_data.sort()
        data_unique = np.array([list_data[i] for i in range(len(list_data)) if i == 0 or np.all(list_data[i] == list_data[i-1]) == False]) 
        n_kmeans = int(global_ratio * len(data_unique))
        print(f'Perform mix_v1 global kmeans algorithm, number kmeans each adding step: {n_kmeans}')
        
        ##Search the 1st cluster center = mean(x)
        cluster_center = np.array([np.mean(data_for_cluster, axis = 0)])
        min_cluster_center = cluster_center
        min_inertia = np.sum((data_for_cluster - cluster_center[0]) ** 2)
        labels = np.array([])
        for i in range(2, n_clusters + 1):
            #print(f'Add {i}th center')
            ##Kmean for the minimum inertia
            inertias = [] 
            min_dis2_centers = pairwise_distances_argmin_min(data_for_cluster, cluster_center)[1] ** 2
            ##result[0] = labels for each data points
            ##result[1] = distance to closest cluster center for each data points
            ##Kmean for the minimum inertia
            for ith_center in data_unique:
                dis2_center = np.sum((data_for_cluster - ith_center) ** 2, axis = 1)
                inertia = np.sum(np.where(min_dis2_centers < dis2_center, min_dis2_centers, dis2_center)) 
                inertias.append([inertia, ith_center])
            low_inertias = sorted(inertias, key = lambda x: x[0])[0:n_kmeans]
            for j in low_inertias:
                new_center = j[1]
                model = KMeans(n_clusters=i, init = np.append(cluster_center, [new_center], axis = 0), n_init=1, ).fit(data_for_cluster)
                if model.inertia_ < min_inertia:
                    min_inertia = model.inertia_
                    min_cluster_center = model.cluster_centers_
                    labels = model.labels_
            cluster_center = min_cluster_center
            print(f'{i}th inertia after kmeans:{min_inertia}')

        print(f'Inertia after mix_v1 global kmeans:{min_inertia}')
        return labels, cluster_center, min_inertia

    ##Mix the global kmeans and fast global kmeans, version 2
    ##Set the multiple to multiply the root of n_sample and get the top list with lower inertia to perform kmeans in each adding step
    ##Ex: v_times = 2, n_sample = 100 there are top 2 * 100^(0.5) = 20 sample with lower initial inertia to perform kmeans.
    def global_fast_mix2(self, list_data, v_times):
        n_clusters = self.n_clusters
        data_for_cluster = np.array(list_data)
        
        ##Skip the duplicated data point
        list_data.sort()
        data_unique = np.array([list_data[i] for i in range(len(list_data)) if i == 0 or np.all(list_data[i] == list_data[i-1]) == False]) 
        n_kmeans = int(v_times * (len(data_unique) ** 0.5))
        print(f'Perform mix_v2 global kmeans algorithm, number kmeans each adding step: {n_kmeans}')
        
        ##Search the 1st cluster center = mean(x)
        cluster_center = np.array([np.mean(data_for_cluster, axis = 0)])
        min_cluster_center = cluster_center
        min_inertia = np.sum((data_for_cluster - cluster_center[0]) ** 2)
        labels = np.array([])
        #k_global = random.sample(range(3, n_clusters + 1), n_global_step)
        for i in range(2, n_clusters + 1):
            inertias = [] 
            min_dis2_centers = pairwise_distances_argmin_min(data_for_cluster, cluster_center)[1] ** 2
            ##result[0] = labels for each data points
            ##result[1] = distance to closest cluster center for each data points
            ##Kmean for the minimum inertia
            for ith_center in data_unique:
                dis2_center = np.sum((data_for_cluster - ith_center) ** 2, axis = 1)
                inertia = np.sum(np.where(min_dis2_centers < dis2_center, min_dis2_centers, dis2_center)) 
                inertias.append([inertia, ith_center])
            low_inertias = sorted(inertias, key = lambda x: x[0])[0:n_kmeans]
            for j in low_inertias:
                new_center = j[1]
                model = KMeans(n_clusters=i, init = np.append(cluster_center, [new_center], axis = 0), n_init=1, ).fit(data_for_cluster)
                if model.inertia_ < min_inertia:
                    min_inertia = model.inertia_
                    min_cluster_center = model.cluster_centers_
                    labels = model.labels_
            cluster_center = min_cluster_center
            print(f'{i}th inertia after kmeans:{min_inertia}')

        print(f'Inertia after mix_v2 global kmeans:{min_inertia}')
        return labels, cluster_center, min_inertia

#    
if __name__ == '__main__':
    pass
