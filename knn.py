import numpy as np
import pandas as pd

class KNN:
    def __init__(self,k=3,metric=None):
        self.k = k
        self.metric = metric  # 1->euclidean 2->manhattan 3-> cosine

    def fit(self,x,y):
        self.x_train = x
        self.y_train = y

    def getDist(self,x_test):
        if self.metric==1:
            dist = self.calculate_euclidean(x_test)
        elif self.metric==2:
            dist = self.calculate_manhattan(x_test)
        elif self.metric==3:
            dist = self.calculate_cosine(x_test)
        return dist

    def getK(self,dist):
        k_nearest_indices = np.argsort(dist, axis=1)[:, :self.k]
        k_nearest_labels = np.array(k_nearest_indices.shape,dtype=int)
        k_nearest_labels = np.array([self.y_train.iloc[i] for i in k_nearest_indices])
        most_freq_labels = np.array([np.argmax(np.bincount(k_nearest_indices[i])) for i in range(k_nearest_labels.shape[0])])
        return most_freq_labels
    
    def HyperparameterTuning(self,x_test,y_test):
        k_metric = []
        accuracies = []
        for metric in range(1,4):
            self.metric = metric
            dist = self.getDist(x_test)
            for k in range(51,100,2): # so that k is odd
                self.k = k
                predicted_labels = self.getK(dist)
                # print(predicted_labels.shape)
                # calculating accuracy
                n = y_test.shape[0]
                correctly_predicted = 0
                predicted_genre = self.y_train.iloc[predicted_labels]
                # print(predicted_genre.shape)
                # print(predicted_genre)
                for i in range(n):
                    correctly_predicted += (predicted_genre.iloc[i]==y_test.iloc[i])
                accuracy = correctly_predicted/n*100
                k_metric.append((k,metric))
                accuracies.append(accuracy)
        accuracy_indices = np.argsort(np.array(accuracies))
        descending_accuracy_indices = accuracy_indices[::-1]
        best_metrics = {k_metric[i]:accuracies[i] for i in descending_accuracy_indices}
        return best_metrics

    def euclidean_dist(self,x_train_batch,x_test):
        x_test2 = np.sum(np.array(x_test)**2,axis=1)
        x_train2 = np.sum((x_train_batch.values)**2,axis=1)
        dists = np.sqrt(x_train2[np.newaxis,:] + x_test2[:,np.newaxis] - 2 * np.dot(x_test, x_train_batch.T)) 
        #chatgpt
        # prompt used - what if i wanted to calculate the euclidean distance of all the points in the test set 
        # from all the points in the train test for implementing knn without using for can i do that?
        # print(dists.shape)
        return dists

    def manhattan_dist(self,x_train_batch,x_test):
        dists = np.abs(x_test.values[:, np.newaxis, :] - x_train_batch.values[np.newaxis, :, :]).sum(axis=2)
        return dists

    def cosine_dist(self,x_train_batch,x_test):
        mag_train = np.sqrt(np.sum((x_train_batch.values)**2,axis=1))
        mag_test = np.sqrt(np.sum((x_test.values)**2,axis=1))
        dists = 1-np.dot(x_test/mag_test[:,np.newaxis],(x_train_batch.T)/mag_train)
        return dists
    

    #computing distances with 3 different distance metrics

    def calculate_euclidean(self,x_test):
        break_point = 1500
        i=0
        euclidean_distances = np.empty((0,0))
        while i<x_test.shape[0]:
            end = min(i+break_point,x_test.shape[0])
            temp = self.euclidean_dist(self.x_train[i:end],x_test)
            i += break_point
            if euclidean_distances.shape == (0,0):
                euclidean_distances = temp
            else:
                euclidean_distances = np.concatenate((euclidean_distances,temp),axis=1)
        return euclidean_distances

    def calculate_manhattan(self,x_test):
        break_point = 1500
        i=0
        manhattan_distances = np.empty((0,0))
        while i<x_test.shape[0]:
            end = min(i+break_point,x_test.shape[0])
            temp = self.manhattan_dist(self.x_train[i:end],x_test)
            i += break_point
            if manhattan_distances.shape == (0,0):
                manhattan_distances = temp
            else:
                manhattan_distances = np.concatenate((manhattan_distances,temp),axis=1)
        return manhattan_distances

    def calculate_cosine(self,x_test):
        break_point = 1500
        i=0
        cosine_distances = np.empty((0,0))
        while i<x_test.shape[0]:
            end = min(i+break_point,x_test.shape[0])
            temp = self.cosine_dist(self.x_train[i:end],x_test)
            i += break_point
            if cosine_distances.shape == (0,0):
                cosine_distances = temp
            else:
                cosine_distances = np.concatenate((cosine_distances,temp),axis=1)
        return cosine_distances

    def predict(self,x_test):
        if self.metric==1:
            dist = self.calculate_euclidean(x_test)
        elif self.metric==2:
            dist = self.calculate_manhattan(x_test)
        elif self.metric==3:
            dist = self.calculate_cosine(x_test)

        k_nearest_indices = np.argsort(dist, axis=1)[:, :self.k]
        k_nearest_labels = np.array(k_nearest_indices.shape,dtype=int)
        k_nearest_labels = np.array([self.y_train.iloc[i] for i in k_nearest_indices])
        most_freq_labels = np.array([np.argmax(np.bincount(k_nearest_indices[i])) for i in range(k_nearest_labels.shape[0])])
        return most_freq_labels