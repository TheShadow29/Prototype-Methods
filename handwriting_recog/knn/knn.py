import random
import math
import heapq
import matplotlib.pyplot as plt

def gaussian_cluster(center,std_dev,count=50):
    return [(random.gauss(center[0],std_dev),
             random.gauss(center[1],std_dev)) for _ in range(count)]

def make_dummy_data(count=50):
    return gaussian_cluster((-4,0),1,count) + gaussian_cluster((4,0),1,count)

def euclidean_dist(x,y):
    return math.sqrt(sum([(a-b)**2 for (a,b) in zip(x,y)]))

def make_knn_classifier(k,data,labels,distance):
    def classifier(x):
        cluster_points = heapq.nsmallest(k, enumerate(data),
                                         key=lambda y : distance(x,y[1]))
        cluster_labels = [labels[i] for (i,pt) in cluster_points]
        return max(set(cluster_labels),key=cluster_labels.count)

    return classifier

# if __name__ == '__main__':
#     training_points = make_dummy_data()
#     training_labels = [1]*50 + [2] *50
#     f = make_knn_classifier(8,training_points,training_labels,euclidean_dist)
#     # print training_points
#     p = (3,0)
#     print (f((3,0)))
#     # plt.plot
#     x_d = [point[0] for point in training_points]
#     x_d.append(p[0])
#     y_d = [point[1] for point in training_points]
#     y_d.append(p[1])
#     color_d = ['r' if training_labels[i] == 1 else 'b' for i in range(len(training_labels))]
#     color_d.append('g')
#     plt.scatter(x_d,y_d,c=color_d)
#     plt.show()

