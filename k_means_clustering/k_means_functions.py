import random
from base64 import b64decode
from json import loads
import numpy as np
import matplotlib.pyplot as plt

def parse(x):
    """
    to parse the digits file into tuples of 
    (labelled digit, numpy array of vector representation of digit)
    """
    #We are getting one line of the digits data
    #first load
    digit = loads(x)
    #now decode using b64
    array = np.fromstring(b64decode(digit["data"]),dtype=np.ubyte)
    #redefine it to be float 64 bits
    array = array.astype(np.float64)
    #return a tuple
    #[0]: Actual digit
    #[1]: The Matrix
    return (digit["label"], array)

def display_digit(digit, labeled = True, title=""):
    """
    graphical display of digit 784 x 1 vector
    """
    if labeled:
        digit = digit[1]
    img = digit
    plt.figure()
    fig = plt.imshow(img.reshape(28,28))
    fig.set_cmap('gray_r')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    if title != "":
        plt.title("Inferred label: " + str(title))
    
def init_centroids(labelled_data, k):
    """
    randomly pick k centers from the data
    """
    return map(lambda x: x[1], random.sample(labelled_data,k))

def sum_cluster(labelled_cluster):
    sum_ = labelled_cluster[0][1].copy()
    for (label, data) in labelled_cluster[1:]:
        sum_ += data
    return sum_

def mean_cluster(labelled_cluster):
    sum_of_pts = sum_cluster(labelled_cluster)
    total_pts = len(labelled_cluster)
    mean_cluster = sum_of_pts * 1.0/total_pts
    return mean_cluster

def form_clusters(labelled_data, unlabelled_centroids):
    """
    given some data and centroids for the data, allocate each datapoint
    to its closest centroid. This forms clusters.
    """
    centroid_indices = range(len(unlabelled_centroids))
    clusters = {c : [] for c in centroid_indices}

    for (label, Xi) in labelled_data:
        smallest_dist = float("inf")
        for cj_index in centroid_indices:
            cj = unlabelled_centroids[cj_index]

            dist = np.linalg.norm(cj - Xi)
            if dist < smallest_dist:
                closest_ind = cj_index
                smallest_dist = dist
        clusters[closest_ind].append((label,Xi))
    return clusters.values()

def move_centroids(labelled_clusters):
    """
    returns a list of centroids corresponding to the clusters.
    """
    new_centroids = []
    for cluster in labelled_clusters:
        new_centroids.append(mean_cluster(cluster))
    return new_centroids

def repeat_until_convergence(labelled_data, labelled_clusters,unlabelled_centroids):
    """
    form clusters around centroids, then keep moving the centroids
    until the moves are no longer significant, i.e. we've found
    the best-fitting centroids for the data.
    """
    prev_max_diff = 0
    while True:
        unlabelled_old_centroids = unlabelled_centroids
        unlabelled_centroids = move_centroids(labelled_clusters)
        labelled_clusters = form_clusters(labelled_data, unlabelled_centroids)
        differences = map(lambda a,b : np.linalg.norm(a-b), unlabelled_centroids, unlabelled_old_centroids)
        max_diff = max(differences)
        difference_change = abs((max_diff-prev_max_diff)/np.mean([prev_max_diff,max_diff])) * 100
        prev_max_diff = max_diff

        if np.isnan(difference_change):
            break

    return labelled_clusters, unlabelled_centroids

def cluster(labelled_data,k):
    """
    runs k-means clustering on the data. It is assumed that the data is labelled.
    """
    centroids = init_centroids(labelled_data,k)
    clusters = form_clusters(labelled_data,centroids)
    final_clusters, final_centroids = repeat_until_convergence(labelled_data, clusters, centroids)

    return final_clusters, final_centroids

def assign_labels_to_clusters(clusters, centroids):
    """
    Assigns a digit label to each cluster.
    Cluster is a list of clusters containing labelled datapoints.
    NOTE: this function depends on clusters and centroids being in the same order.
    """
    labelled_centroids = []
    for i in range(len(clusters)):
        labels = map(lambda x : x[0], clusters[i])
        most_common = max(set(labels),key=labels.count)
        centroid = (most_common,clusters[i])
        labelled_centroids.append(centroid)
    # print("labelled Centroids[0] ")
    # print(labelled_centroids[0])
    return labelled_centroids

def classify_digits(digit, labelled_centroids):
    """
    given an unlabelled digit represented by a vector and a list of
    labelled centroids [(label,vector)], determine the closest centroid
    and thus classify the digit.
    """
    min_dist = float("inf")
    for(label,centroid) in labelled_centroids:
        # print("Digit Line 140 ")
        # print(digit)
        # print("Centroid Line 142 ")
        # print(centroid)
        dist = np.linalg.norm(digit - centroid)
        if (dist < min_dist):
            min_dist = dist
            closest_centroid_label = centroid
    return closest_centroid_label

def get_error_rate(digits, labelled_centroids):
    """
    classifies a list of labelled digits. returns the error rate.
    """
    classified_incorrect = 0
    for(label, digit) in digits:
        #print (digit)
        classified_label = classify_digits(digit, labelled_centroids)
        if classified_label != label:
            classified_incorrect += 1
    error_rate = classified_incorrect/float(len(digits))

    return error_rate



    
if __name__=='__main__':
    with open("digits.base64.json","r") as f:
        digits = list(map(parse, f.readlines()))
    ratio_train = 0.25
    num_validate = int(len(digits) * ratio_train)
    train_set = digits[num_validate:]
    test_set = digits[:num_validate]

    error_rates = {x : None for x in list(range(5,25)) + [100]}
    for k in range(5,25):
        trained_clusters, trained_centroids = cluster(train_set,k)
        labelled_centroids = assign_labels_to_clusters(trained_clusters,trained_centroids)
        error_rate = get_error_rate(test_set,labelled_centroids)
        error_rates[k] = error_rate

    # Show the error rates
    x_axis = sorted(error_rates.keys())
    y_axis = [error_rates[key] for key in x_axis]
    plt.figure()
    plt.title("Error Rate by Number of Clusters")
    plt.scatter(x_axis, y_axis)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Error Rate")
    plt.show()
        
