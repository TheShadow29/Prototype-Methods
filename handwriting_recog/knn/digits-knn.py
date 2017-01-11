import knn
import random

def digitsData():
    ''' Read in the handwritten digits data from the file 'digits.dat', and
    return the data points and their labels as two lists. '''
    with open('../data/digits.dat') as inFile:
        lines = inFile.readlines()

    data = [line.strip().split(',') for line in lines]
    data = [([int(x) for x in point.split()], int(label)) for (point, label) in data]

    return data    
    
def column(A,j):
    return [row[j] for row in A]

def test(data,k):
    random.shuffle(data)
    pts, labels = column(data,0), column(data,1)
    training_data = pts[:800]
    training_labels = labels[:800]

    test_data = pts[800:]
    test_labels = labels[800:]

    f = knn.make_knn_classifier(k,training_data,training_labels,knn.euclidean_dist)

    correct = 0
    total = len(test_labels)

    for (point,label) in zip(test_data,test_labels):
        if (f(point) == label):
            correct += 1
    return correct/total

if __name__ == '__main__':
    data = digitsData()

    print ("k\tcorrect")
    rates=[]
    for k in range(1,50):
        print ("Started iter " + str(k))
        successRate = test(data, k)
        print (str(k) + '\t' + str(successRate))
        rates.append(successRate)
