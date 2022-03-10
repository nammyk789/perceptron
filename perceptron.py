import numpy as np
from scipy.io import loadmat

class Perceptron:

    def __init__(self, training_data, training_labels, testing_data, testing_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels


    def get_scores(self):
        class_1 = []
        class_6 = []
        for idx, data_point in enumerate(self.testing_data):
            data_point = np.append(data_point, 1)
            if self.__accurate_classification(data_point, self.testing_labels[idx]):
                score = np.dot(self.weight, data_point)
                if self.y_i(self.testing_labels[idx]) == 1:
                    class_1.append((idx, score))
                else:
                    class_6.append((idx, score))
        return class_1, class_6


    def test(self):
        """
        returns what percentage of the testing data was accurately classified
        by the model
        """
        accuracy = 0
        for idx, data_point in enumerate(self.testing_data):
            data_point = np.append(data_point, 1)
            if self.__accurate_classification(data_point, self.testing_labels[idx]):
                    accuracy += 1
        return accuracy/len(self.testing_data)


    def train(self, iterations):
        self.weight = [0] * (len(self.training_data[0]) + 1)  # add one to get activation parameter
        self.weight = np.array(self.weight)
        for i in range(iterations):
            self.weight = self.update_weight(self.weight, self.training_data[i], \
                                            self.training_labels[i])
        return self.weight


    def update_weight(self, current_weight, next_data_point, next_data_label):
        """
        run one iteration of the perceptron to update the weight        
        """
        # for activation parameter
        next_data_point = np.append(next_data_point, 1)
        if self.y_i(next_data_label) ==  self.y_hat_i(current_weight, next_data_point):
            return current_weight
        else:
            return current_weight + self.y_i(next_data_label) * next_data_point
    
    def y_hat_i(self, current_weight, next_data_point):
        if np.dot(current_weight, next_data_point) > 0:
            return 1
        else:
            return -1


    def y_i(self, data_label):
        """
        convert data label into binary 1 or -1
        """
        if data_label == [1]:
            return 1
        elif data_label == [6]:
           return -1

    
    def __accurate_classification(self, data_point, label):
        if self.y_i(label) == self.y_hat_i(self.weight, data_point):
            return True
        return False



if __name__ == '__main__':
    #Loading the data
    M = loadmat('MNIST_digit_data.mat')
    images_train,images_test,labels_train,labels_test= M['images_train'],M['images_test'],M['labels_train'],M['labels_test']
    #Filtering data-- keeping only labels 
    labels_train_indices = []
    labels_test_indices = []
    for idx, label in enumerate(labels_train):
        if label in [[1], [6]]:
            labels_train_indices.append(idx)
    for idx, label in enumerate(labels_test):
        if label in [[1], [6]]:
            labels_test_indices.append(idx)
    labels_train_indices = np.array(labels_train_indices)
    labels_test_indices = np.array(labels_test_indices)
    images_train = images_train[labels_train_indices]
    labels_train = labels_train[labels_train_indices]
    images_test = images_test[labels_test_indices]
    labels_test = labels_test[labels_test_indices]
    #randomly permute data points
    inds = np.random.permutation(images_train.shape[0])
    images_train = images_train[inds]
    labels_train = labels_train[inds]


    inds = np.random.permutation(images_test.shape[0])
    images_test = images_test[inds]
    labels_test = labels_test[inds]

    perceptron = Perceptron(images_train, labels_train, images_test, labels_test)
    weight = perceptron.train(5)