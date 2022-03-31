from cgi import test
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, data, label, epsilon, maxiters):
        self.data = data
        self.label = label
        self.learning_rate = epsilon
        self.maxiters = maxiters
        
    
    def fit(self):
        self.m, self.n = self.data.shape
        self.W = np.zeros(self.n)
        self.b = 0
        for i in range(self.maxiters):
            self.update_weights()
        return self
    
    def update_weights(self):
        y_hat = 1 / ( 1 + np.exp( - ( self.data.dot( self.W ) + self.b ) ) )
          
        # calculate gradients
        # np.dot: product and sum
        dw = (1/self.m)*np.dot(self.data.T, (y_hat - self.label.squeeze(-1)))
        db = (1/self.m)*np.sum((y_hat - self.label.squeeze(-1))) 
          
        # update weights    
        self.W = self.W - self.learning_rate * dw    
        self.b = self.b - self.learning_rate * db
          
        return self

    def predict(self, X):
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
        Y = np.where( Z > 0.5, 1, 0 )        
        return Y

    


if __name__ == '__main__':
    data_path = "/home/joslin/algorithms/machine learning/logistic regression/spam_email/data.txt"
    label_path = "/home/joslin/algorithms/machine learning/logistic regression/spam_email/labels.txt"
    data, label = [], []
    # upload data
    with open(data_path) as f_d, open(label_path) as f_l:
        data += [np.fromstring(line, dtype=float, sep=' ') for line in f_d]
        label += [np.fromstring(line, dtype=float, sep=' ') for line in f_l]
    # transfer data to array
    data_array = np.array(data)
    label_array = np.array(label)

    train_vars = [200,500,800,1000,1500,2000]

    accuracy = []

    for each_train in train_vars:
        # data preparation
        train_data = data_array[0: each_train]
        train_label = label_array[0: each_train]
        test_data = data_array[2000:]
        test_label = label_array[2000:]
        # training
        model = LogisticRegression(train_data, train_label, 1e-1, 10000)
        model.fit()
        # prediction
        Y_pred = model.predict(test_data)    
        # accuracy 
        correct = 0       
        count = 0    
        for count in range( np.size( Y_pred ) ) :  
            if test_label[count] == Y_pred[count] :            
                correct = correct + 1
            count = count + 1
        accuracy.append(correct/count)
    
    plt.plot(train_vars, accuracy, 'o-', color='r', label='accuracy')

    plt.xlabel('Train_num')
    plt.ylabel('Accuracy')
    plt.legend(loc = 'best')

    plt.savefig('logistic_accuracy1.png')
    plt.show()   
    




    
   
    