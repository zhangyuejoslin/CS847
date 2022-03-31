import scipy.io
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
import matplotlib.pyplot as plt

def predict(W, b, X):
    # logistic regression
    Z = 1 / ( 1 + np.exp( - ( X.dot(W.T ) + b ) ) )        
    Y = np.where( Z > 0.5, 1, -1 )        
    return Y


ad_data_path = "/home/joslin/algorithms/machine learning/logistic regression/alzheimers/ad_data.mat"
feature_name_path = "/home/joslin/algorithms/machine learning/logistic regression/alzheimers/feature_name.mat"

ad_data = scipy.io.loadmat(ad_data_path)
feature_name = scipy.io.loadmat(feature_name_path)

X_train, Y_train, X_test, Y_test = ad_data['X_train'], ad_data['y_train'], ad_data['X_test'], ad_data['y_test']
accuracy = []

par_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
select_num = []

# upload parameters
parameter_path = "/home/joslin/algorithms/machine learning/logistic regression/alzheimers/parameters/"
for each_file in sorted(os.listdir(parameter_path)):
    parameters = []
    with open(parameter_path+each_file) as f_p:
        parameters += [np.float(line.strip()) for line in f_p]
        w, c = np.array(parameters)[:116], np.array(parameters)[-1]

    Y_pred = predict(w, c, X_test)
    # accuracy
    correct = 0       
    count = 0    
    for count in range( np.size( Y_pred ) ) :  
        if Y_test[count] == Y_pred[count] :            
            correct = correct + 1
        count = count + 1
    select_num.append(np.count_nonzero(w))
    accuracy.append(correct/count)

# for par in par_list:
#     clf = LogisticRegression(penalty='l1', C=par, solver="liblinear", tol=1e-6)
#     clf.fit(X_train, Y_train)
#     w, c = clf.coef_, clf.intercept_
#     #tmp_acc = clf.score(X_test, Y_test)
#     Y_pred = predict(w, c, X_test)
#     # accuracy
#     correct = 0       
#     count = 0    
#     for count in range( np.size( Y_pred ) ) :  
#         if Y_test[count] == Y_pred[count] :            
#             correct = correct + 1
#         count = count + 1
#     accuracy.append(correct/count)
    #accuracy.append(tmp_acc)

#plt.plot(par_list, accuracy, 'o-', color='r', label='accuracy')
plt.plot(par_list, select_num, 'o-', color='blue', label='select_num')

plt.xlabel('Par')
plt.ylabel('select_num')
plt.legend(loc = 'best')

plt.savefig('logistic_selectnum_disease.png')
plt.show() 
print('yue')