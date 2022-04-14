from matplotlib import pyplot as plt
import scipy.io
import cv2
import numpy as np
from numpy import linalg as LA
from sklearn.decomposition import PCA


### Draw Pictures
def draw_pic():
    data = [(0,0),(-1,2),(-3,6),(1,-2),(3,-6)]

    X = list(list(zip(*data))[0])
    Y = list(list(zip(*data))[1])

    plt.scatter(X,Y)
    plt.savefig('/VL/space/zhan1624/PR_homework/PCA/dot.png')
    plt.show()



def pcaSVD(X, num_components):
    # Center X and get covariance matrix C
    n, p = X.shape
    X_meaned = X - np.mean(X , axis = 0)
    X -= X.mean(axis=0)

    # SVD
    u, sigma, vt = np.linalg.svd(X, full_matrices=False)
    eigen_vectors = vt.T
    eigen_values = (sigma**2) / (n-1)

    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()

    # Return principal compnents and eigenvalues to calculate the portion of sample variance explained
    return X_reduced, eigen_values, eigen_vectors




mat = scipy.io.loadmat('/VL/space/zhan1624/PR_homework/PCA/USPS.mat')
X = mat['A'][1].reshape(16,16)
plt.imshow(X,cmap="gray")

error_record=[]
p_list = [10,50,100,200]
data = mat['A']
new_draw_fig = []
for i in p_list:
    pca = PCA(n_components=i)
    pca2_results = pca.fit_transform(data)
    pca2_proj_back=pca.inverse_transform(pca2_results)
    total_loss=LA.norm((data-pca2_proj_back),None)
    error_record.append(total_loss)
    plt.imshow(pca2_proj_back[1].reshape(16,16),cmap="gray")
    plt.show()

plt.figure(figsize=(15,15))
plt.plot(error_record, p_list)
plt.title("reconstruct error of pca")
plt.savefig('pca.png')
plt.show()
print('yue')

