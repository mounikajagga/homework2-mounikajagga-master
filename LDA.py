#By Mounika#
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataread = pd.read_csv(r"SCLC_study_output_filtered_2.csv", header=0, index_col=0)
data = dataread.values
data = data.astype(float)

meanfeature = []
meanfeature.append(np.mean(data[range(20)], axis=0))
meanfeature.append(np.mean(data[range(20,40)], axis=0))


with_S = np.zeros((19, 19))

sc_classmat1 = np.zeros((19, 19))
for row in data[range(20)]:
    row, meanfeature[0] = row.reshape(19,1), meanfeature[0].reshape(19,1)
    sc_classmat1 += (row-meanfeature[0]).dot((row-meanfeature[0]).T)
    
sc_classmat2 = np.zeros((19, 19))         
for row in data[range(20,40)]:
    row, meanfeature[1] = row.reshape(19,1), meanfeature[1].reshape(19,1)
    sc_classmat2 += (row-meanfeature[1]).dot((row-meanfeature[1]).T)
    
with_S += sc_classmat1 + sc_classmat2


total_mean = np.mean(data, axis=0)
total_mean = total_mean.reshape(19,1


S_between = np.zeros((19,19))

n1 = data[range(20),:].shape[0]
mean_vector = meanfeature[0].reshape(19,1)
S_between = S_between + n1 * (mean_vector - total_mean).dot((mean_vector - total_mean).T)

n2 = data[range(20,40),:].shape[0]
mean_vec = meanfeature[1].reshape(19,1)
S_between = S_between + n2 * (mean_vector - total_mean).dot((mean_vector - total_mean).T)


eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(with_S).dot(S_between))
eig_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)


W = eig_pairs[0][1].reshape(19,1)
data_lda = data.dot(W).real

ax = plt.subplot(111)
plt.scatter(x=data_lda[range(20)], y=np.zeros(20), marker='s', color='blue')
plt.scatter(x=data_lda[range(20,40)], y=np.zeros(20), marker='^', color='red')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('My LDA function')
plt.show()

# -----------------------------------------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

y = np.concatenate((np.zeros(20), np.ones(20)))

sklearn = LDA(n_components=1)
lda_fit = sklearn.fit_transform(data, y)


ax = plt.subplot(111)
plt.scatter(x=lda_fit[range(20)], y=np.zeros(20), marker='s', color='blue')
plt.scatter(x=lda_fit[range(20,40)], y=np.zeros(20), marker='^', color='red')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('Sklearn Module')
plt.show()
