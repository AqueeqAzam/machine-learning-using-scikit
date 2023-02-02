# Components of machine learning
# Data representation -> Tunable model + Optimization algo -> Trained model


import numpy as np

"""input_data = np.array([[2.1, 4.5, 5.6],
              [9.4, 7.6, 23.2],
              [3.2, 7.4, -45.6]])
print(input_data)"""

input_data = np.array([[2.1, 2.6, 5.2],
                       [-12.4, 4.7, -3.5],
                       [2, 0.4, 12.5]])
print(input_data)


# Technique of data processing
from sklearn import preprocessing
"""data_binarized = preprocessing.Binarizer(threshold=0.5).transform(input_data)
print("\n Binarized data:\n", data_binarized)

data_binary = preprocessing.Binarizer(threshold=5).transform(input_data)
print('\n Binarised data:\n', data_binary)"""





# Mean removal
"""print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

print('Mean')

print('Mean', input_data.mean(axis=1))
print('Standered deviation', input_data.std(axis=1))"""

# Scaling
"""data_scalar_minmax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled_minmax = data_scalar_minmax.fit_transform(input_data)
print("\n Min max scaled data: \n", data_scalar_minmax)

data_scale_minimax = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scale_minimax = data_scale_minimax.fit_transform(input_data)
print(data_scale_minimax)"""


# Normalization -> Rescale the feature in fixed range between 0 and 1.
"""data_normalized_l1 = preprocessing.normalize(input_data, norm= 'l1')
print("\n normalzed data:\n", data_normalized_l1)"""



