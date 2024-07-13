import numpy as np
from sklearn import preprocessing

# Define sample data
# input_labels = ['red', 'black', 'red', 'green', 'black', 'white']
input_labels = ['red', 'green', 'orange', 'blue', 'black', 'white']
print(input_labels)


# Create and trained of label encoder object
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
print(input_labels)

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
print(input_labels)


test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\n label = ", test_labels)
print("Encoded values =", list(encoded_values))


test_labels = ['white', 'orange', 'green']
encoded_values = encoder.transform(test_labels)
print('Encoder values=', list(encoded_values))

# Decoding
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\n Encoded values =", encoded_values)
Encoded_values = [3, 0, 4, 1]
print("\n Decoded labels =", list(decoded_list))

encoded_values = [3, 2, 0, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print('\n Encoded values=', encoded_values)

# decoding a set of values
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)

print("\nDecoded labels =", list(decoded_list))

encoder_values = [3, 2, 0, 1]
decoder_lsit = encoder.inverse_transform(encoder_values)
print('\n Encoder values=', encoder_values)
print('\n Dncoder values=', list(decoder_lsit))

