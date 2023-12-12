import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users/amrut/OneDrive/Desktop/NeoStats/trained_model.sav', 'rb'))

input_data = (1,1,184,5,3,5,381,443512,5715,72,1030)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person has not taken a Term Deposit')
else:
  print('The person has taken a Term Deposit')