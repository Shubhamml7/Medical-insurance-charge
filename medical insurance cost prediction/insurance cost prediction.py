# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:42:49 2025

@author: ShivamShubham
"""

import numpy as np
import pickle
loaded_model=pickle.load(open("C:/Users/ShivamShubham/Desktop/medical insurance cost prediction/insurance_model.sav","rb"))
input_data=(46,1,33.44,1,1,0)

#changing tuple to numpy array
input_data_as_numpy_array=np.asarray(input_data)
#Reshape the array
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
# find the prediction 
prediction=loaded_model.predict(input_data_reshape)
print(prediction)
print('The insurance cost is USD : ',prediction[0] )
