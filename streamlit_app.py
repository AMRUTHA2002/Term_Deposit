import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a function for Prediction

def Term_Deposit(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person has not taken a Term Deposit'
    else:
        return 'The person has taken a Term Deposit'
  
    
  
def main():
    
    
    # giving a title
    st.title('Term Deposit Prediction')
    
    
    # getting the input data from the user
    
    
    Marital = st.text_input('Marital Status')
    Education = st.text_input('Education')
    Last = st.text_input('Last')
    Campaign = st.text_input('Campaign')
    Poutcome = st.text_input('Poutcome')
    Job = st.text_input('Job')
    Count = st.text_input('Count')
    Annual_Income = st.text_input('Annual Income')
    Balance = st.text_input('Balance')
    Age = st.text_input('Age')
    Duration = st.text_input('Duration')

    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Term Deposit'):
        diagnosis = Term_Deposit([Marital,Education,Last,Campaign,Poutcome,Job,Count,Annual_Income,Balance,Age,Duration])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
  
    
  
