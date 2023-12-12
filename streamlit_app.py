import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('trained_models.sav', 'rb'))


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
    
    
    Job = st.text_input('Job')
    Marital = st.text_input('Marital Status')
    Education= st.text_input('Education')
    Poutcome = st.text_input('Poutcome')
    Age = st.text_input('Age')
    Annual_Income = st.text_input('Annual Income')
    Balance = st.text_input('Balance')
    Duration = st.text_input('Duration')
    Campaign = st.text_input('Campaign')
    Last = st.text_input('Last')
    Count_Txn = st.text_input('NUmber of Transactions')

    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Term Deposit'):
        diagnosis = Term_Deposit([Job,Marital,Education,Poutcome,Age,Annual_Income,Balance,Duration,Campaign,Last,Count_Txn])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
  
    
  
