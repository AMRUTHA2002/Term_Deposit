import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# loading the saved model
loaded_model = pickle.load(open('trained_models.sav', 'rb'))

# Mapping dictionaries
poutcome_mapping = {'failure': 0, 'other': 1, 'pending': 2, 'success': 3, 'unknown': 4}
job_mapping = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4,
               'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9,
               'unemployed': 10, 'unknown': 11}
education_mapping = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}
marital_mapping = {'divorced': 0, 'married': 1, 'single': 2}

# creating a function for Prediction
def Term_Deposit(input_data):
    # Convert categorical inputs to numerical using mapping
    input_data[3] = poutcome_mapping.get(input_data[3], 4)
    input_data[0] = job_mapping.get(input_data[0], 11)
    input_data[2] = education_mapping.get(input_data[2], 3)
    input_data[1] = marital_mapping.get(input_data[1], 2)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person has not taken a Term Deposit'
    else:
        return 'The person has taken a Term Deposit'


def display_graphs():
    # Example graph
    plt.figure(figsize=(8, 6))
    sns.histplot(np.random.randn(1000), kde=True)
    st.pyplot()

    # Add more graphs as needed


def main():
    # giving a title
    st.title('Term Deposit Prediction')

    # Add a sidebar for navigation
    page = st.sidebar.radio("Select Page", ["Prediction", "Graphs"])

    if page == "Prediction":
        # getting the input data from the user
        Job = st.selectbox('Job', list(job_mapping.keys()))
        Marital = st.selectbox('Marital Status', list(marital_mapping.keys()))
        Education = st.selectbox('Education', list(education_mapping.keys()))
        Poutcome = st.selectbox('Poutcome', list(poutcome_mapping.keys()))
        Age = st.text_input('Age')
        Annual_Income = st.text_input('Annual Income')
        Balance = st.text_input('Balance')
        Duration = st.text_input('Duration')
        Campaign = st.text_input('Campaign')
        Last = st.text_input('Last')
        Count_Txn = st.text_input('Number of Transactions')

        # code for Prediction
        diagnosis = ''

        # creating a button for Prediction
        if st.button('Term Deposit'):
            diagnosis = Term_Deposit([Job, Marital, Education, Poutcome, Age, Annual_Income, Balance, Duration, Campaign,
                                      Last, Count_Txn])

        st.success(diagnosis)
    elif page == "Graphs":
        display_graphs()


if __name__ == '__main__':
    main()
