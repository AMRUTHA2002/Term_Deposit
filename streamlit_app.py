import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

    # changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person has not taken a Term Deposit'
    else:
        return 'The person has taken a Term Deposit'


def display_graphs(df):
    customers_with_loan = df[df['loan'] == 'yes']
    percentage_with_insurance = (customers_with_loan['Insurance'] == 'yes').mean() * 100

    # Define a more vibrant color palette
    colors = ['#3498db', '#2ecc71']

    # Create a custom style for the plot
    plt.style.use('default')

    # Use Streamlit's plotting capabilities
    st.subheader('1. Loan and Insurance Analysis')

    # Calculate percentage within the function
    percentage_with_insurance = (customers_with_loan['Insurance'] == 'yes').mean() * 100
    
    # Display pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        [percentage_with_insurance, 100 - percentage_with_insurance],
        autopct='%1.1f%%',
        textprops=dict(color="w"),
        colors=colors,
        wedgeprops=dict(width=0.3, edgecolor='w'),
        explode=(0.1, 0),
        startangle=140
    )
    plt.setp(autotexts, size=12, weight="bold")
    ax.legend(wedges, ['With Insurance', 'Without Insurance'], loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title('Percentage of Customers with a Loan who have Insurance', fontsize=16, fontweight='bold')
    st.pyplot(fig)

    # Display the percentage
    st.write(f'Percentage of customers with a loan who have insurance: {percentage_with_insurance:.2f}%')

    # Create a Streamlit app
    st.subheader('2. Income Insights')

    # Display a histogram with Seaborn
    plt.figure(figsize=(10, 8))
    ax = sns.histplot(df[df['Annual Income'] == 0]['age'], kde=True, palette='viridis', bins=20)

    # Add count labels on top of the bars
    for rect in ax.patches:
        height = rect.get_height()
        count = int(height)
        ax.text(rect.get_x() + rect.get_width() / 2, height, count, ha='center', va='bottom', fontsize=8, color='white')

    # Customize plot aesthetics
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Data Distribution of Customers with No Annual Income', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot using Streamlit
    st.pyplot(plt)

    # Display additional information or insights
    st.write(f'Total number of customers with no annual income: {len(df[df["Annual Income"] == 0])}')



def display_distributions(data):
    customers_without_loan = data[data['loan'] == 'no']
    customers_with_loan = data[data['loan'] == 'yes']

    # Income Distribution
    st.subheader('3. Loan-less Customers Profile')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(customers_without_loan['Annual Income'], bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='No Loan')
    sns.histplot(customers_with_loan['Annual Income'], bins=30, color='green', edgecolor='black', alpha=0.7, label='With Loan')
    ax.set_title('Income Distribution')
    ax.set_xlabel('Income')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    # Balance Distribution
    st.subheader('Balance Distribution')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(customers_without_loan['balance'], bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='No Loan')
    sns.histplot(customers_with_loan['balance'], bins=30, color='green', edgecolor='black', alpha=0.7, label='With Loan')
    ax.set_title('Balance Distribution')
    ax.set_xlabel('Balance')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    # Profession Distribution
    st.subheader('Profession Distribution')
    fig, ax = plt.subplots(figsize=(10, 5))
    customers_without_loan['job'].value_counts().plot(kind='bar', color='skyblue', alpha=0.7, label='No Loan')
    customers_with_loan['job'].value_counts().plot(kind='bar', color='green', alpha=0.7, label='With Loan')
    ax.set_title('Profession Distribution')
    ax.set_xlabel('Profession')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

def display_success_percentage(data):
    successful_contact_data = data[data['Response'] == 'yes']
    success_percentage = successful_contact_data.groupby('contact').size() / len(successful_contact_data) * 100

    # Use Streamlit's plotting capabilities
    st.subheader('4. Communication Strategy Insights')

    # Display the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=success_percentage.index, y=success_percentage.values, color='skyblue')
    plt.title('Success Percentage by Contact Method')
    plt.xlabel('Contact Method')
    plt.ylabel('Success Percentage')
    plt.xticks(rotation=0)
    st.pyplot(plt)

def display_age_statistics(data):
    bins = [18, 30, 40, 50, 60, 70, 100]
    labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '70+']

    # Percentage of Home Loans by Age Group
    data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)
    percentage_home_loans = data.groupby('age_group')['housing'].apply(lambda x: (x == 'yes').mean()) * 100

    # Use Streamlit's plotting capabilities
    st.subheader('5. Age and Home Loans')

    # Display the bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=percentage_home_loans.index, y=percentage_home_loans.values, palette='viridis')
    plt.title('Percentage of Home Loans by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Percentage of Home Loans')
    plt.xticks(rotation=0)
    st.pyplot(plt)

    # Relationship between Annual Income and Age Group
    st.subheader('6.Income and Age Relationship')

    # Display the boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='age_group', y='Annual Income', data=data,color='lightgreen')
    plt.title('Relationship between Annual Income and Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Annual Income')
    st.pyplot(plt)

    # Correlation between age and annual income
    correlation = data['age'].corr(data['Annual Income'])
    st.write(f"Correlation between age and annual income: {correlation}")

def display_response_statistics(data):
    data['Response'] = data['Response'].replace({'no': 0, 'yes': 1})
    # Use Streamlit's plotting capabilities
   # Define a custom color for each bar
    colors = ['skyblue', 'coral', 'lightgreen', 'gold']  # Add more colors as needed

# Create a bar plot with custom colors
    st.subheader('Response Statistics by Education')
    plt.figure(figsize=(12, 6))
    education_response_mean = data.groupby('education')['Response'].mean().sort_values(ascending=False)
    plt.bar(education_response_mean.index, education_response_mean, color=colors)
    plt.title('Response as per Education')
    plt.ylabel('Response %')
    plt.xticks(rotation=0)
    st.pyplot(plt)

    st.subheader('Response Statistics by Marital Status')
    plt.figure(figsize=(12, 6))
    marital_response_mean = data.groupby('marital')['Response'].mean().sort_values(ascending=False)
    plt.bar(marital_response_mean.index, marital_response_mean, color=colors)
    plt.title('Response as per Marital Status')
    plt.ylabel('Response %')
    plt.xticks(rotation=0)
    st.pyplot(plt)

    st.subheader('Response Statistics by Personal Loan')
    plt.figure(figsize=(12, 6))
    loan_response_mean = data.groupby('loan')['Response'].mean().sort_values(ascending=False)
    plt.bar(loan_response_mean.index, loan_response_mean, color=colors)
    plt.title('Response as per Personal Loan')
    plt.ylabel('Response %')
    plt.xticks(rotation=0)
    st.pyplot(plt)

    st.subheader('Response Statistics by Previous Outcome')
    plt.figure(figsize=(12, 6))
    poutcome_response_mean = data.groupby('poutcome')['Response'].mean().sort_values(ascending=False)
    plt.bar(poutcome_response_mean.index, poutcome_response_mean, color=colors)
    plt.title('Response as per Previous Outcome')
    plt.ylabel('Response %')
    plt.xticks(rotation=0)
    st.pyplot(plt)

    st.subheader('Response Statistics by Job')
    plt.figure(figsize=(14, 6))
    job_response_mean = data.groupby('job')['Response'].mean().sort_values(ascending=False)
    plt.bar(job_response_mean.index, job_response_mean, color=colors)
    plt.title('Response as per Job')
    plt.ylabel('Response %')
    plt.xticks(rotation=0)
    st.pyplot(plt)


def main():
    # giving a title
    st.title('Term Deposit Prediction')

    # Add a sidebar for navigation
    page = st.sidebar.radio("Select Page", ["Prediction", "Graphs","Analysis"])

    if page == "Prediction":
        # getting the input data from the user
        Job = st.selectbox('Job', list(job_mapping.keys()))
        Marital = st.selectbox('Marital Status', list(marital_mapping.keys()))
        Education = st.selectbox('Highest Education level', list(education_mapping.keys()))
        Poutcome = st.selectbox('Outcome of the previous campaign', list(poutcome_mapping.keys()))
        Age = st.text_input('Age')
        Annual_Income = st.text_input('Annual Income')
        Balance = st.text_input('Balance')
        Duration = st.text_input('Duration of call')
        Campaign = st.text_input('Number of contacts performed during this campaign')
        Last = st.text_input('Number of days that passed after the client was contacted')
        Count_Txn = st.text_input('Number of Transactions')

        # code for Prediction
        diagnosis = ''

        # creating a button for Prediction
        if st.button('Term Deposit'):
            diagnosis = Term_Deposit([Job, Marital, Education, Poutcome, Age, Annual_Income, Balance, Duration, Campaign,
                                      Last, Count_Txn])

        st.success(diagnosis)
    elif page == "Graphs":
        file_path = 'cleaned_data.xlsx'  # Update with the actual file path
        df = pd.read_excel(file_path)
        display_graphs(df)
        display_distributions(df)
        display_success_percentage(df)
        display_age_statistics(df)  # Pass your DataFrame to the function

    elif page == "Analysis":
        file_path = 'cleaned_data.xlsx'  # Update with the actual file path
        df = pd.read_excel(file_path)
        display_response_statistics(df)



if __name__ == '__main__':
    main()
