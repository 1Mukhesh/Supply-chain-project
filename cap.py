import pandas as pd
import streamlit as st
import pickle
from prophet import Prophet

# Load the model
pickle_in = open('Model (1).pkl', 'rb')
classifier = pickle.load(pickle_in)

# Define the welcome message function
def welcome():
    return 'Welcome all'

# Define the prediction function
def prediction(Date):
    # Convert the input date into the format expected by Prophet
    df = pd.DataFrame({'ds': [pd.to_datetime(Date)]})
    
    try:
        # Make the prediction
        prediction = classifier.predict(df)
        return prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Define the main function for the Streamlit app
def main():
    # Set the title of the web app
    st.title("Supply Chain Forecast")

    # Set the front end elements of the web page
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Supply Chain Forecast Prophet</h1>
    </div>
    """
    
    # Display the front end aspects
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Create a text input for the date
    Date = st.text_input("Date", "Type Here")
    
    # Initialize the result variable
    result = ""
    
    # When the 'Predict' button is clicked, call the prediction function
    if st.button("Predict"):
        result = prediction(Date)
        if result is not None:
            st.success('The forecast is:')
            st.write(result)
        else:
            st.error("Please check the date format and try again.")

# Run the main function
if __name__=='__main__':
    main()
