import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

KNN = joblib.load('KNN.joblib')
DT = joblib.load('DT.joblib')
Logistic = joblib.load('LogisticRegression.joblib')
RF = joblib.load('RF.joblib')
CNNLSTM = load_model('CloudBurstPredictorversion4.h5')

# Page configurations
PAGE_CONFIG = {
    "Home": "Cloudburst Chronicles: Unveiling the Power of Nature",
    "Team": "Cloud Phishers",
    "Predictions": "Get Predictions",
    
}

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", list(PAGE_CONFIG.keys()))

    st.title(PAGE_CONFIG[page])

    if page == "Home":
        #st.write("Welcome to the Home Page!")
        #st.title("Cloud Phishers")
        #st.subtitle("**Get ready to Explore the Fury and Fascination of Nature's Downpour Deluge**")

        st.header("**Get ready to Explore the Fury and Fascination of Nature's Downpour Deluge**")

        st.write("Cloudbursts are extreme weather events characterized by intense rainfall over a short period, often leading to flash floods and landslides.")

        st.write("---")

        st.header("What are Cloudbursts?")
        st.write("A cloudburst is an extreme amount of precipitation in a short period of time,[1] sometimes accompanied by hail and thunder, which is capable of creating flood conditions. Cloudbursts can quickly dump large amounts of water, e.g. 25 mm of the precipitation corresponds to 25,000 metric tons per square kilometre (1 inch corresponds to 72,300 short tons over one square mile). However, cloudbursts are infrequent as they occur only via orographic lift or occasionally when a warm air parcel mixes with cooler air, resulting in sudden condensation. At times, a large amount of runoff from higher elevations is mistakenly conflated with a cloudburst.") 
        st.write("[To know more about this, click here >](https://en.wikipedia.org/wiki/Cloudburst)")

        st.header("Impact of Cloudbursts:")
        st.write("The impact of cloudbursts can be devastating, causing flooding, landslides, and loss of lives and property. Cloudbursts can be destructive on a large scale, especially in the mountains, causing floods, landslides, and mudflows that create terrible losses in the life and livelihood of the masses.")
        st.write("[To get more news related to cloudburst, press here >](https://shorturl.at/zKR36)")

        st.header("Safety Measures:")
        st.write("If you live in an area prone to cloudbursts, it's important to be prepared and follow safety guidelines.")
        st.write("[Read governmengt's directives here>](https://shorturl.at/gkPZ9)")

        st.write("---")

        st.header("**Stay safe and be prepared for extreme weather events!**")

        # Add content for the Home Page here
    elif page == "Team":
        #st.write("Meet Our Team!")
        st.write("**On a mission to ensure a safer tomorrow**")

        st.header("**Our Vision:**")
        st.write("At Cloud Phishers, our vision is to predict the upcoming clodburst with mwximum efficiency. We wish to raise awareness about cloudbursts and their impact on communities. We aim to foster a culture of preparedness and resilience in the face of these natural phenomena. Through education and outreach, we strive to minimize the adverse effects of cloudbursts and protect lives and property.")
        st.write("---")

        st.header("**Meet Our Team:**")

        team_members = [
                {"Name": "Jibitesh Chakraborty", "Designation": "Machine Learning Engineer"},
                {"Name": "Sagnik Basak", "Designation": "Web Application Developer"},
                {"Name": "Nilanjana Dutta", "Designation": "UI/UX designer"},
                {"Name": "Anidipta Pal", "Designation": "Data Analyst"},
                {"Name": "Ashmit Pal", "Designation": "Cyber Security Analyst"},
                {"Name": "Bhumika Adhya", "Designation": "Database Administrator"},
            ]

        for member in team_members:
         st.subheader(member["Name"])
         st.subheader(member["Designation"])

         st.write("---")
          
        st.header("**Contact Us:**")
        st.write("Have questions or want to get involved? Reach out to our team at:")
        st.write("Email: anidipta.pal.cloudphishers@gmail.com")

        st.write("**Together, we can make a difference!**")
        # Add content for the Our Team Page here
    elif page == "Predictions":
        st.subheader("Get Predictions Here!")
        st.write("**Our Model is Based on the Weekly Departure of Rainfall over a Period of 14-Weeks**")
        st.write("Yet to know about departure of rainfall?")
        st.write("[Click here to get more insights]>(https://shorturl.at/ntz69)")
        st.write("**We advice you to enter exact meteorological data for accurate results**")
        inputs = []
        for i in range(14):
            input_value = st.number_input(f"Week {i+1}", step=0.01)  # Allows positive and negative values
            inputs.append(input_value)

        if st.button("Submit"):
            # Process the input values when the submit button is clicked
            #st.write(str(inputs))
            X = np.array(inputs)
            X = X.reshape([-1,14])
            Y_KNN = KNN.predict(X)
            Y_DT = DT.predict(X)
            Y_RF = RF.predict(X)
            Y_Logistic = Logistic.predict(X)
            X = X.reshape([1,14,1])
            Y_CNNLSTM = CNNLSTM.predict(X)
            
            if Y_KNN == 1:
                st.write("Result of K-Nearest Neighbor Algorithm = Cloud Burst Positive")
            else :
                st.write("Result of K-Nearest Neighbor Algorithm = Cloud Burst Negative")

            if Y_DT == 1:
                st.write("Result of Decision Tree Algorithm = Cloud Burst Positive")
            else :
                st.write("Result of Decision Tree Algorithm = Cloud Burst Negative")

            if Y_RF == 1:
                st.write("Result of Random Forest Algorithm = Cloud Burst Positive")
            else :
                st.write("Result of Random Forest Algorithm = Cloud Burst Negative")

            st.write("Result of Logistic Regression = " + str(Y_Logistic[0]*100) + "%")
            st.write("Result of Stacked CNN-LSTM Model = " + str(Y_CNNLSTM[0][0]*100) + "%")
        # Add content for the Get Predictions Page here

    elif page == "Page-4":
        st.title("Welcome to Page-4")

if __name__ == "__main__":
    main()