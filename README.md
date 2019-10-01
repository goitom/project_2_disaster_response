# Disaster Response Pipeline Project

### Project Description
This project implements a NLP classifier on text messages generated during natural disasters. Based on the content of the message, the algorithm assigns the message to one or more categories, assisting first responders in more accurately assessing needs on the ground. The associated web app allows a user to enter a message, and see how the algorithm classifies it.

![image](plot1_snapshot.jpg)

![image](plot2_snapshot.jpg)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
