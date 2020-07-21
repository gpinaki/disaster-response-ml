# Disaster Response Pipeline And Web App

### Project Overview
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided into the following three key sections:

Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
Build a machine learning pipeline to train the which can classify text message in various categories
Run a web app which can show model results in real time. The Web App will also Data Visualizations based on the data available. 

### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - Run ETL pipeline that cleans the data and stores in the SQLLite database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Run ML pipeline that trains the classifier and saves 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or use the localhost for any issue

### Installation and dependencies
Python 3.5+
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
NLP Libraries: NLTK
SQLlite Database Libraries: SQLalchemy
Model Loading and Saving Library: Pickle
Web App and Data Visualization: Flask, Plotly, matplotlib


### Codebase Overview
*app/templates/*: templates/html files for web app
*data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and stores data in a SQLite database
*models/train_classifier.py: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use
*app/run.py: This file is used to launch the Flask web app. Some data visualizations are done through this run.py


#### Data Visualizations

![Web App Dashboard](/screenshots/App_dashboard.png)

![Reported Messages by Weather Conditions](/screenshots/relative_reporting_by_weather_condition.png)

![Casualties reported by various weather conditions](/screenshots/reported_casualties_by_weather_condition.png)

![Reported Messages by Genre](/screenshots/message_breakdown_by_genre.png)


### Licensing, Authors, Acknowledgements

This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.
