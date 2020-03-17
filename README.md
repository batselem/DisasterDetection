# Disaster Response Pipeline Project

### Required Libraries

- nltk 3.3.0
- numpy 1.15.2
- pandas 0.23.4
- scikit-learn 0.20.0
- sqlalchemy 1.3.15
- plotly
- flask

### Project Motivation
Nowadays, microblogging becomes the heart of online communications and the means for people to express their personal opinion about various things that are happening around the world. The speed of microblogging is much faster than other emergency services. That is why it is so intriguing to learn from it and use it for reporting on natural disasters. So, in this project, I tried to detect disasterous events instantly in real time and categorize emergency messages about disasters into different categories such as flood and earthquake.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgements
I'd like to thank reviews from udacity and Figure Eight for providing the dataset.
