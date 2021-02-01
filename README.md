# Disaster Response Pipelines


### Table of Contents

1. [Project Motivation](#motivation)
2. [Project Description](#descriptions)
3. [Files Description](#files)
4. [Instructions](#instructions)
5. [Dependencies](#Dependencies)
6. [Acknowledgements](#Acknowledgements)
7. [Web App Screenshots](#Screenshots)


## Project Motivation<a name="motivation"></a>

The goal of the project is to analyze disaster data and build a Natural Language Processing (NLP) model for an API that classifies disaster messages. 
This dataset is provided by [Figure Eight](https://www.figure-eight.com/) which contains pre-labelled tweet and messages from real-life disaster events. 
Through a web app developed with this project, an emergency worker can input a new message and get a prediction for the response categories the message is likely to belong, reducing the potential reaction time of the responding organizations.

## Project Descriptions<a name = "descriptions"></a>
The project has three components:

1. **ETL Pipeline:** `process_data.py` file contain the script to create ETL pipline which:

- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. **ML Pipeline:** `train_classifier.py` file contain the script to create ML pipline which:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. **Flask Web App:** the web app enables the user to enter a disaster message, and then view the categories of the message.

The web app also contains some visualizations that describe the data. 
 
  
## Files Description <a name="files"></a>

The files structure is arranged as below:

	- README.md: read me file
	- ETL Pipeline Preparation.ipynb: contains ETL pipeline preparation code
	- ML Pipeline Preparation.ipynb: contains ML pipeline preparation code
	- workspace
		- \Data
			- disaster_categories.csv: categories dataset
			- disaster_messages.csv: messages dataset
			- DisasterResponse.db: disaster response database
			- process_data.py: ETL process

		- \app
			- run.py: flask file to run the app

		- \templates
			- master.html: main page of the web application 
			- go.html: result web page

		- \images
			- Screenshot_app_mainpage: app home page
			- Screenshot_app_message_classifier_page: app message classifier page
		- \models
			- train_classifier.py: classification code


## Instructions <a name="instructions"></a>

To execute the app follow the instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Dependencies <a name="Dependencies"></a>

- Python 3.*
- Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly


## Acknowledgements <a name="Acknowledgements"></a>

-[Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model


## Web App Screenshots <a name="Screenshots"></a>
1. Homepage
![Homepage](https://github.com/ankitaggarwal64/Disaster-Response-Pipelines/blob/main/images/Screenshot_app_mainpage.JPG)

2. Message classifier page
![Message classifier page](https://github.com/ankitaggarwal64/Disaster-Response-Pipelines/blob/main/images/Screenshot_app_message_classifier_page.JPG)

