# Next Generation Job Search
NLP project - Search Engine for Data Science jobs using SBERT model.

A prototype implementation for a web based job search engine.

This code finds the most relevant parts of each job description
This code uses an advanced NLP model and creative preprocessing techniques to find the most relevant job for you!

### Web search UI

Just type the job that you are looking for and press Enter,
and the jobs that best fit your query will appear at the top of the list!

![](UI.JPG)

Each job description is preprocessed to keep only the text that describes the job specifications.
A dadabase of vector represantations is created, and the user query is compared to jobs in the database using Cosine Similarity. 

### Basic Software Architecture

![](Architecture.JPG)

The sample data is taken from [https://www.kaggle.com/code/ranand60/analysis-of-job-posting-data-scientist-in-us/data](https://www.kaggle.com/code/ranand60/analysis-of-job-posting-data-scientist-in-us/data).

### Install requirements 
```
pip install -r requirements.txt
```
### Run Program
In order to run a job search:
```
python app.py
```
### Open UI
Enter link to LOCALHOST:5000: http://127.0.0.1:5000
