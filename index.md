---
layout: default
title: Portfolio
nav_order: 1
---
**Welcome to my portfolio**  
Below is a list of school and personal projects. Each has a description and links to a final report, source code, or a notebook.

**Table of Contents:**
- [**Personal Projects**](#personal-projects)
  - [**Exploratory Data Analysis, Data Collection, and Cleaning**](#exploratory-data-analysis-data-collection-and-cleaning)
    - [*Scraping the IWF Website for Event Results And Athlete Data*](#scraping-the-iwf-website-for-event-results-and-athlete-data)
  - [**Machine Learning**](#machine-learning)
    - [*Disaster Tweet Classification in R Using a linear SVM model*](#disaster-tweet-classification-in-r-using-a-linear-svm-model)
    - [*Predicting Survivors of the Titanic in Python*](#predicting-survivors-of-the-titanic-in-python)
    - [*Recognizing Handwritten Digits with a Convolutional Neural Network using Python and TensorFlow*](#recognizing-handwritten-digits-with-a-convolutional-neural-network-using-python-and-tensorflow)
  - [**SQL and General Programming**](#sql-and-general-programming)
    - [*Recreating Wordle in Python with a SQL Database for Scores*](#recreating-wordle-in-python-with-a-sql-database-for-scores)
- [**Past School Projects**](#past-school-projects)
  - [**Visualization**](#visualization)
    - [*Visualization and Exploration of the Gapminder Dataset Using ggplot/tidyverse Packages in R*](#visualization-and-exploration-of-the-gapminder-dataset-using-ggplottidyverse-packages-in-r)
  - [**Time Series Analysis**](#time-series-analysis)
    - [*Analysis of Homicides in the US Over Time Using R and an ARMA/SARIMA Model*](#analysis-of-homicides-in-the-us-over-time-using-r-and-an-armasarima-model)
  - [**Regression Analysis**](#regression-analysis)
    - [*Reproducing the Results and Logistic Regression Model of a study on Modeling Prison Sentencing From Facial Features*](#reproducing-the-results-and-logistic-regression-model-of-a-study-on-modeling-prison-sentencing-from-facial-features)
  - [**Machine Learning**](#machine-learning-1)
    - [*Predicting Ebay Car Prices Using a Random Forest Model in R*](#predicting-ebay-car-prices-using-a-random-forest-model-in-r)
    - [*Fitting a Bayesian Hierarchical Model on Fake Flu Data. Simulated with an MCMC algorithm using R and Rjags/jags.*](#fitting-a-bayesian-hierarchical-model-on-fake-flu-data-simulated-with-an-mcmc-algorithm-using-r-and-rjagsjags)

<br /><br />

# **Personal Projects**

## **Exploratory Data Analysis, Data Collection, and Cleaning**

### *Scraping the IWF Website for Event Results And Athlete Data*

The repository is [github.com/cluffa/IWF_data](https://github.com/cluffa/IWF_data). I scraped many pages for all event results and athlete stats from the International Weightlifting Federation's website. I used python for scraping. I needed the data to be easily imported into R, so I used R to clean and save the data.

I am currently exploring the data and working on an analysis. It will cover some topics such as competition strategy, comparison of countries and athletes, and predicting PED use.

Work in progress analysis [here](https://cluffa.github.io/IWF_data/).

## **Machine Learning**

### *Disaster Tweet Classification in R Using a linear SVM model*  

View this project's [R Notebook](https://cluffa.github.io/disaster_tweets_nlp_svm/).  

The data is a collection of tweets that have been labeled as pertaining to a disaster or not. For example, one might be about the damage of an earthquake while another is about a sports team. Each tweet has a text body, keyword, and location. I used a linear support vector machine (SVM) model and tested the model with combinations of text body, keyword, and location.

### *Predicting Survivors of the Titanic in Python*  

View this project's [Jupyter Notebook](https://github.com/cluffa/titanic/blob/master/titanicV2.ipynb). Most of the graphing and data exploration was done in the [first version of the notebook](https://github.com/cluffa/titanic/blob/master/titanic.ipynb) where I added no features.

```
Id  Survived  Pclass                                            Name     Sex   Age  SibSp  Parch         Ticket     Fare Cabin Embarked
 1         0       3                       Braund, Mr. Owen Harris    male  22.0      1      0      A/5 21171   7.2500   NaN        S
 2         1       1  Cumings, Mrs. John Bradley (Florence Brig...  female  38.0      1      0       PC 17599  71.2833   C85        C
 3         1       3                        Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3...   7.9250   NaN        S
 4         1       1  Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0         113803  53.1000  C123        S
 5         0       3                      Allen, Mr. William Henry    male  35.0      0      0         373450   8.0500   NaN        S
```
 
 The main goal was to try to predict survivors based on what we know about each passenger. I used scikit-learn pipelines to make a clear transformation pipeline for the data. This includes encoding, multivariate imputing, as well as training. I used a gradient boosting classifier model where hyperparameters were optimized by grid search and cross-validation.

This notebook was used to submit scores to Kaggle's "Titanic: Machine Learning From Disaster" competition. With feature engineering like multivariate imputing and matching families, I achieved an accuracy score of 0.801 when submitting. This put me in the top 5% of the leader board.

### *Recognizing Handwritten Digits with a Convolutional Neural Network using Python and TensorFlow*  

View this project's [Jupyter Notebook](https://github.com/cluffa/digit_recognizer/blob/master/digits_tfnn.ipynb) using neural networks and [Jupyter Notebook](https://github.com/cluffa/digit_recognizer/blob/master/digits.ipynb) using a standard machine learning model.  

![digits](./images/digits.png)

This is another Kaggle competition. The goal was to classify handwritten digits like the ones above. I wanted to compare the accuracy of traditional machine learning models with a convolutional neural network (CNN). I achieved 97.4% testing accuracy with an XGBoost model and 99.1% with a CNN using a TensorFlow keras sequential model. The traditional model did much better than I was expecting. However, these images are centered and scaled to be similar to each other. In a more uncontrolled environment I would expect the accuracy of the traditional type of model to drop off.

## **SQL and General Programming**

### *Recreating Wordle in Python with a SQL Database for Scores*

Repository: <https://github.com/cluffa/wordpy>

| Game      | Game History/Leader board |
| ----------- | ----------- |
| ![game](images/gameplay.png)  | ![stats](images/stats.png) |  

**Some of the queries:**  

| Logging | Rankings | Personal Game History |
| ----------- | ----------- | ----------- |
| ![example](images/queries1.png) | ![example](images/queries2.png) | ![example](images/queries3.png) |  

This was a fun side project where my goal was to recreate wordle and use a SQL database for leader boards. I was able to further my understanding of databases and postgreSQL by using elephantSQL to host a table of the scores. Every time a game is completed, a log is added to the database with a name, date, word, number of guesses, and some other metrics. Then a query is executed to display basic stats and game history.

<br /><br />

# **Past School Projects**

## **Visualization**

### *Visualization and Exploration of the Gapminder Dataset Using ggplot/tidyverse Packages in R*

View this project's [final report](https://github.com/cluffa/stat5730project/blob/master/final_report_Alex_Cluff.pdf) and [source code](https://github.com/cluffa/stat5730project/blob/master/final_report_Alex_Cluff.Rmd).  

![ggplot graph](https://github.com/cluffa/stat5730project/raw/master/final_report_Alex_Cluff_files/figure-gfm/unnamed-chunk-4-1.png)  

The focus of this project was to explore the famous Gapminder dataset visually using ggplot graphs like the one above. I also used the other tidyverse packages like dplyr to manipulate the data in different ways to create well formatted data that fits into summary tables.

## **Time Series Analysis**

### *Analysis of Homicides in the US Over Time Using R and an ARMA/SARIMA Model*

View this project's [final report](https://cluffa.github.io/stat5550project/) and [source code](https://github.com/cluffa/stat5550project/blob/master/final_project_Alex_Cluff.Rmd).  
![homicide predictions](https://cluffa.github.io/stat5550project/index_files/figure-html/unnamed-chunk-13-1.png)  

This project was based around forecasting methods. I find overall trends, monthly seasonality, and fit ARIMA and SARIMA models. I compare the two model's performance as well as fit. I then forecasted homicides for the next 24 months.

## **Regression Analysis**

### *Reproducing the Results and Logistic Regression Model of a study on Modeling Prison Sentencing From Facial Features*

View this project's [final report](https://github.com/cluffa/stat3302project/blob/main/group_project.pdf) and [source code](https://github.com/cluffa/stat3302project/blob/main/model.R).  

![model coefficient table](https://github.com/cluffa/stat3302project/blob/main/table2.png?raw=true)  

The idea for this assignment was to gain experience and become more comfortable reading and interpreting scientific research papers. We also learned the importance of reproducibility and transparency. My group was tasked with reproducing the results and models from this paper and reporting on them. I was in charge of the modeling as well as the table for the models, both of which are created with the source code I linked. The picture above is a replication of the table used in the original paper. Interestingly, we ended up finding a small mistake in the paper.

## **Machine Learning**

### *Predicting Ebay Car Prices Using a Random Forest Model in R*

View this project's [final report](https://github.com/cluffa/stat4620project/blob/master/final_report_made_in_colaboration_with_classmates.pdf), [source code for the random forest model](https://github.com/cluffa/stat4620project/blob/master/randomForest.R), and [source code for cleaning the data](https://github.com/cluffa/stat4620project/blob/master/clean_autos_dataset.R).  

This was a group project. I handled the random forest model as well as the data cleaning. We each tried a model and compared results. The random forest model came out on top based on testing MSE. I had to learn a lot about resource allocation to complete this project. The dataset had 180k rows and 10+ possible predictors. I quickly found out that I would not be able to easily tune and train the model. The training process ended up taking a few days with cross-validation on less than half of the full dataset. I had to weigh run time vs accuracy and pick model parameters early on in the training process.

### *Fitting a Bayesian Hierarchical Model on Fake Flu Data. Simulated with an MCMC algorithm using R and Rjags/jags.*

View this project's [final report](https://github.com/cluffa/stat3303project/blob/master/Final_project_Alex_Cluff.pdf), [report source code](https://github.com/cluffa/stat3303project/blob/master/Final_project_Alex_Cluff.Rmd), and [model fitting source code](https://github.com/cluffa/stat3303project/blob/master/fit.R)  

The setup for this project:  
"There are two tests for influenza strain K9C9. The data collected consists of 10 countries and 100 pairs of
test results. The more accurate of the tests will be assumed fact. The less accurate test, EZK, is the area of
interest for this project. A Bayesian hierarchical model will be fit and it will be simulated with an MCMC
algorithm using R/jags."  
I fit the model, assessed fit, and interpreted the results in the context of a global pandemic.
