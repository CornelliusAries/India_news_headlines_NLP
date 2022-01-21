# Project description:
[OUTDATED] Based on the headline_text column provided by the dataset India News Headlines Dataset, determine what emotions may be associated with this article - Sentiment analysis.

Due to dataset issues we decided to change it for Sentimental Analysis for Tweets dataset which is a similar NLP problem.
## Goal of the project
Based on the "message to examine" column provided by the dataset, determine if author of the tweet is depressed or not.

## Project framework
In this project we plan to use PyTorch Transformers, which is a library of state-of-the-art pre-trained models for Natural Language Processing (NLP).

## Dataset
This dataset consists of 10282 unique tweets from both depressed and not depressed people.

https://www.kaggle.com/gargmanas/sentimental-analysis-for-tweets

## Models
In this project we want to focus on pretrained Bert model.

## Using the depression classifier
To use our deployed depression classifier type this command to your Linux terminal
curl -X POST "https://europe-west1-twitter-depression-classifier.cloudfunctions.net/model_predict-10" -H "Content-Type:application/json" --data '{"input_data":"Tweet to examine"}'

Where {"input_data":"Tweet to examine"}' is the tweet you want to examine.
