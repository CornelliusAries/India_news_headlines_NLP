# Project description:

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
To use our deployed depression classifier type this command to your Linux terminal <br />
curl -X POST "https://europe-west1-twitter-depression-classifier.cloudfunctions.net/model_predict-10" -H "Content-Type:application/json" --data '{"input_data":"Tweet to examine"}'

## Deployment on Hugging Face

https://huggingface.co/spaces/qisan/Depressed_sentimental_analysis

Where {"input_data":"Tweet to examine"}' is the tweet you want to examine.

## Authors:
Chen Liang <br />
Karol Kubala <br />
Kornel Kowalczyk 
