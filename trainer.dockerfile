# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY Sentimental_Analysis_for_Tweets_NLP/requirements.txt requirements.txt
COPY Sentimental_Analysis_for_Tweets_NLP/setup.py setup.py
COPY Sentimental_Analysis_for_Tweets_NLP/src/ src/
COPY Sentimental_Analysis_for_Tweets_NLP/data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py", "models", "data/raw", "reports/figures"]