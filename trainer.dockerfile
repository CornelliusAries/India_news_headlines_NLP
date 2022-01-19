# Base image
FROM python:3.7-slim

WORKDIR /root
# install python 
RUN apt update && \ 
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

COPY Sentimental_Analysis_for_Tweets_NLP/requirements.txt /root/requirements.txt
COPY Sentimental_Analysis_for_Tweets_NLP/setup.py /root/setup.py
COPY Sentimental_Analysis_for_Tweets_NLP/src/ /root/src/
COPY Sentimental_Analysis_for_Tweets_NLP/data/ /root/data/


RUN pip install -r /root/requirements.txt --no-cache-dir

ENTRYPOINT ["python", "/root/src/models/train_model_gcp.py"]