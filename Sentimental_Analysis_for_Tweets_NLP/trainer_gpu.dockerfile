# Base image
FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04  
# or whatever image you want to use

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /app/ 
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py", "models", "data/raw", "reports/figures"]