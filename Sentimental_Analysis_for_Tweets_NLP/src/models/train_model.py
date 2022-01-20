from data import DataModule
from model import DepressionClassifier
import pytorch_lightning as pl

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 10

data_module = DataModule('data/raw/sentiment_tweets3.csv', BATCH_SIZE)
model = DepressionClassifier(2, 'bert-base-cased')
# trainer = pl.Trainer(auto_scale_batch_size='power', gpus=1, deterministic=True, max_epochs=5)
trainer = pl.Trainer(max_epochs = 10)
trainer.fit(model, data_module)