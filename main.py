from torch.utils.data import DataLoader
from tokenize.encoder import WordEncoder, GloveEncoder
from data.dataset import NewsDataset
from models.model import NewsClassifier, GloveClassifier
from training.early_stopping import EarlyStopping
from training.trainer import Trainer
from option.config import Config
import torch
import torch.nn as nn
import data.data_processing as dp

# load config
CONFIG_PATH = './option/config.json'
cfg = Config(CONFIG_PATH)

# set model options
EMBEDDING_DIM = cfg.get('MODEL.EMBEDDING_DIM')
GLOVE_NAME = cfg.get('MODEL.GLOVE_NAME')
GLOVE_DIM = cfg.get('MODEL.GLOVE_DIM')
HIDDEN_DIM = cfg.get('MODEL.HIDDEN_DIM')
OUTPUT_DIM = cfg.get('MODEL.OUTPUT_DIM')
LEARNING_RATE = cfg.get('MODEL.LEARNING_RATE')
FREEZE = cfg.get('MODEL.GLOVE_FREEZE')

# set train options
MODEL_PATH = cfg.get('TRAIN.MODEL_PATH')
BATCH_SIZE = cfg.get('TRAIN.BATCH_SIZE')
EPOCHS = cfg.get('TRAIN.EPOCHS')
ELMO_MODE = cfg.get('TRAIN.ELMO_MODE')
ELMO_PATH = './option/elmo_config.json'

# set scheduler setting
SCHED_MODE = cfg.get('TRAIN.SCHEDULER.MODE')
SCHED_FACTOR = cfg.get('TRAIN.SCHEDULER.FACTOR')
SCHED_PATIENCE = cfg.get('TRAIN.SCHEDULER.PATIENCE')
SCHED_VERBOSE = cfg.get('TRAIN.SCHEDULER.VERBOSE')

# set early stopping options
MODEL_PATH = cfg.get('ES.MODEL_PATH')
GLOVE_MODEL_PATH = cfg.get('ES.GLOVE_MODEL_PATH')
PATIENCE = cfg.get('ES.PATIENCE')
VERBOSE = cfg.get('ES.VERBOSE')
DELTA = cfg.get('ES.DELTA')

# load data
train_data = dp.load_data(f"{MODEL_PATH}/AGtrain.csv")
test_data = dp.load_data(f"{MODEL_PATH}/AGtest.csv")

# extract topic data
train_topic = train_data.loc[:, ["Class Index", "Title"]].reset_index(
    drop=True).sort_values(by="Class Index").reset_index(drop=True)
train_topic = train_topic.rename(columns={
    "Class Index": "category",
    "Title": "title"
})
test_topic = test_data.loc[:, ["Class Index", "Title"]].reset_index(
    drop=True).sort_values(by="Class Index").reset_index(drop=True)
test_topic = test_topic.rename(columns={
    "Class Index": "category",
    "Title": "title"
})

# subtract 1 from category index for preventing cuda error
train_topic['category'] -= 1
test_topic['category'] -= 1

# create encoder
word_encoder = WordEncoder()
glove_encoder = GloveEncoder(name=GLOVE_NAME, dim=GLOVE_DIM)

# create dataset
train_dataset = NewsDataset(train_topic,
                            word_encoder,
                            build_vocab=True,
                            elmo_mode=ELMO_MODE)
test_dataset = NewsDataset(test_topic, word_encoder, elmo_mode=ELMO_MODE)
glove_train_dataset = NewsDataset(train_topic,
                                  glove_encoder,
                                  elmo_mode=ELMO_MODE)
glove_test_dataset = NewsDataset(test_topic,
                                 glove_encoder,
                                 elmo_mode=ELMO_MODE)

# create dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
glove_train_loader = DataLoader(glove_train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
glove_test_loader = DataLoader(glove_test_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

# News Classifier
classifier = NewsClassifier(len(word_encoder.vocab),
                            EMBEDDING_DIM,
                            HIDDEN_DIM,
                            OUTPUT_DIM,
                            elmo_mode=ELMO_MODE,
                            elmo_option=ELMO_PATH)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, SCHED_MODE,
                                                       SCHED_FACTOR,
                                                       SCHED_PATIENCE,
                                                       SCHED_VERBOSE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
early_stopping = EarlyStopping(MODEL_PATH, PATIENCE, VERBOSE, DELTA)
trainer = Trainer(classifier,
                  criterion,
                  optimizer,
                  scheduler,
                  device,
                  early_stopping,
                  elmo_mode=ELMO_MODE)
trainer.fit(train_loader, test_loader, epochs=EPOCHS)
trainer.loss_graph()
trainer.metric_graph()

# Glove Classifier
glove_classifier = GloveClassifier(glove_encoder.glove.vectors,
                                   HIDDEN_DIM,
                                   OUTPUT_DIM,
                                   freeze=FREEZE,
                                   elmo_mode=ELMO_MODE,
                                   elmo_option=ELMO_PATH)
optimizer = torch.optim.Adam(glove_classifier.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, SCHED_MODE,
                                                       SCHED_FACTOR,
                                                       SCHED_PATIENCE,
                                                       SCHED_VERBOSE)
early_stopping = EarlyStopping(GLOVE_MODEL_PATH, PATIENCE, VERBOSE, DELTA)
trainer = Trainer(glove_classifier,
                  criterion,
                  optimizer,
                  scheduler,
                  device,
                  early_stopping,
                  elmo_mode=ELMO_MODE)
trainer.fit(glove_train_loader, glove_test_loader, epochs=EPOCHS)
trainer.loss_graph()
trainer.metric_graph()
