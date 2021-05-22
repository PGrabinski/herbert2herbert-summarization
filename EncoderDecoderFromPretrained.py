import torch
import transformers
import numpy as np
from transformers import EncoderDecoderModel, HerbertTokenizer, RobertaModel

tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
# initialize Bert2Bert from pre-trained checkpoints
model = EncoderDecoderModel.from_encoder_decoder_pretrained("allegro/herbert-klej-cased-v1", "allegro/herbert-klej-cased-v1") 

# Save loaded model
model.save_pretrained('allegro')