import torch
from transformers import EncoderDecoderModel, HerbertTokenizer

tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
# initialize Bert2Bert from pre-trained checkpoints
model = EncoderDecoderModel.from_encoder_decoder_pretrained("allegro/herbert-klej-cased-v1",
                                                            "allegro/herbert-klej-cased-v1")

# Save loaded model
model.save_pretrained('./models/allegro_seq2seq')
tokenizer.save_pretrained('./models/allegro_seq2seq')
