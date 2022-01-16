from torch.utils.data import Dataset
import torch
class TweetsDataset(Dataset):

  def __init__(self, message, depression, tokenizer, max_len):
    self.message = message
    self.depression = depression
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.message)
  
  def __getitem__(self, item):
    message = str(self.message[item])
    depression = self.depression[item]

    encoding = self.tokenizer.encode_plus(
      message,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'tweet_text': message,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'depression': torch.tensor(depression, dtype=torch.long)
    }
    