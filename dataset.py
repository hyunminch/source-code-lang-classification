import os 

import torch
from torch.utils.data import Dataset


def chunkstring(text, length):
    '''
        Break the text into equal length strings so that
        they all have the same dimensions
    '''
    parts = [text[i: i + length] for i in range(0, len(text), length)]
    return parts

class CodeDataset(Dataset):
    def __init__(self, data_path, langs, max_length=1024):
        self.max_length = max_length
        self.langs = langs
        self.build_lang_indices()

        self.load(data_path)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        x = self.encode_to_tensor(code)
        lang = self.labels[idx]
        y = self.lang_indices[lang]
        return x, y

    def build_lang_indices(self):
        self.lang_indices = dict()
        for idx, lang in enumerate(self.langs):
            self.lang_indices[lang] = idx

    def encode_to_tensor(self, code):
        tensor = torch.zeros(128, self.max_length)

        for idx, ch in enumerate(code):
            o = ord(ch)
            if o >= 128:
                o = ord(' ')
            tensor[o][idx] = 1.0

        return tensor

    def load(self, data_path):
        self.codes = [] 
        self.labels = []

        for lang in self.langs:
            for item in os.listdir(data_path + "/" + lang):
                item_full_path = os.path.join(data_path + "/" + lang, item)

                with open(item_full_path) as f:
                    try:
                        text = f.read()
                        chunks = chunkstring(text, self.max_length)
                        for chunk in chunks:
                            self.codes.append(chunk)
                            self.labels.append(lang)
                    except Exception:
                        print('Exception occurred during reading...')
    