from tqdm import tqdm
import torch

class ShiftingDataset:

    def __init__(self, seq_length):
        self.tokens = []
        self.data = []
        self.seq_length = seq_length

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        if len(self.data) == 0:
            raise Exception('error: chunks have not been generated yet')
        return len(self.data)

    def preprocess(self, dataset, tokenizer):
        '''
        Pretokenizes the dataset and returns token Tensors of size `seq_length`.
        '''
        print('Preprocessing dataset of length', len(dataset))
        for entry in tqdm(dataset):
            # add document tokens + end of text token
            self.tokens.extend(tokenizer(entry['content'])['input_ids'] + [tokenizer.eos_token_id])
        print('Total token count:', len(self.tokens))
    
    def regenerate(self, offset=0):
        '''
        Regenerates dataset chunks, offsetting by `offset` tokens to increase diversity.
        '''
        print('Regenerating with offset', offset)
        # offset master token list
        if offset > 0:
            self.tokens = self.tokens[offset:] + self.tokens[:offset]
        # create new chunks
        self.data = []
        for i in tqdm(range(0, len(self.tokens), self.seq_length)):
            chunk = self.tokens[i:i+self.seq_length]
            # pad 0 to input and -100 to labels if len(chunk) < seq_length
            input_ids = chunk + [0] * (self.seq_length - len(chunk))
            labels = chunk + [-100] * (self.seq_length - len(chunk))
            # attention mask is all 1s unless len(chunk) < seq_length
            attention_mask = [1] * (len(chunk)) + [0] * (self.seq_length - (len(chunk)))
            assert len(input_ids) == len(attention_mask)
            self.data.append({'input_ids': torch.LongTensor(input_ids),
                              'attention_mask': torch.LongTensor(attention_mask),
                              'labels': torch.LongTensor(labels)})