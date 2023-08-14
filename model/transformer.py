import torch 
import torch.nn as nn
import numpy as np
import math

class Embedding(nn.Module) :
    def __init__(self, vocab_size=32000, d_model=512) :
        super(Embedding, self).__init__()
        self.d_model = d_model = 512
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=d_model,
                                            padding_idx=0) ## the entries at padding_idx do not contribute to the gradient
    
    def positional_vector(self, pos) :
        return [pos/np.power(10000, 2 * i / self.d_model) for i in range(self.d_model)]
    
    def positional_encode(self, sequence_length) :
        positional_encodings = np.array([self.positional_vector(pos) for pos in range(sequence_length)])
        positional_encodings[:, 0::2] = np.sin(positional_encodings[:, 0::2])
        positional_encodings[:, 1::2] = np.cos(positional_encodings[:, 1::2])
        
        positional_encodings = np.pad(positional_encodings, 
                                      (0, self.d_model - sequence_length), 
                                      'constant', 
                                      constant_values=0)
        
        positional_encodings = torch.FloatTensor(positional_encodings)
        
        if torch.cuda.is_available() :
            positional_encodings = positional_encodings.cuda()
        return positional_encodings
    
    def forward(self, x) :
        embedded_x = self.embedding_layer(x)
        positon_x = self.positional_encode(sequence_length = x.size(dim=1))
        
        return embedded_x + positon_x

class AttentionLayer(nn.Module) :
    def __init__(self, d_model=512, n_head=8, is_masked=False) :
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.is_masked = is_masked
        
        self.d_query = d_model // n_head
        self.d_key = d_model // n_head
        self.d_value = d_model // n_head
        
        ## for simple implement use one 512 * 512 layer rather than eight 512 * 64 layer
        self.query_layer = nn.Linear(self.d_model, self.d_model, dtype=torch.float32) 
        self.key_layer = nn.Linear(self.d_model, self.d_model, dtype=torch.float32) 
        self.value_layer = nn.Linear(self.d_model, self.d_model, dtype=torch.float32)
        
        self.linear = nn.Linear(self.d_model, self.d_model, dtype=torch.float32) 
        
    def calculate_attention(self, query, key, value) :
        attention = torch.bmm(key, query.transpose(1, 2)) / math.sqrt(self.d_key)
        
        if self.is_masked :
            masking_matrix = torch.triu(torch.ones(attention.size()[1], attention.size()[2])) * (-1.0e9)
            if torch.cuda.is_available() :
                masking_matrix = masking_matrix.cuda()
            attention = attention + masking_matrix
        
        attention_score = nn.functional.softmax(attention, dim=1)
        attention_value = torch.bmm(attention_score, value)
        return attention_value
    
    def forward(self, query, key, value) :
        queries = self.query_layer(query)
        keys = self.key_layer(key)
        values = self.value_layer(value)
        
        ## multi head attention
        output = torch.cat([self.calculate_attention(query=queries[i*self.d_query:(i+1)*self.d_query], 
                                                    key=keys[i*self.d_key:(i+1)*self.d_key], 
                                                    value=values[i*self.d_value:(i+1)*self.d_value]) 
                                                    for i in range(self.n_head)], dim=0)
        
        output = self.linear(output)
        return output

class Encoder(nn.Module) :
    def __init__(self, d_model=512, n_head=8, d_fc=2048):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_fc = d_fc
        self.multihead_attention = AttentionLayer(d_model=self.d_model, n_head=self.n_head)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_fc, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(self.d_fc, self.d_model, dtype=torch.float32)
        )
        self.layer_norm = nn.LayerNorm(self.d_model, dtype=torch.float32)
        
    def forward(self, x) :
        h1 = x + self.multihead_attention(query=x, key=x, value=x)
        out1 = self.layer_norm(x + h1)
        
        h2 = self.ffn(out1)
        out2 = self.layer_norm(out1 + h2)
        return out2

class Decoder(nn.Module) :
    def __init__(self, d_model=512, d_fc=2048) :
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.d_fc = d_fc
        
        self.multihead_attention = AttentionLayer(d_model=self.d_model, n_head=self.n_head)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_fc, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(self.d_fc, self.d_model, dtype=torch.float32)
        )
        self.layer_norm = nn.LayerNorm(self.d_model, dtype=torch.float32)
        
    def forward(self, x, y) :
        h1 = self.multihead_attention(query=y, key=y, value=y, is_masked=True)
        out1 = self.layer_norm(y + h1)
        
        h2 = self.multihead_attention(query=y, key=x, value=x)
        out2 = self.layer_norm(out1 + h2)
        
        h3 = self.ffn(out2)
        out3 = self.layer_norm(out3 + h3)
        return out3
        
class Transformer(nn.Module) :
    def __init__(self, 
                 d_model=512, 
                 n_head=8, 
                 n_encoder=6, 
                 n_decoder=6,
                 vocab_size=32000) :
        
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.vocab_size = vocab_size
        
        self.encoders = [Encoder(d_model=self.d_model, n_head=self.n_head) for i in range(self.n_encoder)]
        self.decoders = []
        self.in_embedding = Embedding(vocab_size=self.vocab_size, d_model=self.d_model)
        self.out_embedding = Embedding(vocab_size=self.vocab_size, d_model=self.d_model)
        self.linear = torch.nn.Linear(d_model, d_model, dtype=torch.float32)

    def forward(self, x, y) :
        encoded_x = self.encode(x)
        output = self.decode(encoded_x, y)
        return output 
    
    def encode(self, x) :   
        encoded_x = self.in_embedding(x)
        for encoder in self.encoders : 
            encoded_x = encoder(encoded_x)
        return encoded_x
        
    def decode(self, encoded_x, y) :
        ## docode output
        decoded_y = self.out_embedding(y)
        for decoder in self.decoders :
            decoded_y = decoder(encoded_x, decoded_y)
        
        ## convert decoder output to predicted next token probabilities
        print(decoded_y.dtype)

        output = self.linear(decoded_y)
        output = nn.functional.softmax(output, dim=1)
        return output
    
def test():
    print("---start test---")
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    
    input = torch.LongTensor([[i for i in range(512)] for j in range(128)])
    output = torch.LongTensor([[i for i in range(512)] for j in range(128)])
    model = Transformer(d_model=512, n_head=8, n_encoder=2, n_decoder=2, vocab_size=1000).to(device)

    output = model(input, output)
    print(output)
    print(output[0].shape)

    
if __name__ == "__main__":
    print("test")
    test()