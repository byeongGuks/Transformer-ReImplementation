import torch
import torch.nn as nn
import numpy as np
import math

class Multi_Head_Attention(nn.Module):
    def __init__ (self, model_dimension=512, h=8) :
        super(Multi_Head_Attention, self).__init__()
        self.model_dimension = model_dimension
        self.num_head = h
        self.key_dimension = model_dimension / h
        self.value_dimension = model_dimension / h
        
        self.query_layer = nn.Linear(model_dimension, model_dimension)
        self.key_layer = nn.Linear(model_dimension, model_dimension)
        self.value_layer = nn.Linear(model_dimension, model_dimension)
        
        self.linear_transform = nn.Linear(model_dimension, model_dimension)
        
        
    ## query :  key dimension
    ## key : key dimension
    ## value : value dimension
    def __calculate_attention(self, query, key, value) :
        attention = torch.mm(key, torch.transpose(query, 0, 1)) / math.sqrt(self.key_dimension) ## dot product attention / key dimension
        attention_score = torch.nn.functional.softmax(attention, dim=1) 
        attention_value = torch.mm(attention_score, value) ## sequence length * value dimenstion 
        return attention_value
        
    def forward (self, query, key, value, is_masked = False) :
        queries = self.query_layer(query)
        keys = self.key_layer(key)
        values = self.value_layer(value)
        
        ## multi head attention
        output = [self.__calculate_attention(query=queries[i*self.key_dimension::(i+1)*self.key_dimension], 
                                             key=keys[i*self.key_dimension::(i+1)*self.key_dimension], 
                                             value=values[i*self.key_dimension::(i+1)*self.key_dimension]) for i in range(self.num_head)]
        
        output = self.linear_transform(output)
        return output
        
        
        
class Embedding(nn.Module):
    def __init__ (self, vocab_size, model_dimension=512,) :
        super(Embedding, self).__init__()
        self.model_dimension = 512
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, 
                                            embedding_dim=model_dimension, 
                                            padding_idx=0)
        
    def __make_positional_vector (self, pos) :
        return [pos/np.power(10000, 2*(hidden_i//2)/ self.model_dimension) for hidden_i in range(self.model_dimension)]
        
    
    def __make_positional_encodings (self, sequence_length) :
        positional_encodings = np.array([self.__make_positional_vector(i) for i in range(sequence_length)])
        positional_encodings[:, 0::2] = np.sin(positional_encodings[:, 0::2])
        positional_encodings[:, 1::2] = np.cos(positional_encodings[:, 1::2])
        return positional_encodings

    def forward (self, x) :
        embedded_x = self.embedding_layer(x) ## todo : . In the embedding layers, we multiply those weights by √dmodel.
        position_x = self.__make_positional_encodings(len(x))
        return embedded_x + position_x


class Encoder(nn.Module):
    def __init__ (self, model_dimension = 512, fc_dimension=2048) :
        super(Encoder, self).__init__()
        self.multi_head_attention = Multi_Head_Attention()
        self.fc_layer = nn.Sequential(
            nn.Linear(model_dimension, fc_dimension),
            nn.ReLU(),
            nn.Linear(fc_dimension, model_dimension)
        )
        self.layer_norm = nn.LayerNorm(model_dimension)
        
    
    def forward(self, x) :
        ## attention layer
        h1 = self.multi_head_attention(query=x, key=x, value=x)
        out1 = self.layer_norm(x + h1)
        
        ## fully connected layer
        h2 = self.fc_layer(out1)
        out2 = self.layer_norm(out1 + h2)
        return out2
        
        
    
class Decoder(nn.Module):
    def __init__ (self, model_dimension = 512, fc_dimension=2048) :
        super(Decoder, self).__init__()
        self.multi_head_attention = Multi_Head_Attention()
        self.fc_layer = nn.Sequential(
            nn.Linear(model_dimension, fc_dimension),
            nn.ReLU(),
            nn.Linear(fc_dimension, model_dimension)
        )
        self.layer_norm = nn.LayerNorm(model_dimension)
    
    def forward(self, x, y) :
        ## masked multi head attention
        h1 = self.multi_head_attention(query=y, key=y, value=y, is_masked = True)
        out1 = self.layer_norm(y + h1)
        
        
        ## multi head attention
        h2 = self.multi_head_attention(query=y, key=x, value=x)
        out2 = self.layer_norm(out1 + h2)
        
        ## fully connected layer
        h3 = self.fc_layer(out2)
        out3 = self.layer_norm(out2 + h3) 
        return out3
    
class TransFormerModel(nn.Module) :
    def __init__(self, model_dimension = 512, num_head = 8, num_encoder = 6, num_decoder = 6, vocab_size=32000):
        self.model_dimension = model_dimension
        self.num_head = num_head
        self.num_encoder = num_encoder
        self.num_decoder = num_decoder
        self.vocab_size = vocab_size
        self.encoders = [Encoder(self.model_dimension) for i in range(num_encoder)]
        self.decoders = [Decoder(self.model_dimension) for i in range(num_encoder)]
        self.in_embedding = Embedding(self.model_dimension)
        self.out_embedding = Embedding(self.model_dimension)
        self.Linear = nn.Linear(model_dimension, model_dimension)
                
    
    def forward(self, x, y):
        embedded_x = self.in_embedding(x)
        embedded_y = self.out_embedding(y) 
        
        for encoder in self.encoders :
            embedded_x = encoder(embedded_x)
        
        for decoder in self.decoders :
            embedded_y = decoder(embedded_x, embedded_y)
        
        output = self.Linear(embedded_y)
        output = torch.nn.functional.softmax(output, dim=1) 
        return output

    
def test():
    print("---start test---")
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    
    input = torch.tensor([1,3,5,7])
    output = torch.tensor([2,4,6,8])
    model = TransFormerModel(model_dimension=512, num_head=1, num_encoder=1, num_decoder=1, vocab_size=20).to(device)
    output = model(input, output)
    print(output)

    
if __name__ == "__main__":
    print("test")
    test()
