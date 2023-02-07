import torch
import torch.nn as nn
import numpy as np
import math

class Multi_Head_Attention(nn.Module):
    def __init__ (self, model_dimension=512, h=8, key_dimension=64, value_dimension=64) :
        super(Multi_Head_Attention, self).__init__()
        self.model_dimension = model_dimension
        self.num_head = h
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension
        
        self.query_layer = nn.Linear(model_dimension, self.key_dimension)
        self.key_layer = nn.Linear(model_dimension, self.key_dimension)
        self.value_layer = nn.Linear(model_dimension, self.value_dimension)
        
        self.linear_transform = nn.Linear()
    
    ## query : sequence_length * key dimension
    ## key : sequence_length * key dimension
    ## value : sequence_length * value dimension
    def __calculate_attention(self, query, key, value) :
        attention = torch.mm(key, torch.transpose(query, 0, 1)) / math.sqrt(self.key_dimension) ## sequence length * sequence length 
        attention_score = torch.nn.functional.softmax(attention, dim=1) ## sequence length * sequence length 
        attention_value = torch.mm(attention_score, value) ## sequence length * value dimenstion 
        return attention_value
        
    def forward (self, query, key, value, is_masked = False) :
        q1 = self.query_layer(query)
        k1 = self.key_layer(key)
        v1 = self.value_layer(value)
        
        output = [self.__calculate_attention(query=q1, key=k1, value=v1) for i in range(self.num_head)]
        return self.__calculate_attention(query=q1, key=k1, value=v1)
        
        
        
class Embedding :
    def __init__ (self, vocal_size, model_dimension=512,) :
        super(Encoder, self).__init__()
        self.model_dimension = 512
        self.embedding_layer = nn.Embedding(num_embeddings=vocal_size, 
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
        embedded_x = self.embedding_layer(x)
        position_x = self.__make_positional_encodings(len(x))
        return embedded_x + position_x


class Encoder :
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
        
        
    
class Decoder :
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
    
    