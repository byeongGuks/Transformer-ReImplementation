import torch
import torch.nn as nn
import numpy as np
import math

class Multi_Head_Attention(nn.Module):
    def __init__ (self, model_dimension=512, h=8) :
        super(Multi_Head_Attention, self).__init__()
        self.model_dimension = model_dimension
        self.num_head = h
        self.key_dimension = int(model_dimension / h)
        self.value_dimension = int(model_dimension / h)
        
        self.query_layer = nn.Linear(model_dimension, model_dimension, dtype=torch.float64)
        self.key_layer = nn.Linear(model_dimension, model_dimension, dtype=torch.float64)
        self.value_layer = nn.Linear(model_dimension, model_dimension, dtype=torch.float64)
        
        
        self.linear_transform = nn.Linear(model_dimension, model_dimension, dtype=torch.float64)
        
        
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
        output = torch.cat([self.__calculate_attention(query=queries[i*self.key_dimension:(i+1)*self.key_dimension], 
                                             key=keys[i*self.key_dimension:(i+1)*self.key_dimension], 
                                             value=values[i*self.key_dimension:(i+1)*self.key_dimension]) for i in range(self.num_head)], dim=0)
  
        output = self.linear_transform(output)
        return output
        
        
class Embedding(nn.Module):
    def __init__ (self, vocab_size, model_dimension=512) :
        super(Embedding, self).__init__()
        self.model_dimension = 512
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, 
                                            embedding_dim=model_dimension, 
                                            padding_idx=0)

    def __make_positional_vector (self, pos) :
        return [pos/np.power(10000, 2*(hidden_i//2)/ self.model_dimension) for hidden_i in range(self.model_dimension)]
    
    def __make_positional_encodings (self, sequence_length, batch_size = 128) :
        positional_encodings = np.array([[self.__make_positional_vector(i) for i in range(sequence_length)] for i in range(batch_size)])
        positional_encodings[:, 0::2] = np.sin(positional_encodings[:, 0::2])
        positional_encodings[:, 1::2] = np.cos(positional_encodings[:, 1::2])

        return positional_encodings

    def forward (self, x) :
        print(x.size())
        embedded_x = self.embedding_layer(x) ## todo : . In the embedding layers, we multiply those weights by √dmodel.
        position_x = self.__make_positional_encodings(sequence_length= x.size(dim=1), batch_size=x.size(dim=0))
        
        print(len(x))
        print(embedded_x.size())
        print(torch.tensor(position_x).size())
        return embedded_x + torch.tensor(position_x)

class Encoder(nn.Module):
    def __init__ (self, model_dimension = 512, num_head = 8, fc_dimension=2048) :
        super(Encoder, self).__init__()
        self.model_dimension = model_dimension
        self.num_head = num_head
        self.fc_dimension = fc_dimension
        self.multi_head_attention = Multi_Head_Attention(model_dimension=self.model_dimension)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.model_dimension, self.fc_dimension, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(self.fc_dimension, self.model_dimension, dtype=torch.float64)
        )
        self.layer_norm = nn.LayerNorm(self.model_dimension, dtype=torch.float64)
        
    
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
            nn.Linear(model_dimension, fc_dimension, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(fc_dimension, model_dimension, dtype=torch.float64)
        )
        self.layer_norm = nn.LayerNorm(model_dimension, dtype=torch.float64)
    
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
        super(TransFormerModel, self).__init__()
        self.model_dimension = model_dimension
        self.num_head = num_head
        self.num_encoder = num_encoder
        self.num_decoder = num_decoder
        self.vocab_size = vocab_size
        self.encoders = [Encoder(model_dimension = self.model_dimension) for i in range(num_encoder)]
        self.decoders = [Decoder(model_dimension = self.model_dimension) for i in range(num_encoder)]
        self.in_embedding = Embedding(model_dimension = self.model_dimension, vocab_size=vocab_size)
        self.out_embedding = Embedding(model_dimension = self.model_dimension, vocab_size=vocab_size)
        self.Linear = nn.Linear(model_dimension, model_dimension, dtype=torch.float64)
                
    
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
    
    input = torch.LongTensor([[i for i in range(512)] for j in range(128)])
    output = torch.LongTensor([[i for i in range(512)] for j in range(128)])
    model = TransFormerModel(model_dimension=512, num_head=8, num_encoder=2, num_decoder=2, vocab_size=1000).to(device)
    output = model(input, output)
    print(output)

    
if __name__ == "__main__":
    print("test")
    test()
