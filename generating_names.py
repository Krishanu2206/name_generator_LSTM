import torch
import torch.nn as nn
import string
import random
import sys
import torch.optim.optimizer
from unidecode import unidecode
from torch.utils.tensorboard import SummaryWriter

##DEVICE CONFIG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##GET CHARACTERS FROM STRING-PRINTABLE
all_characters = string.printable
n_characters = len(all_characters)
print(all_characters)

##READ LARGE TEXT FILE 
file = unidecode(open('names.txt').read())

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        # self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x) ##return will be a 2D tensor but lstm requires 3d tensor[N, L, Hin] so unsqueeze N=batch_size and L=seq_length and Hin=input_size
        out, (hidden, cell) = self.lstm(out.unsqueeze(dim=1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell

class Generator(): ##to generate some txt randomly
    def __init__(self):
        self.chunk_len=250
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_every=50
        self.hidden_size = 256
        self.num_layers = 2
        self.lr=0.003

    def char_tensor(self, string): ##takes a character and maps that to a vector of size 100
        tensor = torch.zeros(len(string)).long().to(device)
        for i, c in enumerate(string):
            tensor[i] = all_characters.index(c)
        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(file) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_str = file[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()

    def generate(self, initial_str='Ab', predict_len = 100, temperature = 0.85):
        hidden, cell = self.rnn.init_hidden(batch_size = self.batch_size)
        initial_inp = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str)-1):
            _, (hidden, cell) = self.rnn(initial_inp[p].view(1).to(device), hidden, cell)

        last_char = initial_inp[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char =all_characters[top_char]
            predicted+= predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted

    #input_size, hidden_size, num_layers, output_size
    def train(self):
        generated_names = []
        self.rnn = RNN(n_characters, self.hidden_size, self.num_layers, n_characters).to(device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(f'runs/name0') #for tensorboard

        print("=>STARTING TRAINING")

        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len): #so for 250 iterations its going to predict 250 characters
                output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell) ##passing a character as the input with the hidden and cell
                loss += criterion(output, target[:, c]) ##output -> predicting the next_character

            loss.backward()
            optimizer.step()
            loss = loss.item()/self.chunk_len

            if epoch % self.print_every == 0:
                print(f"Epoch: {epoch}/{self.num_epochs}, Loss: {loss:.4f}")
                generated_names.append(self.generate())
                print(self.generate())
                with open('generated_names.txt', 'w') as f:
                    for name in generated_names:
                        f.write(name + '\n')
                
            writer.add_scalar('Train Loss', loss, global_step=epoch)  # log the scalar value


gennames = Generator()
gennames.train()
            
