import random
import argparse
from model import *
from train import *
from dataset import *
from embedding import *
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--maxlen', type=int, default=128)
parser.add_argument('--use_embedding', type=bool, default=True)
args = parser.parse_args()
print(args)

seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_dataset = Dataset(args, return_alltext())
test_dataset = Dataset(args, return_alltext_test())
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = RNN(args)
train(model, args, train_dataloader, test_dataloader)
