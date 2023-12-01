from GraphValueFunc import GCN
import torch, os
from torch.utils.tensorboard import SummaryWriter

args = dict({
    'lr': 0.001,
    'dropout': None,  # 0.3
    'cuda': False,  # torch.cuda.is_available(),
    'num_channels': 16,
})


class NeuralNet:

    def __init__(self, num_features):
        self.qfunction = GCN(hidden_channels=args['num_channels'], num_features=num_features)
        if args['cuda']:
            self.qfunction.cuda()
        self.optimizer = torch.optim.Adam(self.qfunction.parameters(), lr=args['lr'])
        self.new_update = False
        self.updates_no = 0
        self.losses = []
        self.writer = SummaryWriter()

    def update(self, loss):
        self.losses.append(loss)
        self.updates_no += 1
        self.new_update = True
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, nx_graph):
        self.qfunction.eval()
        with torch.no_grad():
            return self.qfunction(nx_graph)

    def select_action(self, nx_graph):
        return self.predict(nx_graph).argmax(dim=0)

    def get_max_q(self, nx_graph):
        return self.predict(nx_graph).max(dim=0)[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        self.new_update = False
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists!")
        torch.save({
            'state_dict': self.qfunction.state_dict(),
        }, filepath)

    def write_avg_loss(self, n_iter):
        if len(self.losses) == 0:
            return
        avg_loss = sum(self.losses) / len(self.losses)
        self.losses = []
        self.writer.add_scalar('Loss/train', avg_loss, n_iter)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args['cuda'] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.qfunction.load_state_dict(checkpoint['state_dict'])
