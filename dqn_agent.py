import torch, torch.nn as nn, random, numpy as np
from collections import deque, namedtuple

Transition = namedtuple("Transition", "s a r s2 done")

class QNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,32,8,4), nn.ReLU(),    # stack 4 frames
            nn.Conv2d(32,64,4,2), nn.ReLU(),
            nn.Conv2d(64,64,3,1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024,512), nn.ReLU(),
            nn.Linear(512,n_actions))
    def forward(self,x): return self.net(x/255.)

class DQNAgent:
    def __init__(self, n_act, device):
        self.q, self.tgt = QNet(n_act).to(device), QNet(n_act).to(device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=1e-4)
        self.buf = deque(maxlen=500_000)
        self.device, self.n_act = device, n_act

    def act(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.n_act)
        with torch.no_grad():
            q = self.q(torch.as_tensor(state, device=self.device)[None])
        return int(q.argmax())

    def remember(self,*args): self.buf.append(Transition(*args))
    def sample(self,b): return random.sample(self.buf,b)

    def train_step(self,batch,γ=0.99):
        batch = Transition(*zip(*batch))
        s  = torch.as_tensor(np.stack(batch.s),device=self.device)
        a  = torch.as_tensor(batch.a,device=self.device).unsqueeze(1)
        r  = torch.as_tensor(batch.r,device=self.device).unsqueeze(1)
        s2 = torch.as_tensor(np.stack(batch.s2),device=self.device)
        d  = torch.as_tensor(batch.done,device=self.device).unsqueeze(1)

        with torch.no_grad():
            tgt_q = self.tgt(s2).max(1,keepdim=True)[0]
            y = r + γ * tgt_q * (1-d)

        q = self.q(s).gather(1,a)
        loss = nn.functional.smooth_l1_loss(q,y)

        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()
