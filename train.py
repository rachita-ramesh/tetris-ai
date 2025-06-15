import wandb, torch, numpy as np
from env_wrapper import make_env
from dqn_agent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = make_env()
agent = DQNAgent(n_act=env.action_space.n, device=device)
wandb.init(project="tetris-dqn")

eps, eps_min, eps_decay = 1.0, 0.05, 1e-6
steps, sync_every, batch = 0, 10_000, 32
state, _ = env.reset()
frames = np.repeat(state,4,axis=2)   # stack 4

while True:
    action = agent.act(frames.transpose(2,0,1), eps)
    next_state, reward, done, trunc, _ = env.step(action)
    next_frames = np.concatenate([frames[:,:,1:], next_state], axis=2)
    agent.remember(frames.transpose(2,0,1), action, reward, next_frames.transpose(2,0,1), done)
    frames = next_frames
    steps += 1; eps = max(eps_min, eps - eps_decay)

    if len(agent.buf) > 50_000 and steps % 4 == 0:
        loss = agent.train_step(agent.sample(batch))
        wandb.log({"loss": loss, "eps": eps}, step=steps)

    if steps % sync_every == 0:
        agent.tgt.load_state_dict(agent.q.state_dict())

    if done or trunc:
        state, _ = env.reset()
        frames = np.repeat(state,4,axis=2)
