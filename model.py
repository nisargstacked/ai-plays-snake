import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, file_name: str = "model.pth") -> None:
        os.makedirs("model", exist_ok=True)
        torch.save(self.state_dict(), os.path.join("model", file_name))


class QTrainer:
    def __init__(
        self,
        model: Linear_QNet,
        lr: float,
        gamma: float,
        target_update_interval: int = 1000,
    ):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        self.target_update_interval = target_update_interval
        self.training_steps = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done) -> None:
        state = torch.as_tensor(state, dtype=torch.float)
        next_state = torch.as_tensor(next_state, dtype=torch.float)
        action = torch.as_tensor(action, dtype=torch.long)
        reward = torch.as_tensor(reward, dtype=torch.float)
        done = torch.as_tensor(done, dtype=torch.bool)

        # Add a batch dimension when training from one move.
        if state.ndim == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Double DQN: online network chooses the next action, while the
        # target network evaluates it. No gradient flows through this target.
        with torch.no_grad():
            next_actions = self.model(next_state).argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(next_state).gather(
                1, next_actions
            ).squeeze(1)
            q_targets = reward + self.gamma * next_q_values * (~done)

        predictions = self.model(state)
        targets = predictions.detach().clone()
        action_indices = action.argmax(dim=1)
        batch_indices = torch.arange(state.shape[0])
        targets[batch_indices, action_indices] = q_targets

        self.optimizer.zero_grad()
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
