# Learnings: Snake AI

Notes on what I learned while building a reinforcement-learning agent that plays Snake with PyTorch and Pygame. The sections marked **My addition** are changes I made on top of it.


---

## 1. Environment setup

- Created an isolated Conda environment (`snake_ai`) so this project's packages don't clash with anything else on my machine.
- Worked on macOS with VS Code.
- **Decision: Python 3.10 instead of the tutorial's older version.** I chose 3.10 because PyTorch did not work on the older Python versions.
- **What I learned:** a tutorial's tool versions can be outdated. Check what the key library (here PyTorch) actually supports before picking the Python version, and record the choice so the project is reproducible.

## 2. Building the game with Pygame

- Built a human-playable Snake game before touching any AI. The agent needs a working environment to learn from, so the game had to be solid first.
- **What I learned:**
  - How a Pygame game is structured: a game loop, handling input events, drawing to the screen, and controlling the frame rate.
  - How to represent the snake, food, collisions and score in code.
  - How to separate the game logic from the player's input, so an AI agent can later control the snake instead of a human.

## 3. Setting up the agent

- The agent learns by playing, getting a reward signal, and adjusting its behaviour.
- The core pieces I had to set up:
  - **State:** 11 Boolean features: danger straight, danger to the right, danger to the left; the snake’s current direction (left, right, up, or down); and whether the food is left, right, above, or below the snake.
  - **Actions:** Three one-hot choices: continue straight, turn right, or turn left.
  - **Reward:** +10 for eating food, −10 for a collision or timeout, and 0 for an ordinary move.
- **What I learned:** the agent only learns what the reward tells it to. The way the state and reward are defined shapes the behaviour as much as the model does.

## 4. Training the model

- Trained a neural network (PyTorch) to estimate how good each action is in a given state (Deep Q-Learning).
- The loop, in plain words: play a move, observe the result and reward, store the experience, then train the network on stored experiences so its estimates improve.
- **What I learned:**
  - How to set up a model, a loss function and an optimizer in PyTorch.
  - How training and playing differ: during play the agent chooses moves, during training it also updates the network.
  - How to track progress with a score plot, and why the early games look random.
- Results: After 100 games, Best Score : 51, Average Score : 5.15.

## 5. My addition: 

Target network and Double DQN

- **The problem:** in plain Q-learning, the network that picks the next move is also the one that scores it, so it tends to overestimate values and the training targets keep shifting.
- **What I changed:** I added a second, slowly updated **target network**. One network chooses the next move, and the slowly updated network estimates its value. This makes the training targets steadier and reduces overestimation.
- **What I learned:** stability matters in reinforcement learning. A small structural change (two networks instead of one) can change how smoothly training behaves.
- **What I observed:** In the run shown in my graph, scores varied a lot: many games scored 0, while a few reached 1. The average score rose at first, then fluctuated and ended around 0.30 after roughly 45 games. This showed me that early training is noisy and that I need to compare several runs before deciding whether Double DQN improved learning.

Disabling gradient tracking when choosing a move

- When the agent is only choosing a move, it doesn't need to compute gradients, because nothing is being trained at that moment.
- **What I changed:** I disabled gradient tracking in that step, which avoids unnecessary computation.
- **What I learned:** PyTorch tracks operations by default so it can compute gradients later. Turning that off where it's not needed saves work, which adds up over thousands of moves.

---

## Mistakes and dead ends

- One challenge was interpreting the score graph: a single high score did not mean the agent had learned a reliable strategy. I learned to watch the running average across many games as well as the best score. I also backed up the existing checkpoint before changing the trainer, so I could keep the earlier model if needed.

## If I rebuilt this

- I would decide on a fixed training budget and evaluation method at the start, save the results from each run, and compare changes using multiple runs. I would also add a clear option to load a saved checkpoint so I could continue training instead of always starting from scratch.
