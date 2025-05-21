# Shallow Q-Network for Tic-Tac-Toe

This project is part of the **Artificial Intelligence (CS F407)** coursework at BITS Pilani K.K. Birla Goa Campus. It explores reinforcement learning by implementing a **Shallow Q-Network (SQN)** to play the game of **Tic-Tac-Toe**. The goal is to understand the fundamentals of Q-learning and function approximation using neural networks.

## Project Overview

While Tic-Tac-Toe is a solved game, this project avoids hardcoding strategies and instead trains an agent using Q-learning with a shallow neural network. The project involves:

* Implementing an SQN using Keras and TensorFlow
* Training the agent through self-play using an **epsilon-greedy policy**
* Employing an **experience replay buffer** for stable learning
* Evaluating the trained model against different types of opponents
* Visualizing the agent's performance over time

## Key Features

* **Shallow Q-Network (SQN)** with two hidden layers using ReLU activation
* **Experience Replay Buffer** implemented with Python’s `deque`
* **Epsilon-Greedy Strategy** to balance exploration and exploitation
* **Model Training** using mini-batches sampled from the replay buffer
* **Performance Evaluation** against random and smart opponents

## Training and Evaluation

* The model is trained over thousands of games using experience replay.
* Gradual adjustment of the `epsilon` parameter ensures better policy convergence.
* Evaluation includes matchups against opponents with different intelligence levels (`smartMovePlayer1 = 0.0`, `0.5`, `1.0`).
* Graphs plot win/loss/draw trends over training episodes.

## Files Included

* `main.py`: Contains the implementation of the `PlayerSQN` agent
* `report.pdf`: Project report with methodology, results, and analysis

## Technologies Used

* Python 3.8+
* TensorFlow & Keras
* NumPy
* Matplotlib (for performance plots)

## How to Run

```bash
python main.py 0.5
```

## Report Highlights

* Design decisions and implementation details
* Training curves and result analysis
* Challenges faced and how they were overcome
* Suggestions for future improvements (e.g., deeper networks or advanced RL techniques)
