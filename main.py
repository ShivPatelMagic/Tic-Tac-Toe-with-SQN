import os
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from TicTacToe import TicTacToe

class PlayerSQN:
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.99
        self.learning_rate = 0.00005 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "YourBITSid_MODEL.h5")
        
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path, custom_objects={'mse': 'mean_squared_error'})
                self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}. Creating a new model.")
                self.model = self.build_model()
        else:
            print("No existing model found. Creating a new model.")
            self.model = self.build_model()
        
        self.replay_buffer = deque(maxlen=10000)  
        self.q_values_log = []

    def build_model(self):
        """Builds the neural network model."""
        model = Sequential([
            Input(shape=(9,)), 
            Dense(128, activation='relu'), 
            Dropout(0.2), 
            Dense(128, activation='relu'),
            Dropout(0.2),  
            Dense(64, activation='relu'), 
            Dense(9, activation='linear') 
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def sample_experiences(self, batch_size):
        """Sample mini-batches from the replay buffer."""
        return random.sample(self.replay_buffer, min(len(self.replay_buffer), batch_size))

    def train(self, batch_size=64): 
        """Train the model using sampled experiences."""
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.sample_experiences(batch_size)
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        q_values = self.model.predict(states, verbose=0)
        q_next_values = self.model.predict(next_states, verbose=0)
        
        for i in range(len(batch)):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(q_next_values[i])
            q_values[i][actions[i]] = target
        
        self.model.fit(states, q_values, epochs=1, verbose=0)

    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        valid_actions = [i for i in range(9) if state[i] == 0]
        if not valid_actions:
            return None
        
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        self.q_values_log.append(q_values[0])  # Log Q-values
        return max(valid_actions, key=lambda x: q_values[0][x])
    
    def move(self, board_state):
        """Convert board state to a valid move using the model."""
        return self.select_action(np.array(board_state))

    def update_epsilon(self):
        """Decay epsilon."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        """Stores the experience in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def save_model(self, file_name):
        """Saves the trained model to a file."""
        self.model.save(file_name)
        print(f"Model saved to {file_name}")

def plot_graphs(rewards, epsilon_values, q_values_log):
    cumulative_rewards = [sum(rewards[:i+1]) for i in range(len(rewards))]
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cumulative_rewards)), cumulative_rewards, label="Cumulative Reward")
    plt.title("Cumulative Reward vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.savefig("cumulative_reward.png")
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(range(len(epsilon_values)), epsilon_values, label="Epsilon Decay", color='orange')
    plt.title("Epsilon Decay vs. Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon Value")
    plt.legend()
    plt.grid()
    plt.savefig("epsilon_decay.png")
    plt.show()

    if q_values_log:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(q_values_log[0])), q_values_log[0], color='blue', alpha=0.7)
        plt.title("Q-Value Distribution for Initial State")
        plt.xlabel("Actions (0-8)")
        plt.ylabel("Q-Value")
        plt.grid()
        plt.savefig("q_value_distribution.png")
        plt.show()

def main(smartMovePlayer1):
    """
    Simulates a TicTacToe game between Player 1 (random/smart player) and Player 2 (SQN-based player).
    """
    playerSQN = PlayerSQN()
    game = TicTacToe(smartMovePlayer1, playerSQN)

    num_episodes = 500
    rewards = []  
    epsilon_values = []  

    for episode in range(num_episodes):
        game.board = [0] * 9
        game.current_winner = None
        state = game.board.copy()
        done = False
        player_turn = 1
        total_reward = 0

        print(f"Episode {episode + 1}/{num_episodes}")

        while not done:
            if player_turn == 1:
                if game.empty_positions():
                    empty_positions = game.empty_positions()
                    position = random.choice(empty_positions)
                    game.make_move(position, 1)
                    print(f"Player 1 (Random/Smart) chooses position: {position + 1}")
                    player_turn = 2
                else:
                    done = True
            else:
                action = playerSQN.move(state)
                game.make_move(action, 2)
                print(f"Player 2 (SQN) chooses position: {action + 1}")
                player_turn = 1

                next_state = game.board.copy()
                reward = game.get_reward()
                total_reward += reward
                done = game.is_full() or game.current_winner is not None

                playerSQN.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    playerSQN.train()
                    break

        rewards.append(total_reward)
        epsilon_values.append(playerSQN.epsilon)
        playerSQN.update_epsilon()

        print(f"Reward for Player 2: {reward}\n")

    playerSQN.save_model('2022A7PS1157G_MODEL.h5')

    # Generate graphs
    #plot_graphs(rewards, epsilon_values, playerSQN.q_values_log)

if __name__ == "__main__":
    try:
        smartMovePlayer1 = float(sys.argv[1])
        assert 0 <= smartMovePlayer1 <= 1
    except:
        print("Usage: python YourBITSid.py <smartMovePlayer1>")
        print("Example: python YourBITSid.py 0.5")
        sys.exit(1)
    
    main(smartMovePlayer1)