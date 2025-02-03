# Deep Convolutional Q-Learning for Pac-Man

This project implements a Deep Convolutional Q-Learning algorithm to train an AI agent to play Pac-Man. The agent learns to navigate the maze, collect pellets, avoid ghosts, and maximise its score through deep reinforcement learning.

## Project Overview

The project uses a Deep Q-Network (DQN) with convolutional layers to process the game's visual input and learn optimal actions. The implementation leverages PyTorch for the neural network and OpenAI's Gymnasium for the Pac-Man environment.

### Technical Components

1. **Neural Network Architecture**
   - Convolutional layers for processing visual input
   - Batch normalisation layers for training stability
   - ReLU activation functions
   - Fully connected layers for Q-value prediction
   - Input: Game screen (128x128x3)
   - Output: Q-values for 9 possible actions

2. **Q-Learning Implementation**
   - Experience replay buffer (10,000 samples)
   - ε-greedy exploration strategy
   - Target network for stable learning
   - Discount factor (γ) of 0.99
   - Learning rate of 5e-4

3. **Training Process**
   - Maximum 2,000 episodes
   - Up to 10,000 timesteps per episode
   - Epsilon decay from 1.0 to 0.01 (exploration rate)
   - Target score: 500 points averaged over 100 episodes

## Requirements

- Python 3.x
- PyTorch
- Gymnasium with Atari environments
- NumPy
- Other dependencies listed in the notebook

## Project Structure

- `Deep_Convolutional_Q_Learning_for_Pac_Man.ipynb`: Main implementation notebook
- Training visualisation and results
- Saved model checkpoints (when training completes)

## How It Works

1. **State Processing**
   - Game frames are preprocessed to 128x128 RGB images
   - Frames are normalised and converted to PyTorch tensors

2. **Action Selection**
   - Agent chooses actions using ε-greedy policy
   - Explores random actions early in training
   - Gradually shifts to exploiting learned strategies

3. **Learning Process**
   - Stores experiences in replay buffer
   - Randomly samples batches for learning
   - Updates Q-values using Bellman equation
   - Periodically updates target network

4. **Performance Monitoring**
   - Tracks average score over 100 episodes
   - Saves model when performance target is reached
   - Visualises trained agent's gameplay

## Results

The agent is trained until it achieves an average score of 500 points over 100 consecutive episodes, or until it reaches the maximum number of training episodes. The final trained model can be visualised playing the game using the included video generation code.

## Usage

1. Open the notebook in a GPU-enabled environment (Google Colab recommended)
2. Install required dependencies
3. Run cells sequentially to train the agent
4. View results and generated gameplay videos

## Future Improvements

- Implement prioritised experience replay
- Add double DQN architecture
- Experiment with different network architectures
- Add frame stacking for temporal information
- Implement rainbow DQN improvements

## License

This project is open-source and available for educational and research purposes.
