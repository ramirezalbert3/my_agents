# Agents library explanation
This library attempts to provide a set of utilities to approach reinforcement learning with:
- Clear and defined responsabilities split between resources
- Clean and flexible APIs

## Core resources
### Agents (TODO)
In charge of implementing a specific RL algorithm such as:
- DQN (with replay from memory)
- DDQN
- Distributional DQN
- Prioritized experience replay DQN
- 'Rainbow' Agent

```
# API
train(), act(), process_observation(), Q(), policy(), V(),
save/load() # maybe this belongs to the network model?
```
### Models (TODO)
The neural network definition should be independent of the algorithm
The algorithm might just be the same, but an environment might require for example image pre-processing (convolutional layers)
```
# API
Keras-based API (fit, predict, etc)
```
### Epsilon policies
TODO

## Utility resources
### State serializers
These should feed an agent with a ready to use state in the expected shape
```
# API
serialize(), deserialize()
```
### Runners
These should handle the base program flow (epochs, episodes) and calls to rest of the modules
```
# API
train(), demonstrate(), render(), run_episode(), run_epoch()
```
### Plotting and history utils (TODO)
