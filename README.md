# Agents library explanation
The aim of this library is to provide a set of utilities to approach reinforcement learning with:
- Clearly defined responsabilities split between classes
- Clean and exchangable APIs

## Center Classes
### Agents
In charge of implementing a specific RL algorithm such as:
- DQN (with replay from memory)
- DDQN
- Distributional DQN
- Prioritized experience replay DQN
- 'Rainbow' Agent

API: train(), act(), process_observation(), Q(), policy(), V(), + save/load()

### Models (TODO)
The neural network definition should be independent of the algorithm
The algorithm might just be the same, but an environment might require image pre-processing (convolutional layers)
API: Keras API (fit, predict, etc)

### Epsilon polciies
TODO

## Utilities
### State serializers
These should feed an agent with a ready to use state in the expected shape
API: serialize(), deserialize()

### Runners
These should handle the base program flow (epochs, episodes) and calls to rest of the modules
API: train(), demonstrate(), render()
