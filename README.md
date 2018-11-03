# Agents library explanation
This library attempts to provide a set of utilities to approach reinforcement learning with:
- Clear and defined responsabilities split between resources
- Clean and flexible APIs

## Library Diagram
```
+--------+
|        |
| Runner | train()
|        | demonstrate()
+--------+


+-----+            +-----------------+              +-------+               +-------+
|     |   state    |                 | serialize()  |       |               |       |
| Env +------------> StateSerializer +--------------> Agent +---------------> Model |
|     |            |                 |              |       |               |       |
+--+--+            +-----------------+              +--+----+               +---^---+
   |                                                   |                        |
   |                                                   | act()                  |
   |                      action                       |                        |
   <---------------------------------------------------+                        |
   |                                                   |                        |
   | step()                                            |                        |
   |                reward, next_state                 |                        |
   +--------------------------------------------------->                        |
                                                       |                        |
                                                       | process_observation()  |
                                                       |                        |
                                                       |        train()         |
                                                       +------------------------+
```

## Core resources
### Agents (TODO)
In charge of implementing a specific RL algorithm such as:
- DQN (with replay from memory) [Minh 2015: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf]
- DDQN [Van Hesselt 2015: https://arxiv.org/pdf/1509.06461.pdf]
- Distributional DQN [Bellemare 2017: https://arxiv.org/pdf/1707.06887.pdf]
- Prioritized experience replay DQN [T. Schaul 2015: https://arxiv.org/pdf/1511.05952.pdf]
- Multi-step learning [De Asis 2017 https://arxiv.org/pdf/1703.01327.pdf]
- Rainbow Agent [Hessel 2017: https://arxiv.org/pdf/1710.02298.pdf]

```
API:
train(), act(), process_observation(), Q(), policy(), V(),
save/load() # maybe this belongs to the network model?
```
### Models (TODO)
The neural network definition should be independent of the algorithm
The algorithm might just be the same, but an environment might require for example image pre-processing (convolutional layers)
Initial design could be just to pass a keras-model to the agents constructor
```
API:
Keras-based API (fit, predict, etc)
```
### Epsilon policies (TODO)

## Utility resources
### State serializers
These should feed an agent with a ready to use state in the expected shape
```
API:
serialize(), deserialize()
```
### Runners
These should handle the base program flow (epochs, episodes) and calls to rest of the modules
```
API:
train(), demonstrate(), render(), run_episode(), run_epoch()
```
### Plotting and history utils (TODO)
