# DRLND_p3_test
 Test of DRLND Project 3 with Prioritized Experience Replay
----
See https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb for the inspiration / pattern for the Prioritized Experience Replay.

See class `NaivePrioritizedBuffer` and references to it in `ddpg_agent.py` for my implementation.

My major issue is that I don't fully understand how it shuold be implemented, especially with regard to priorities in the function `update_priorities`. The data I have to play with for this is the initial part of an MSE calculation which leaves me with `batch_indices` (i.e. my batch size) of shape `(1024, )` which combined with `batch_priorities` (from the MSE) of shape `(1024,1024)` which when combined in the processing loop means I am trying to push an array of 1024 into each position in the target `self.priorities` list, which is expecting only one value in each position. So, I get `"ValueError: setting an array element with a sequence"`. 

I've tried `np.mean` and `np.max` of these values but neither produce a viable outcome. If I know exactly what this is all meant to do then I might be better equipped to make the correct choices - maybe this needs to go in as a 1024 array?

Please feel free to play with this and get it going - if you don't have the time / resources that's OK too - I've given it a shot!   

This is my debugging of these values - you can find this in the notebook....
    
```
Episode 10 (3 sec)	Current Score: 0.05	Mean Score: -0.00	Moving Average Score: 0.000
Episode 20 (1 sec)	Current Score: -0.00	Mean Score: 0.05	Moving Average Score: 0.000
Episode 24 (0 sec)	Current Score: -0.00	Mean Score: -0.00	Moving Average Score: -0.00> g:\deeplearning\udacity\deep-reinforcement-learning\p3_collab-compet\ddpg_agent.py(171)update_priorities()
-> for idx, prio in zip(batch_indices, batch_priorities):
(Pdb) type(batch_indices)
<class 'numpy.ndarray'>
(Pdb) type(batch_priorities)
<class 'numpy.ndarray'>
(Pdb) type(self.priorities)
<class 'numpy.ndarray'>
(Pdb) self.priorities.shape
(200000,)
(Pdb) batch_indices.shape
(1024,)
(Pdb) batch_priorities.shape
(1024, 1024)
(Pdb) n
> g:\deeplearning\udacity\deep-reinforcement-learning\p3_collab-compet\ddpg_agent.py(172)update_priorities()
-> self.priorities[idx] = prio
(Pdb) self.priorities[idx]
1.0
(Pdb) prio
array([5.8160003e-05, 5.8160003e-05, 5.8160003e-05, ..., 5.8160003e-05,
       5.8160003e-05, 5.8160003e-05], dtype=float32)
(Pdb) idx
608
(Pdb) batch_indices[0]
608
(Pdb) np.mean(prio)
8.4208776e-05
(Pdb) n
ValueError: setting an array element with a sequence.
> g:\deeplearning\udacity\deep-reinforcement-learning\p3_collab-compet\ddpg_agent.py(172)update_priorities()
-> self.priorities[idx] = prio
```  