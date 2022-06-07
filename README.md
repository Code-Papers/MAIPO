# Multi-agent Inductive Policy Optimization

This repo is the code implementation of the paper titled "Multi-agent Inductive Policy Optimization" (MAIPO). In this paper, a novel multi-agent reinforcement learning algorithms is proposed and this repo contains all details of MAIPO.

## Dependency

1. Python3 (include numpy, tensorflow1.0 etc.)
2. [Pettingzoo]([GitHub - Farama-Foundation/PettingZoo: Gym for multi-agent reinforcement learning](https://github.com/Farama-Foundation/PettingZoo))
3. [FAST.Farm]([GitHub - OpenFAST/openfast: Main repository for the NREL-supported OpenFAST whole-turbine and FAST.Farm wind farm simulation codes.](https://github.com/OpenFAST/openfast))

## How to run the code

```
python3 ./run.sh > run.log 2>&1 &
```

Please use the following command to see other input parameters of the train.py file.

```
python3 train.py --help
```


