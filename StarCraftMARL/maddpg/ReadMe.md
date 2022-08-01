# Expertmental record
Author: Yubo Huang

This file is used to record the experimental result of MADDPG.

## Baseline simulation

This simulation uses the default parameters of MADDPG

Result directory: ./learning_curves - ./learning_curves4

run.log - run4.log

## Test the influence of the episode length

Maximal episode length = 50

Result directory: ./learning_curves5

run5.log

### Resluts

| Game | len=50 |
|  ----  | ----  |
|Game 1| -12
|Game 2| 14
|Game 3| -32
|Game 4| -10
|Game 5| -109
|Game 6| -108
|Game 7| -653
|Game 8| 44
|Game 9| 86

## Baseline result after Modifing the distribution

The softcatgorial distribution is wrong in the initial distribution file. So I modifited this file and then compute the new result

result dirsctory: ./learning_curves6-10
log file: run6.log - run10.log

### Results
| Game | len=50 |
|  ----  | ----  |
|Game 1| -38
|Game 2| -23
|Game 3| -37
|Game 4| -28
|Game 5| -163
|Game 6| -165
|Game 7| -693
|Game 8| -4
|Game 9| -39

### Conclusion

I do not know why the result is extremely bad after using the write distribution function. Maybe it caused by unsuitable hyperparameters. Maybe MADDPG has bad performance. I will tune the parameter next. 

## Add reward mask

In this experiment, I want to observe whether the result will change after adding the reward mash, that is the reward of an agent is set to 0 after it dies. Note, I use the initial wrong distribution function in this experiment. 