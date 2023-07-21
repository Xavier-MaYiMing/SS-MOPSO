### SS_MOPSO: A self-organized speciation based multi-objective particle swarm optimizer

##### Reference: Qu B, Li C, Liang J, et al. A self-organized speciation based multi-objective particle swarm optimizer for multimodal multi-objective problems[J]. Applied Soft Computing, 2020, 86: 105886.

##### The SS_MOPSO belongs to the category of multi-objective evolutionary algorithms (MOEAs). SS_MOPSO is a powerful algorithm to solve the multi-modal multi-objective optimization (MMO) problems which uses speciation as a niching method.

| Variables | Meaning                                  |
| --------- | ---------------------------------------- |
| npop      | Population size                          |
| iter      | Iteration number                         |
| lb        | Lower bound                              |
| ub        | Upper bound                              |
| omega     | Inertia weight (default = 0.7298)        |
| c1        | Acceleration constant 1 (default = 2.05) |
| c2        | Acceleration constant 2 (default = 2.05) |
| rs        | Species radius (default = 0.05)          |
| n_POA     | Maximum POA size (default = 10)          |
| nvar      | The dimension of decision space          |
| pos       | The position of particles                |
| vmax      | Maximum velocity                         |
| vmin      | Minimum velocity                         |
| vel       | Velocity                                 |
| objs      | Objectives                               |
| nobj      | The dimension of objective space         |
| POA       | Personal optimal archive                 |
| POA_objs  | The objectives of POA                    |
| specs     | Speciations                              |
| nbest     | The neighbor best                        |
| pbest     | The personal best                        |
| ps        | Pareto set                               |
| pf        | Pareto front                             |

#### Test problem: MMF1



$$
\left\{
\begin{aligned}
&f_1(x)=|x_1-2|\\
&f_2(x)=1-\sqrt{|x_1 - 2|}+2(x_2-\sin{(6 \pi |x_1 - 2| + \pi)})^2\\
&1 \leq x_1 \leq 3, -1 \leq x_2 \leq 1
\end{aligned}
\right.
$$



#### Example

```python
if __name__ == '__main__':
    t_npop = 800
    t_iter = 100
    t_lb = np.array([1, -1])
    t_ub = np.array([3, 1])
    main(t_npop, t_iter, t_lb, t_ub)
```

##### Output:

![](https://github.com/Xavier-MaYiMing/SS-MOPSO/blob/main/Pareto%20front.png)

![](https://github.com/Xavier-MaYiMing/SS-MOPSO/blob/main/Pareto%20set.png)



```python
Iteration 10 completed.
Iteration 20 completed.
Iteration 30 completed.
Iteration 40 completed.
Iteration 50 completed.
Iteration 60 completed.
Iteration 70 completed.
Iteration 80 completed.
Iteration 90 completed.
Iteration 100 completed.
```

