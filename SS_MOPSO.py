#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/17 11:50
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : SS_MOPSO.py
# @Statement : A self-organized speciation based multi-objective particle swarm optimizer (SS-MOPSO)
# @Reference : Qu B, Li C, Liang J, et al. A self-organized speciation based multi-objective particle swarm optimizer for multimodal multi-objective problems[J]. Applied Soft Computing, 2020, 86: 105886.
import numpy as np
import matplotlib.pyplot as plt


def cal_obj(x):
    # MMF1, 1 <= x[0] <= 3, -1 <= x[1] <= 1
    [x1, x2] = x
    f1 = np.abs(x1 - 2)
    f2 = 1 - np.sqrt(f1) + 2 * (x2 - np.sin(6 * np.pi * f1 + np.pi)) ** 2
    return np.array([f1, f2])


def nd_sort(objs):
    # fast non-dominated sort
    npop = len(objs)
    nobj = len(objs[0])
    n = np.zeros(npop, dtype=int)  # the number of particles that dominate this particle
    s = []  # the index of particles that this particle dominates
    rank = np.zeros(npop, dtype=int)  # Pareto rank
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def special_crowding_distance(pos, objs, pfs):
    # calculate the special crowding distance (SCD)
    (npop, dim) = pos.shape
    nobj = objs.shape[1]
    cd_x = np.zeros(npop)  # CD in decision space
    cd_f = np.zeros(npop)  # CD in objective space
    scd = np.zeros(npop)  # SCD
    for key in pfs.keys():
        pf = np.array(pfs[key])
        if len(pf) == 1:
            cd_x[pf[0]] = 1
            cd_f[pf[0]] = 1
            continue
        temp_pos = pos[pf]
        temp_obj = objs[pf]

        # calculate CD in decision space
        xmin = np.min(temp_pos, axis=0)
        xmax = np.max(temp_pos, axis=0)
        dx = xmax - xmin
        for i in range(dim):
            if dx[i] == 0:
                for j in range(len(pf)):
                    cd_x[pf[j]] += 1
            else:
                rank = np.argsort(temp_pos[:, i])
                cd_x[pf[rank[0]]] += 2 * (pos[pf[rank[1]], i] - pos[pf[rank[0]], i]) / dx[i]
                cd_x[pf[rank[-1]]] += 2 * (pos[pf[rank[-1]], i] - pos[pf[rank[-2]], i]) / dx[i]
                for j in range(1, len(pf) - 1):
                    cd_x[pf[rank[j]]] += (pos[pf[rank[j + 1]], i] - pos[pf[rank[j - 1]], i]) / dx[i]
        cd_x[pf] /= dim

        # calculate CD in objective space
        fmin = np.min(temp_obj, axis=0)
        fmax = np.max(temp_obj, axis=0)
        df = fmax - fmin
        for i in range(nobj):
            if df[i] == 0:
                for j in range(len(pf)):
                    cd_f[pf[j]] += 1
            else:
                rank = np.argsort(temp_obj[:, i])
                cd_f[pf[rank[0]]] += 1
                cd_f[pf[rank[-1]]] += 0
                for j in range(1, len(pf) - 1):
                    cd_f[pf[rank[j]]] += (objs[pf[rank[j + 1]], i] - objs[pf[rank[j - 1]], i]) / df[i]
        cd_f[pf] /= nobj

        # calculate SCD
        cd_x_avg = np.mean(cd_x[pf])
        cd_f_avg = np.mean(cd_f[pf])
        flag = np.logical_or(cd_x[pf] > cd_x_avg, cd_f[pf] > cd_f_avg)
        scd[pf] = np.where(flag, np.max((cd_x[pf], cd_f[pf]), axis=0), np.min((cd_x[pf], cd_f[pf]), axis=0))
    return scd


def nd_scd_sort(pos, objs):
    # sort the particles according to the Pareto rank and special crowding distance
    pos = np.array(pos)
    objs = np.array(objs)
    npop = pos.shape[0]
    pfs, rank = nd_sort(objs)
    scd = special_crowding_distance(pos, objs, pfs)
    temp_list = []
    for i in range(len(pos)):
        temp_list.append([pos[i], objs[i], rank[i], scd[i]])
    temp_list.sort(key=lambda x: (x[2], -x[3]))
    next_pos = np.zeros((npop, pos.shape[1]))
    next_objs = np.zeros((npop, objs.shape[1]))
    next_rank = np.zeros(npop)
    for i in range(npop):
        next_pos[i] = temp_list[i][0]
        next_objs[i] = temp_list[i][1]
        next_rank[i] = temp_list[i][2]
    return next_pos, next_objs, next_rank


def speciation(pos, rs):
    # self-organized speciation
    specs = []
    flag = np.full(pos.shape[0], True)
    while np.any(flag):
        remain = np.where(flag)[0]
        temp = pos[remain] - pos[remain[0]]
        dis = np.sqrt(np.sum(temp ** 2, axis=1))
        selected = np.where(dis <= rs)[0]
        flag[remain[selected]] = False
        specs.append(remain[selected])
    return specs


def main(npop, iter, lb, ub, omega=0.7298, c1=2.05, c2=2.05, rs=0.05, n_POA=10):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param omega: inertia weight (default = 0.7298)
    :param c1: acceleration constant 1 (default = 2.05)
    :param c2: acceleration constant 2 (default = 2.05)
    :param rs: species radius (default = 0.05)
    :param n_POA: maximum POA size (default = 10)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pos = np.random.uniform(lb, ub, (npop, nvar))  # positions
    vmax = 0.5 * (ub - lb)  # maximum velocity
    vmin = -vmax  # minimum velocity
    vel = np.random.uniform(vmin, vmax, (npop, nvar))  # velocity
    objs = np.array([cal_obj(x) for x in pos])  # objectives
    nobj = objs.shape[1]  # the dimension of objective space
    POA = [np.array([pos[i].copy()]) for i in range(npop)]  # personal optimal archive
    POA_objs = [np.array([objs[i].copy()]) for i in range(npop)]
    rs *= np.sqrt(np.sum((ub - lb) ** 2))  # speciation radius

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 10 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. Sort the particles
        pos, objs = nd_scd_sort(pos, objs)[: 2]

        # Step 2.2. Self-organized speciation
        specs = speciation(pos, rs)

        # Step 2.3. Update velocity and position
        for spec in specs:
            nbest = pos[spec[0]]
            for i in spec:
                pbest = POA[i][0]
                vel[i] = omega * vel[i] + c1 * np.random.random(nvar) * (pbest - pos[i]) + c2 * np.random.random(nvar) * (nbest - pos[i])
                vel[i] = np.min((vel[i], vmax), axis=0)
                vel[i] = np.max((vel[i], vmin), axis=0)
                pos[i] += vel[i]
                pos[i] = np.min((pos[i], ub), axis=0)
                pos[i] = np.max((pos[i], lb), axis=0)
                objs[i] = cal_obj(pos[i])

                # Step 2.4. Update POA
                POA[i] = np.concatenate((POA[i], pos[i].reshape(1, nvar)), axis=0)
                POA_objs[i] = np.concatenate((POA_objs[i], objs[i].reshape(1, nobj)), axis=0)
                POA[i], POA_objs[i] = nd_scd_sort(POA[i], POA_objs[i])[: 2]
                if len(POA[i]) > n_POA:
                    POA[i] = POA[i][: n_POA, :]
                    POA_objs[i] = POA_objs[i][: n_POA, :]

    # Step 3. Sort the results
    EXA = np.concatenate([POA[i] for i in range(npop)], axis=0)
    EXA_objs = np.concatenate([POA_objs[i] for i in range(npop)], axis=0)
    EXA, EXA_objs, rank = nd_scd_sort(EXA, EXA_objs)
    if EXA.shape[0] > npop:
        EXA = EXA[: npop, :]
        EXA_objs = EXA_objs[: npop, :]
        rank = rank[: npop]
    ind = np.where(rank == 1)[0]
    ps = EXA[ind]
    pf = EXA_objs[ind]
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    for p in ps:
        x1.append(p[0])
        x2.append(p[1])
    for o in pf:
        x3.append(o[0])
        x4.append(o[1])
    # Plot the Pareto set
    plt.figure()
    plt.scatter(x1, x2, marker='o', color='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Pareto set')
    plt.savefig('Pareto set')
    plt.show()
    # Plot the Pareto front
    plt.figure()
    plt.scatter(x3, x4, marker='o', color='red')
    plt.xlabel('objective 1')
    plt.ylabel('objective 2')
    plt.title('Pareto front')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    t_npop = 800
    t_iter = 100
    t_lb = np.array([1, -1])
    t_ub = np.array([3, 1])
    main(t_npop, t_iter, t_lb, t_ub)
