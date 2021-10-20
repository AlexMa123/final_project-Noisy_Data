from numba import njit, jit, prange
import matplotlib.pyplot as plt
import numpy as np

@njit
def one_step(x, p):
    random_number = np.random.rand()
    if x == 0:
        if random_number > 0.5:
            return 1
        else:
            return -1
    if x > 0:
        if random_number > p:
            return -1
        else:
            return 1
    if x < 0:
        if random_number > p:
            return 1
        else:
            return -1


@jit
def STEPS(p, Q, delta=4, final_x=20, N=20, M=10):
    x = 0
    total_time = 0
    while x < final_x:
        total_time += delta
        num_success = 0
        trace = np.full((N), x)
        for i in range(N):
            for _ in range(delta):
                if trace[i] < final_x:
                    trace[i] = trace[i] + one_step(trace[i], p)
        num_success = trace[trace > x].size
 
        while num_success == 0:
            trace = np.concatenate((trace, np.full((M), x)))
            for i in range(1, M+1):
                for _ in range(delta):
                    if trace[-i] < final_x:
                        trace[-i] = trace[-i] + one_step(trace[-i], p)
            num_success = trace[trace > x].size
        P = num_success / trace.size
        if P > Q:
            select_trace = np.random.randint(trace.size)
            x = trace[select_trace]
        else:
            if np.random.rand() < Q:
                ind = np.where(trace > x)[0]
                select_trace = np.random.randint(ind.size)
                x = trace[ind[select_trace]]
            else:
                ind = np.where(trace <= x)[0]
                select_trace = np.random.randint(ind.size)
                x = trace[ind[select_trace]]
    return total_time


@jit
def STEPS_modify(p, Q, final_x=100, N=20, M=10):
    x = 0
    t = 0
    while x < final_x:
        trace = np.full((N), x)
        times = np.full((N), t)
        for i in range(N):
            absorption_wall = x-1
            while trace[i] > absorption_wall:
                absorption_wall = absorption_wall + 1
                trace[i] = trace[i] + one_step(trace[i], p)
                times[i] += 1
        num_success = trace[trace > x].size

        while num_success == 0:
            trace = np.concatenate((trace, np.full((M), x)))
            times = np.concatenate((times, np.full((M), t)))
            for i in range(1, M + 1):
                absorption_wall = x - 1
                while trace[-i] > absorption_wall:
                    absorption_wall += 1
                    trace[-i] = trace[-i] + one_step(trace[-i], p)
                    times[-i] += 1
            num_success = trace[trace > x].size
        P = num_success / trace.size
        if P > Q:
            select_trace = np.random.randint(trace.size)
            x = trace[select_trace]
            t = times[select_trace]
        else:
            if np.random.rand() < Q:
                ind = np.where(trace > x)[0]
                select_trace = np.random.randint(ind.size)
                x = trace[ind[select_trace]]
                t = times[ind[select_trace]]
            else:
                ind = np.where(trace <= x)[0]
                select_trace = np.random.randint(ind.size)
                x = trace[ind[select_trace]]
                t = trace[ind[select_trace]]
    return t



if __name__ == "__main__":
    print(STEPS_modify(0.2, 0.5, N=100, M=50))

    # fpt0 = np.zeros(500)
    # fpt1 = np.zeros(500)
    # fpt2 = np.zeros(500)
    # fpt3 = np.zeros(500)
    # fpt4 = np.zeros(500)
    # fpt5 = np.zeros(500)
    # for i in range(500):
    #     print(i)
    #     fpt0[i] = STEPS_modify(0.2, 0.5, N=100, M=50)
    #     fpt1[i] = STEPS_modify(0.2, 0.6, N=100, M=50)
    #     fpt2[i] = STEPS_modify(0.2, 0.7, N=100, M=50)
    #     fpt3[i] = STEPS_modify(0.2, 0.8, N=100, M=50)
    #     fpt4[i] = STEPS_modify(0.2, 0.9, N=100, M=50)
    #     fpt5[i] = STEPS_modify(0.2, 1, N=100, M=50)
    # np.save("fptm0_5.npy", fpt0)
    # np.save("fptm1_6.npy", fpt1)
    # np.save("fptm2_7.npy", fpt2)
    # np.save("fptm3_8.npy", fpt3)
    # np.save("fptm4_9.npy", fpt4)
    # np.save("fptm5_10.npy", fpt5)

    # fpt0 = np.zeros(500)
    # fpt1 = np.zeros(500)
    # fpt2 = np.zeros(500)
    # fpt3 = np.zeros(500)
    # fpt4 = np.zeros(500)
    # fpt5 = np.zeros(500)
    # for i in range(500):
    #     fpt0[i] = STEPS(0.2, 0.5, 5, N=100, M=50)
    #     fpt1[i] = STEPS(0.2, 0.6, 5, N=100, M=50)
    #     fpt2[i] = STEPS(0.2, 0.7, 5, N=100, M=50)
    #     fpt3[i] = STEPS(0.2, 0.8, 5, N=100, M=50)
    #     fpt4[i] = STEPS(0.2, 0.9, 5, N=100, M=50)
    #     fpt5[i] = STEPS(0.2, 1, 5, N=100, M=50)
    # np.save("fpt0_5.npy", fpt0)
    # np.save("fpt1_6.npy", fpt1)
    # np.save("fpt2_7.npy", fpt2)
    # np.save("fpt3_8.npy", fpt3)
    # np.save("fpt4_9.npy", fpt4)
    # np.save("fpt5_10.npy", fpt5)