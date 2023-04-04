import random
import numpy as np
from deap import algorithms, base, creator, tools

# 定义自定义函数，这里以 2 变量函数为例
def my_func(x, y):
    return (x)**2 +(y)**2

# 定义优化问题
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate(individual):
    # 计算自定义函数的值
    x, y = individual
    return (my_func(x, y),)

toolbox = base.Toolbox()

# 定义遗传算法的参数
toolbox.register("attr_float", random.uniform, -5, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行遗传算法
pop = toolbox.population(n=50)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=False)

# 输出结果
best_ind = tools.selBest(pop, k=1)[0]
print('最优解:', best_ind)
print('最优值:', best_ind.fitness.values[0])