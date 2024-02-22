import sys, logging
import numpy as np
from dmosopt import dmosopt

def zdt3(x):
    num_variables = len(x)
    f = np.zeros(2)
    g = 1+ 9./float(num_variables-1)*np.sum(x[1:]) # g(x)
    f[0] = x[0] # F_1(x)
    h = 1. - np.sqrt(f[0]/g)-f[0]/g*np.sin(10*np.pi*f[0])
    f[1] = g*h
    return f

def obj_fun(pp):
    param_values = np.asarray([pp[k] for k in sorted(pp)])
    res = zdt3(param_values)
    return res

def zdt3_pareto(n_points=200):
    #n_points is not the dimension of zdt problem, but is the number of points generate in the Pareto front
    f = np.zeros([n_points,2])
    f[:,0] = np.linspace(0,1,n_points)
    f[:,1] = 1.0 - (f[:,0])**2
    #To simplify calculation of Pareto front and also make it more visible, let x2,...x30=0,so g = 1
    return f

if __name__ == '__main__':
    space = {}
    for i in range(30):
        space['x%d' % (i+1)] =[0.0,1.0]#every list in dic space are [0,1]
    problem_para = {}
    objective_names = ['y1', 'y2']
    
    #create optimizer
    dmosopt_params = {'opt_id': 'dmosopt_zdt3',
                      'obj_fun_name': 'zdt3_GJM.obj_fun',#file_name.obj_fun
                      'problem_parameters': problem_para,
                      'space': space,
                      'objective_names': objective_names,
                      'population_size': 200,
                      'num_generations': 200,
                      'initial_maxiter': 10,
                      'optimizer': 'nsga2',
                      'termination_conditions': True,
                      'n_initial': 3,
                      'n_epochs': 2}
    best = dmosopt.run(dmosopt_params, verbose=True)
    if best is not None:
        import matplotlib.pyplot as plt
        bestx, besty = best
        x, y = dmosopt.dopt_dict['dmosopt_zdt3'].optimizer_dict[0].get_evals()
        besty_dict = dict(besty)
        
        # plot results
        plt.plot(y[:,0],y[:,1],'b.',label='evaluated points')
        plt.plot(besty_dict['y1'],besty_dict['y2'],'r.',label='best points')
    
        y_true = zdt3_pareto()#原来前面写的pareto front的计算只是用来画图啊
        plt.plot(y_true[:,0],y_true[:,1],'k-',label='True Pareto')
        plt.legend()#添加图例
        
        plt.savefig("example_dmosopt_zdt3.svg")