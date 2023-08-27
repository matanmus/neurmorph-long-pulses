"""
Script applies input to stimulate the reservoir and saves the solution of the fluid fields as an npz data file

1d compressible fluid calculation (Adiabatic) with van-der-Waals equation of state using local stimuli
Equations follow Slemrod 1984
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.matlib
import os
import math

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

import logging
logger = logging.getLogger(__name__)

N = 4096
dt = 0.3e-3

# parameters
w0=2
T0=0.93
cv=600

iter=5 #number of iterations between plots
dom=5 #range of space

valueList = np.arange(1.3, 2.1, 0.01)
print(valueList)
print(len(valueList))
for value in valueList:
    A1 = math.floor(value)
    A2 = math.floor((value - A1)*10)
    A3 = math.floor(((value - A1)*10 - A2)*10)
    A4 = math.floor((((value - A1)*10 - A2)*10 - A3)*10)
    print(A1, A2, A3, A4)
    xInput = [2.5, 1, -1, -2.5, 0, 0] #stimulation position
    A = [10 * A1, 10 * A2, 10 * A3, 10 * A4, 0, 0]  # stimulation amplitude

    # Bases and domain
    x_basis = de.Fourier('x', N, interval=(-dom, dom), dealias=3/2)
    domain = de.Domain([x_basis], np.float64)

    # Problem
    problem = de.IVP(domain, variables=['u', 'w', 'ddw', 'E', 'T', 'p'])

    problem.parameters['mu'] = 1
    problem.parameters['A'] = 1
    problem.parameters['delta'] = 100
    problem.parameters['Cv'] = cv
    #External stimulation parameters
    problem.parameters['A1'] = A[0]
    problem.parameters['A2'] = A[1]
    problem.parameters['A3'] = A[2]
    problem.parameters['A4'] = A[3]
    problem.parameters['A5'] = A[4]
    problem.parameters['A6'] = A[5]
    problem.parameters['t0'] = 0.1
    problem.parameters['x1'] = xInput[0]
    problem.parameters['x2'] = xInput[1]
    problem.parameters['x3'] = xInput[2]
    problem.parameters['x4'] = xInput[3]
    problem.parameters['x5'] = xInput[4]
    problem.parameters['x6'] = xInput[5]

    problem.substitutions['Iext1'] = "-A1*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x1))*exp(-128*(x-x1)**2)"
    problem.substitutions['Iext2'] = "-A2*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x2))*exp(-128*(x-x2)**2)"
    problem.substitutions['Iext3'] = "-A3*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x3))*exp(-128*(x-x3)**2)"
    problem.substitutions['Iext4'] = "-A4*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x4))*exp(-128*(x-x4)**2)"
    problem.substitutions['Iext5'] = "-A5*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x5))*exp(-128*(x-x5)**2)"
    problem.substitutions['Iext6'] = "-A6*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x6))*exp(-128*(x-x6)**2)"

    problem.add_equation("dt(w) - dx(u) = 0")
    problem.add_equation("dt(u) + dx(p) - mu*dx(dx(u)) + mu**2*A*dx(ddw) = Iext1 + Iext2 + Iext3 + Iext4 + Iext5 + Iext6")
    problem.add_equation("dt(E)  - delta*dx(dx(T)) = -dx(u)*p - u*dx(p) + mu*u*dx(dx(u)) + mu*dx(u)**2 - A*mu**2*dx(u)*ddw - A*mu**2*u*dx(ddw) + A*mu**2*dx(dx(u))*dx(w) + A*mu**2*dx(u)*ddw")
    #additional equations
    problem.add_equation("ddw - dx(dx(w)) = 0")
    problem.add_equation("Cv*T  - E =  3/w - u**2/2 - A*mu**2*dx(w)**2/2")
    problem.add_equation("p  = 8*T/(3*w-1) - 3/w**2")# + Iext*(1 - (t - t0)/abs(t-t0))*exp(-128*(x-x0)**2)")

    # Build solver
    solver = problem.build_solver(de.timesteppers.SBDF2)
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    solver.stop_sim_time = 0.3

    # Initial conditions
    x = domain.grid(0)
    u = solver.state['u']
    w = solver.state['w']
    E = solver.state['E']
    T = solver.state['T']
    p = solver.state['p']

    w['g'] = w0
    u['g'] = 0.
    E['g'] = T0*cv-3/w0
    T['g'] = T0
    p['g'] = 8*T0/(3*w0-1) - 3/w0**2

    u_list = [np.copy(u['g'])]
    w_list = [np.copy(w['g'])]
    E_list = [np.copy(E['g'])]
    T_list = [np.copy(T['g'])]
    p_list = [np.copy(p['g'])]
    t_list = [solver.sim_time]

    # Main loop
    while solver.ok:
        solver.step(dt)

        if solver.iteration % iter == 0:
            u.set_scales(1, keep_data=True)
            w.set_scales(1, keep_data=True)
            E.set_scales(1, keep_data=True)
            T.set_scales(1, keep_data=True)
            p.set_scales(1, keep_data=True)
            u_list.append(np.copy(u['g']))
            w_list.append(np.copy(w['g']))
            E_list.append(np.copy(E['g']))
            T_list.append(np.copy(T['g']))
            p_list.append(np.copy(p['g']))
            t_list.append(solver.sim_time)

        if solver.iteration % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

    u_array = np.array(u_list)
    w_array = np.array(w_list)
    E_array = np.array(E_list)
    T_array = np.array(T_list)
    p_array = np.array(p_list)
    t_array = np.array(t_list)
    rho_array=1/w_array
    t_plots = np.array(np.arange(0., solver.stop_sim_time, iter*dt))

    # Save data
    filename = '%.3f' % value
    filename += '_data.npz'
    np.savez(filename, value = value, A = A, xInput = xInput, w0 = w0, dom = dom, N = N, dt=dt, iter=iter, t_array=t_array, t_plots=t_plots, rho_array=rho_array, u_array=u_array, E_array=E_array, p_array=p_array, T_array=T_array)
