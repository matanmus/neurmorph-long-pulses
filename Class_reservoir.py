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

def Class_reservoir(A, n):
    logger = logging.getLogger(__name__)

    N = 4096
    dt = 0.3e-3

    # parameters
    w0=10#2
    T0=0.93
    cv=600

    iter=5 #number of iterations between plots
    dom=8 #5 #range of space

    amp = 100

    #xInput = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5] #stimulation position
    xInput = [-7, -5, -3, -1, 1, 3, 5, 7]  # stimulation position

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
    problem.parameters['A1'] = A[0]*amp
    problem.parameters['A2'] = A[1]*amp
    problem.parameters['A3'] = A[2]*amp
    problem.parameters['A4'] = A[3]*amp
    problem.parameters['A5'] = A[4]*amp
    problem.parameters['A6'] = A[5]*amp
    problem.parameters['A7'] = A[6]*amp
    problem.parameters['A8'] = A[7]*amp
    problem.parameters['t0'] = 0.1
    problem.parameters['x1'] = xInput[0]
    problem.parameters['x2'] = xInput[1]
    problem.parameters['x3'] = xInput[2]
    problem.parameters['x4'] = xInput[3]
    problem.parameters['x5'] = xInput[4]
    problem.parameters['x6'] = xInput[5]
    problem.parameters['x7'] = xInput[6]
    problem.parameters['x8'] = xInput[7]

    problem.substitutions['Iext1'] = "-A1*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x1))*exp(-128*(x-x1)**2)"
    problem.substitutions['Iext2'] = "-A2*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x2))*exp(-128*(x-x2)**2)"
    problem.substitutions['Iext3'] = "-A3*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x3))*exp(-128*(x-x3)**2)"
    problem.substitutions['Iext4'] = "-A4*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x4))*exp(-128*(x-x4)**2)"
    problem.substitutions['Iext5'] = "-A5*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x5))*exp(-128*(x-x5)**2)"
    problem.substitutions['Iext6'] = "-A6*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x6))*exp(-128*(x-x6)**2)"
    problem.substitutions['Iext7'] = "-A7*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x7))*exp(-128*(x-x7)**2)"
    problem.substitutions['Iext8'] = "-A8*(1 - (t - t0)/abs(t-t0))*(-2*128*(x-x8))*exp(-128*(x-x8)**2)"

    problem.add_equation("dt(w) - dx(u) = 0")
    problem.add_equation("dt(u) + dx(p) - mu*dx(dx(u)) + mu**2*A*dx(ddw) = Iext1 + Iext2 + Iext3 + Iext4 + Iext5 + Iext6 + Iext7 + Iext8")
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
    # Generate a sequence of images (to stack as a movie using imageJ)
    folder = 'data/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = '%i' % n
    filename += '_data.npz'
    np.savez(folder + filename, A = A, xInput = xInput, w0 = w0, dom = dom, N = N, dt=dt, iter=iter, t_array=t_array, t_plots=t_plots, rho_array=rho_array, u_array=u_array, E_array=E_array, p_array=p_array, T_array=T_array)

'''
#real x space
dx=2*dom/N
X=np.cumsum(w_array, axis=1)*dx/w0 - dom #Should make a better way to find the middle

#Plot a 2d image
#xmesh, ymesh = quad_mesh(x=x, y=t_array)
xmesh=X
ymesh=np.transpose(np.matlib.repmat(t_array, len(x), 1))

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1_plot = ax1.pcolormesh(xmesh, ymesh, rho_array, cmap='RdBu_r')
fig1.colorbar(ax1_plot,ax=ax1)
ax1.set_ylabel('t')
ax1.set_title(r'$\tilde{\rho}$')

ax2_plot = ax2.pcolormesh(xmesh, ymesh, E_array, cmap='RdBu_r')
plt.colorbar(ax2_plot,ax=ax2)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_title(r'$\tilde{E}$')

ax3_plot = ax3.pcolormesh(xmesh, ymesh, p_array, cmap='RdBu_r')
plt.colorbar(ax3_plot,ax=ax3)
ax3.set_ylabel('t')
ax3.set_title(r'$\tilde{\pi}$')

ax4_plot = ax4.pcolormesh(xmesh, ymesh, T_array, cmap='RdBu_r')
plt.colorbar(ax4_plot,ax=ax4)
ax4.set_xlabel('x')
ax4.set_ylabel('t')
ax4.set_title(r'$\tilde{\theta}$')
# plt.xticks([-dom, -dom/2, 0, dom/2, dom],[r'$-5\pi$', r'$-5\pi/2$', r'$0$', r'$5\pi/2$', r'$5\pi$'])

plt.subplots_adjust(wspace=0.5, hspace=1)
plt.savefig('try.png', dpi=300)

np.savez('data.npz', t_array=t_array, t_plots=t_plots, rho_array=rho_array, u_array=u_array, E_array=E_array, p_array=p_array, T_array=T_array)
print('data saved')
'''