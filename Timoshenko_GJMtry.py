
'''
steps:
~set elements:n_elem,start,direction,normal,base_length,base_radius,density,youngs_modulus,shear_modulus
~Adding damping
~Adding constrain
~Adding force
~system finalization
~run simulation
'''

import numpy as np

# Import modules
from elastica.modules import BaseSystemCollection, Constraints, Forcing, Damping

# Import Cosserat Rod Class
from elastica.rod.cosserat_rod import CosseratRod

# Import Damping Class
from elastica.dissipation import AnalyticalLinearDamper

# Import Boundary Condition Classes
from elastica.boundary_conditions import OneEndFixedRod, FreeRod
from elastica.external_forces import EndpointForces

# Import Timestepping Functions
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

#setting up elements
n_elem = 100
density = 1000
nu = 1e-4 #粘性阻尼系数
E = 1e6 #杨式模量
poisson_ratio = 99 #for shear modulus 这是由下面的剪切模量推算出来的
shear_modulus = E /(poisson_ratio + 1.0)

start = np.zeros((3,))#not sure what is it for
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 3.0 #棍子长度
base_radius = 0.25 #棍子底部半径
base_area = np.pi * base_radius ** 2

end_force = np.array([-10.0, 0.0, 0.0])

#initialize time at 0.0
time = 0.0
dl = base_length / n_elem
dt = 0.01 * dl

class GJM_BeamSimulator(BaseSystemCollection,Constraints,Forcing,Damping):
    #这里虽然什么函数都没有定义，但是继承了BaseSystemCollection,Constraints,Forcing,Damping里的函数
    pass

dynamic_update_sim = GJM_BeamSimulator()

def set_up(rod):
    dynamic_update_sim.append(rod)

    #Adding damping
    dl = base_length / n_elem
    dt = 0.01 * dl
    dynamic_update_sim.dampen(rod).using(
        AnalyticalLinearDamper,
        damping_constant = nu,
        time_step = dt
    )

    #Adding constrain
    dynamic_update_sim.constrain(rod).using( 
        OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)#这里的idx是索引的意思
        )

    #Adding force
    origin_force = np.array([0.0, 0.0, 0.0])
    end_force = np.array([-10.0, 0.0, 0.0])
    ramp_up_time = 5.0
    dynamic_update_sim.add_forcing_to(rod).using(
        EndpointForces,
        origin_force,
        end_force,
        ramp_up_time = ramp_up_time
    )

shearable_rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus = shear_modulus
)

unshearable_start = np.array([0.0, -1.0, 0.0])
unshearable_rod = CosseratRod.straight_rod(
    n_elem,
    unshearable_start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus = E/(-0.85+1.0)
)

set_up(shearable_rod)
set_up(unshearable_rod)

dynamic_update_sim.finalize()

#analytical solution
# Compute beam position for sherable and unsherable beams.这是仿真结果与analytical的结果对比，所以低下那些因子可能是公式
def analytical_result(arg_rod, arg_end_force, shearing=True, n_elem=500):
    base_length = np.sum(arg_rod.rest_lengths)
    arg_s = np.linspace(0.0, base_length, n_elem)#返回有几维
    if type(arg_end_force) is np.ndarray:
        acting_force = arg_end_force[np.nonzero(arg_end_force)]
    else:
        acting_force = arg_end_force
    acting_force = np.abs(acting_force)
    linear_prefactor = -acting_force / arg_rod.shear_matrix[0, 0, 0]
    quadratic_prefactor = (
        -acting_force
        / 2.0
        * np.sum(arg_rod.rest_lengths / arg_rod.bend_matrix[0, 0, 0])
    )
    cubic_prefactor = (acting_force / 6.0) / arg_rod.bend_matrix[0, 0, 0]
    if shearing:
        return (
            arg_s,
            arg_s * linear_prefactor
            + arg_s ** 2 * quadratic_prefactor
            + arg_s ** 3 * cubic_prefactor,
        )
    else:
        return arg_s, arg_s ** 2 * quadratic_prefactor + arg_s ** 3 * cubic_prefactor

def run_and_update_plot(simulator, dt, start_time, stop_time, ax):
    from elastica.timestepper import extend_stepper_interface
    from elastica.timestepper.symplectic_steppers import PositionVerlet

    timestepper = PositionVerlet()
    do_step, stages_and_updates = extend_stepper_interface(timestepper, simulator)#这里返回的do_step是一个函数
# 
#     stepper_methods = StepperMethodCollector(ConcreteStepper)
#     do_step_method = (
#         _SystemCollectionStepper.do_step
#         if is_this_system_a_collection
#         else _SystemInstanceStepper.do_step
#     )
#     return do_step_method, stepper_methods.step_methods()
# 
    

    n_steps = int((stop_time - start_time) / dt)
    time = start_time
    for i in range(n_steps):
        time = do_step(timestepper, stages_and_updates, simulator, time, dt)
    plot_timoshenko_dynamic(shearable_rod, unshearable_rod, end_force, time, ax)
    return time

def plot_timoshenko_dynamic(shearable_rod, unshearable_rod, end_force, time, ax):
    import matplotlib.pyplot as plt
    from IPython import display

    analytical_shearable_positon = analytical_result(
        shearable_rod, end_force, shearing=True
    )
    analytical_unshearable_positon = analytical_result(
        unshearable_rod, end_force, shearing=False
    )

    ax.clear()
    ax.grid(which="major", color="grey", linestyle="-", linewidth=0.25)
    ax.plot(
        analytical_shearable_positon[0],
        analytical_shearable_positon[1],
        "k--",
        label="Timoshenko",
    )
    ax.plot(
        analytical_unshearable_positon[0],
        analytical_unshearable_positon[1],
        "k-.",
        label="Euler-Bernoulli",
    )

    ax.plot(
        shearable_rod.position_collection[2, :],
        shearable_rod.position_collection[0, :],
        "b-",
        label="shearable rod",
    )
    ax.plot(
        unshearable_rod.position_collection[2, :],
        unshearable_rod.position_collection[0, :],
        "r-",
        label="unshearable rod",
    )

    ax.legend(prop={"size": 12}, loc="lower left")
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_title("Simulation Time: %0.2f seconds" % time)
    ax.set_xlim([-0.1, 3.1])
    ax.set_ylim([-0.045, 0.002])


import matplotlib.pyplot as plt
# from IPython import display

evolve_for_time = 10.0
update_interval = 1.0e-1

# update the plot every 1 second
fig = plt.figure(figsize=(5, 4), frameon=True, dpi=150)
ax = fig.add_subplot(111)
first_interval_time = update_interval + time
last_interval_time = time + evolve_for_time
for stop_time in np.arange(
    first_interval_time, last_interval_time + dt, update_interval
):
    time = run_and_update_plot(dynamic_update_sim, dt, time, stop_time, ax)
    #display.clear_output(wait=True)
    #display.display(plt.gcf())
plt.show()
plt.close()