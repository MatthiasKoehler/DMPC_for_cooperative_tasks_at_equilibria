# %%
"""Imports"""
import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
from datetime import datetime
import auxiliaries as aux
import polytope

# %% [markdown]
# ### Set parameters for the simulation.

# %%
horizon = 20  # Set the prediction horizon.
disc_step_size = 0.05  # Set step size for discretization.
last_simulation_time = 10  # Set the simulation time in seconds.

coop_weight = 12  # Set a weight for the cooperation offset cost.

last_simulation_time = int(last_simulation_time/disc_step_size)  # Convert the simulation time into seconds.

# %%
print('Horizon is', horizon*disc_step_size, 'seconds.')
print('Simulation time is', last_simulation_time, 'which corresponds to', last_simulation_time*disc_step_size, 'seconds.')

# %%
"""Define the agents."""
num_agents = 3  # Set the number of agents.

agents = []  # Initialize a list to collect the agents in.
for i in range(num_agents):
    # Initialize the quadrotor agent.
    agents.append(aux.Quadrotor(disc_step_size))

# Specify the shift in the altitude.
altitude_shifts = [0.0, -2.0, -4.0]

# Specifiy initial states.
initial_state_list = [
  np.array([[1.0], [1.0], [0.], [0], [0], [0], [0], [0], [0], [0]]),
  np.array([[4.0], [3.0], [2.3], [0], [0], [0], [0], [0], [0], [0]]),
  np.array([[7.0], [3.0], [3.9], [0], [0], [0], [0], [0], [0], [0]])
  ]

for i, agent in enumerate(agents):
    x = cas.MX.sym('x', 10)
    u = cas.MX.sym('u', 3)
    output_map = cas.Function(
      'output', 
      [x, u], 
      [cas.vertcat(x[0], x[1], x[2] + altitude_shifts[i])], 
      ['x', 'u'], 
      ['y']) 
    agent.output_map = output_map


# %% [markdown]
# ### Constraints

# %% [markdown]
# Try out polytopes for the position on the $x$-$y$-plane.

# %%
state_constraint_vertices = []  # Save the vertices of the state constraints in this list.
state_constraint_polytopes = []  # Save the polytopes in this list to access their halfplane representation.
cooperation_constraint_vertices = []  # Save the vertices of the cooperation outputs' constraints in this list.
cooperation_constraint_polytopes = []  # Save the polytopes in this list to access their halfplane representation.

r_feasible_list = []  # Provide some admissible references on the boundary of their constraint sets.

GRAV = agents[0].g
KT = agents[0].k_thrust

# Parameters of the first polytope:
b1 = 2
b2 = 3
h1 = 1.5
h2 = 2
sp = [0.0, 0.0]  # Start point (lower left corner).
vertices = np.array([[sp[0]             , sp[1]],
                     [sp[0] + b1 + b2   , sp[1]],
                     [sp[0] + b1 + b2   , sp[1] + h1],
                     [sp[0] + b1        , sp[1] + h1 + h2],
                     [sp[0]             , sp[1] + h1 + h2]])
state_constraint_vertices.append(vertices)
state_constraint_polytopes.append(polytope.qhull(vertices))
sp_offset = [0.1, 0.1]
mult_offset = 0.9
vertices = np.array([[sp[0] + sp_offset[0]            , sp[1] + sp_offset[1]],
                     [sp[0] - sp_offset[0] + b1 + b2   , sp[1] + sp_offset[1]],
                     [sp[0] - sp_offset[0] + b1 + b2   , sp[1] - sp_offset[1] + h1],
                     [sp[0] - sp_offset[0] + b1        , sp[1] - sp_offset[1] + h1 + h2],
                     [sp[0] + sp_offset[0]            , sp[1] - sp_offset[1] + h1 + h2]])
cooperation_constraint_vertices.append(vertices)
cooperation_constraint_polytopes.append(polytope.qhull(vertices))

r_feasible_list.append([(np.array([[0.1, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T, 
                         np.array([[0.0, 0.0, GRAV/KT]]).T)])

b1 = 1.0
h1 = 3.5
sp = [3.5, 0.0]  # Start point (lower left corner).
vertices = np.array([[sp[0]        , sp[1]],
                     [sp[0] + b1   , sp[1]],
                     [sp[0] + b1   , sp[1] + h1],
                     [sp[0]        , sp[1] + h1]])
state_constraint_vertices.append(vertices)
state_constraint_polytopes.append(polytope.qhull(vertices))
sp_offset = [0.1, 0.1]
mult_offset = 0.9
vertices = np.array([[sp[0] + sp_offset[0]     , sp[1]+ sp_offset[1]],
                     [sp[0] - sp_offset[0] + b1, sp[1] + sp_offset[1]],
                     [sp[0] - sp_offset[0] + b1, sp[1] - sp_offset[1] + h1],
                     [sp[0] + sp_offset[0]     , sp[1] - sp_offset[1] + h1]])
cooperation_constraint_vertices.append(vertices)
cooperation_constraint_polytopes.append(polytope.qhull(vertices))

r_feasible_list.append([(np.array([[3.1, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T, 
                         np.array([[0.0, 0.0, GRAV/KT]]).T)])

b1 = 3
b2 = 2
h1 = 1.5
h2 = 2
sp = [3.0, 0.0]  # Start point (lower left corner).
vertices = np.array([[sp[0]             , sp[1]],
                     [sp[0] + b1 + b2   , sp[1]],
                     [sp[0] + b1 + b2   , sp[1] + h1 + h2],
                     [sp[0] + b1        , sp[1] + h1 + h2],
                     [sp[0]             , sp[1] + h1]])
state_constraint_vertices.append(vertices)
state_constraint_polytopes.append(polytope.qhull(vertices))
sp_offset = [0.1, 0.1]
mult_offset = 0.9
vertices = np.array([[sp[0] + sp_offset[0]            , sp[1] + sp_offset[1]],
                     [sp[0] - sp_offset[0] + b1 + b2  , sp[1] + sp_offset[1]],
                     [sp[0] - sp_offset[0] + b1 + b2  , sp[1] - sp_offset[1] + h1 + h2],
                     [sp[0] + sp_offset[0] + b1       , sp[1] - sp_offset[1] + h1 + h2],
                     [sp[0] + sp_offset[0]            , sp[1] - sp_offset[1] + h1]])
cooperation_constraint_vertices.append(vertices)
cooperation_constraint_polytopes.append(polytope.qhull(vertices))

r_feasible_list.append([(np.array([[3.1, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T, 
                         np.array([[0.0, 0.0, GRAV/KT]]).T)])

fig, ax = plt.subplots()
ax.grid(True, which='both')  # Draw grid lines.
ax.axis('equal')  # Make plot a box.

for vertices in state_constraint_vertices:
    ax.fill(vertices[:,0], vertices[:,1],
        edgecolor='black', linewidth=1,
        facecolor=(0.5,0.5,0.5,0.5))  # draw polytope
for vertices in cooperation_constraint_vertices:
    ax.fill(vertices[:,0], vertices[:,1],
        edgecolor='orange', linewidth=1,
        facecolor=(0.5,0.2,0.2,0.2))  # draw polytope

plt.show()

# %%
"""Set constraints for each agent."""
for i, agent in enumerate(agents):
    ## Set the state constraints.
    # Use the defined polytopes for constraints on the first and second position (the planar ones.)
    A = np.hstack([state_constraint_polytopes[i].A, np.zeros([state_constraint_polytopes[i].A.shape[0], agent.state_dim - 2])])
    b = np.vstack(state_constraint_polytopes[i].b) 
    # Allow a difference of 1.0 to the shifted base altitude of the agents.
    A = np.vstack([A, np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0, 0, 0, 0]])])
    b = np.vstack([b, np.array([[-altitude_shifts[i] + 1.0], [-(-altitude_shifts[i] - 1.0)]])])
    
    # Bound the remaining states.
    A = np.vstack([A, np.array([[0, 0, 0,  1,  0,  0,  0,  0,  0,  0], 
                                [0, 0, 0, -1,  0,  0,  0,  0,  0,  0],
                                [0, 0, 0,  0,  1,  0,  0,  0,  0,  0], 
                                [0, 0, 0,  0, -1,  0,  0,  0,  0,  0],
                                [0, 0, 0,  0,  0,  1,  0,  0,  0,  0], 
                                [0, 0, 0,  0,  0, -1,  0,  0,  0,  0],
                                [0, 0, 0,  0,  0,  0,  1,  0,  0,  0], 
                                [0, 0, 0,  0,  0,  0, -1,  0,  0,  0],
                                [0, 0, 0,  0,  0,  0,  0,  1,  0,  0], 
                                [0, 0, 0,  0,  0,  0,  0, -1,  0,  0],
                                [0, 0, 0,  0,  0,  0,  0,  0,  1,  0], 
                                [0, 0, 0,  0,  0,  0,  0,  0, -1,  0],
                                [0, 0, 0,  0,  0,  0,  0,  0,  0,  1], 
                                [0, 0, 0,  0,  0,  0,  0,  0,  0, -1]])])
    b = np.vstack([
        b, 
        np.array([
            [np.pi/4], [np.pi/4], # theta; pitch angle
            [np.pi/4], [np.pi/4], # phi; roll angle
            [2.0], [2.0], # v1
            [2.0], [2.0], # v2
            [2.0], [2.0], # v3
            [3.0], [3.0], # omega_theta
            [3.0], [3.0]  # omega_phi
        ])])
    agent.state_constraints['A'] = A
    agent.state_constraints['b'] = b

    # Set input constraints.
    agent.input_constraints['A'] = np.array([
        [ 1,  0,  0], # u1
        [-1,  0,  0], # u1
        [ 0,  1,  0], # u2
        [ 0, -1,  0], # u2
        [ 0,  0,  1], # u3
        [ 0,  0, -1]  # u3
        ])
    agent.input_constraints['b'] = np.array([
        [np.pi/9],  [np.pi/9],      # u1
        [np.pi/9],  [np.pi/9],      # u2
        [2*agent.g],   [0]])     # u3 (thrust)
    
    # Set constraints on the cooperation input reference.
    agent.cooperation_constraints['Au'] = agent.input_constraints['A']
    agent.cooperation_constraints['bu'] = np.array([
            [0.0], [0.0],  
            [0.0], [0.0],
            [19.5], [0.05]
        ])

    ## Set the state constraints.
    # Use the defined polytopes for constraints on the first and second position (the planar ones.)
    Ay = np.hstack([cooperation_constraint_polytopes[i].A, np.zeros([cooperation_constraint_polytopes[i].A.shape[0], 1])])
    by = np.vstack(cooperation_constraint_polytopes[i].b)
    agent.cooperation_constraints['Ay'] = np.vstack([Ay, np.array([[0, 0, 1], [0, 0, -1]])])
    agent.cooperation_constraints['by'] = np.vstack([by, np.array([[0.9], [0.9]])])
    
    Ax = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    agent.cooperation_constraints['Ax'] = np.vstack(
        [Ay@Ax, np.array([
            [0, 0,  1,  0,  0,  0,  0,  0,  0,  0], 
            [0, 0, -1,  0,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  1,  0,  0,  0,  0,  0,  0], 
            [0, 0,  0, -1,  0,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  1,  0,  0,  0,  0,  0], 
            [0, 0,  0,  0, -1,  0,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  1,  0,  0,  0,  0], 
            [0, 0,  0,  0,  0, -1,  0,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  1,  0,  0,  0], 
            [0, 0,  0,  0,  0,  0, -1,  0,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  1,  0,  0], 
            [0, 0,  0,  0,  0,  0,  0, -1,  0,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  1,  0], 
            [0, 0,  0,  0,  0,  0,  0,  0, -1,  0],
            [0, 0,  0,  0,  0,  0,  0,  0,  0,  1], 
            [0, 0,  0,  0,  0,  0,  0,  0,  0, -1]])])
    agent.cooperation_constraints['bx'] = np.vstack([
        by, 
        np.array([
            [-altitude_shifts[i] + 0.9], [-(-altitude_shifts[i] - 0.9)], # altitude
            [0.], [0.], # theta; pitch angle
            [0.], [0.], # phi; roll angle
            [0.], [0.], # v1
            [0.], [0.], # v2
            [0.], [0.], # v3
            [0.], [0.], # omega_theta
            [0.], [0.]  # omega_phi
        ])])
    

# %%
"""Neighbours"""
agents[0].neighbours = [agents[2]]
agents[1].neighbours = [agents[2]]
agents[2].neighbours = [agents[0], agents[1]]

# %%
"""Define the stage costs."""
for agent in agents:
    # Define the artificial equilibrium. 
    x = cas.MX.sym('x', agent.state_dim)
    u = cas.MX.sym('u', agent.input_dim)
    xT = cas.MX.sym('xT', agent.state_dim)
    uT = cas.MX.sym('uT', agent.input_dim)
    
    # Set the weight for the distance of the state to the equilibrium.
    Q = np.diag([
        1., 1., 1.,                             # z1, z2, z3
        0.1/(np.pi/4)**2, 0.1/(np.pi/4)**2,     # theta, phi
        0.01/4, 0.01/4, 1/4,                 # v1, v2, v3
        0.01/3**2, 0.01/3**2                    # omega_theta, omega_phi
    ])
            
    # Set the weight for the distance of the input to the equilibrium.
    R = np.diag([
        0.001/np.pi/9**2, 
        0.001/np.pi/9**2, 
        0.001/(2*agent.g)**2
        ])
        
    # Add stage cost to agents.
    agent.stage_cost = cas.Function(
        'stage_cost', 
        [x, u, xT, uT], 
        [ (x - xT).T@Q@(x - xT) + (u - uT).T@R@(u - uT) ],
        ['x', 'u', 'xT', 'uT'], 
        ['l'])
    agent.stage_cost_weights = {'Q': Q, 'R': R}

# %%
"""Cooperation offset cost"""
yc1 = cas.MX.sym('yc1', 3)
yc2 = cas.MX.sym('yc2', 3)

# Consensus on all outputs.
bilat_coop_cost = cas.Function('cooperation_cost', [yc1, yc2], [ coop_weight*(yc1 - yc2).T@(yc1 - yc2) ], ['yc1', 'yc2'], ['V_ij^c'])

# Set the bilateral cooperation cost for each agent.
for agent in agents:
    agent.bilat_coop_cost = bilat_coop_cost
    
del yc1, yc2

# %%
"""Terminal ingredients"""
aux.compute_terminal_ingredients_for_quadrotor(
    agent=agents[0],
    grid_resolution=1, 
    num_decrease_samples=1000, 
    alpha = 0.03,
    alpha_tol = 1e-9,
    references_are_equilibria=True,
    compute_size_for_decrease=False,
    compute_size_for_constraints=True,
    epsilon=1e-2,
    verbose=2,
    solver='MOSEK')
for agent in agents[1:]:
    agent.terminal_ingredients = agents[0].terminal_ingredients
    agent.terminal_ingredients['type'] = 'generalized'

# %% [markdown]
# ### Simulation of the closed loop

# %%
"""Apply the MPC algorithm."""
# Keep track of the closed-loop system.
closed_loop_evolution = []

for i, agent in enumerate(agents):
    # Initialize the current state of the agent.
    agent.current_state = initial_state_list[i].copy()
    # Save the closed-loop state evolution of each agent as an attribute of the agent.
    agent.cl_x = [agent.current_state.copy()]
    # Take output given by initial state as first cooperation output.
    agent.current_cooperation_output = np.copy(agent.current_state[0:3])

for t in range(last_simulation_time+1):
    if t % 10 == 0:
        print(f'Current time: {t*disc_step_size} seconds at simulation step {t}.')
    # Track the time.
    closed_loop_evolution.append([None]*len(agents))
    
    # Go in sequence over the agents.
    for agent_index, agent in enumerate(agents):
        closed_loop_evolution[t][agent_index] = {"time":t}
        agent.current_time = t
        # Keep track of the current state.
        closed_loop_evolution[t][agent_index].update({"current_state":np.copy(agent.current_state)})
        
        # Generate a warm start. Decision vector: (u, x, uc, xc, yc)
        if t == 0:
            #warm_start = np.zeros((u.shape[0] + x.shape[0] + uc.shape[0] + xc.shape[0] + yc.shape[0], 1))
            warm_start = []
            # Add warm start of the input trajectory.
            for i in range(horizon):
                warm_start.append(np.array([[0], [0], [9.81/0.91]]))
            # Add warm start of the state trajectory.
            for i in range(horizon):
                warm_start.append(agent.current_state)
            # Add warm start of the cooperation input.
            warm_start.append(np.array([[0], [0], [9.81/0.91]]))
            # Add warm start of the cooperation state.
            warm_start.append(agent.current_state)
            # Add warm start of the cooperation output.
            warm_start.append(agent.current_state[0:3])
            warm_start = np.concatenate(warm_start)
        else:
            if agent.terminal_ingredients['type'] == 'equality':
                warm_start = []
                # Append warm start of the input trajectoy by taking the old one, shifting it and appending the currently optimal equilibrium's input.
                for i in range(horizon-1):
                    warm_start.append(agent.current_MPC_sol["u_opt"][0:agent.input_dim, i+1:i+2])
                warm_start.append(agent.current_MPC_sol["uc_opt"])
                # Append warm start of the state trajectory by taking the old one, shifting it and appending the currently optimal equilibrium's state.
                for i in range(horizon-1):
                    warm_start.append(agent.current_MPC_sol["x_opt"][0:agent.state_dim, i+1:i+2])
                warm_start.append(agent.current_MPC_sol["xc_opt"])

                # Append warm start of cooperation input.
                warm_start.append(agent.current_MPC_sol["uc_opt"])
                # Append warm start of the cooperation state.
                warm_start.append(agent.current_MPC_sol["xc_opt"])
                
                # Append the warm start of the cooperation output.
                warm_start.append(agent.current_MPC_sol["yc_opt"])
                
                warm_start = np.concatenate(warm_start)
            else:
                warm_start = []
                # Create the shifted warm start.
                for i in range(horizon-1):
                    warm_start.append(agent.current_MPC_sol["u_opt"][0:agent.input_dim, i+1:i+2])
                X = agent.terminal_ingredients['X']
                Y = agent.terminal_ingredients['Y']
                P = X['static'].copy()
                K = Y['static'].copy()
                P = np.linalg.inv(P)
                # Ensure that P is symmetric.
                P = 0.5*(P + P.T)
                K_terminal_opt = K@P  # Compute the terminal control matrix.
                
                Kf = agent.current_MPC_sol["uc_opt"] + K_terminal_opt@(agent.current_MPC_sol["x_opt"][0:agent.state_dim, -1:] - agent.current_MPC_sol["xc_opt"])
                warm_start.append(Kf)
                # Append warm start of the state trajectory by taking the old one, shifting it and appending the currently optimal equilibrium's state.
                for i in range(horizon-1):
                    warm_start.append(agent.current_MPC_sol["x_opt"][0:agent.state_dim, i+1:i+2])
                warm_start.append(agent.dynamics(agent.current_MPC_sol["x_opt"][0:agent.state_dim, -1:], Kf))

                # Append warm start of cooperation input.
                warm_start.append(agent.current_MPC_sol["uc_opt"])
                # Append warm start of the cooperation state.
                warm_start.append(agent.current_MPC_sol["xc_opt"])
                
                # Append the warm start of the cooperation output, build that from the available data of the neighbours.
                yc_ws = agent.current_MPC_sol["yc_opt"]
                for neighbour in agent.neighbours:
                    yc_ws += 1e-3*(neighbour.current_MPC_sol["yc_opt"] - agent.current_MPC_sol["yc_opt"])
                warm_start.append(yc_ws)
                
                warm_start = np.concatenate(warm_start)
        
        # Solve the MPC problem.
        agent.current_MPC_sol = aux.MPC_for_cooperation(agent, horizon=horizon, warm_start=warm_start, terminal_ingredients=agent.terminal_ingredients)
        # Keep track of the solution.
        closed_loop_evolution[t][agent_index].update(agent.current_MPC_sol)
        
        # Update the current state of the agent. 
        #agent.current_state = np.copy(agent.current_MPC_sol["x_opt"][0:agent.state_dim, 1:2])
        agent.current_state = np.copy(agent.dynamics(agent.current_state, agent.current_MPC_sol["u_opt"][0:agent.input_dim, 0:1]))
        agent.cl_x.append(np.array(agent.current_state))
                
        # Update the cooperation output of the agent.
        agent.current_cooperation_output = np.copy(agent.current_MPC_sol["yc_opt"])

# %% [markdown]
# # Plots
# #### Plot the closed-loop evolution.

# %%
for agent in agents:
    if type(agent.cl_x) == list:
        agent.cl_x = np.hstack(agent.cl_x)
        
colours = [
    "#0072B2",  # blue
    "#D55E00",  # orange
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#56B4E9",  # light blue
    "#E69F00",  # yellow-orange
    "#B22222",  # red
    "#6A3D9A",  # purple
    "#117733",  # teal green
    "#88CCEE",  # cyan
    "#DDCC77",  # muted yellow-orange
]

# %%
# Extract the closed-loop evolution.
time_steps = range(0, last_simulation_time+1)

# Create a vector with time in seconds.
time = []
for t in time_steps:
    time.append(t*disc_step_size)
    
fig1_states, (ax_states_1, ax_states_2, ax_states_3) = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(20,6))
fig2_states, (ax_states_4, ax_states_5, ax_states_6) = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(20,6))
fig3_states, (ax_states_7, ax_states_8, ax_states_9, ax_states_10) = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(20,6))
fig_inputs, (ax_u1, ax_u2, ax_u3) = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(20,6))

for ax_i in fig1_states.axes:
    ax_i.grid(True)
for ax_i in fig2_states.axes:
    ax_i.grid(True)
for ax_i in fig3_states.axes:
    ax_i.grid(True)
for ax_i in fig_inputs.axes:
    ax_i.grid(True)
    
fig2, ax2 = plt.subplots()
ax2.grid(True)
ax3 = plt.figure().add_subplot(projection='3d')
ax3.grid(True)

end_states = []  # Save the state at the end of the simulation.
for agent in agents:
    # Extract state evolution of agent.
    agent_state_evo = []
    for t in time_steps:
        agent_state_evo.append(closed_loop_evolution[t][agents.index(agent)]["current_state"])
    # Build state evolution matrix.
    state_evo_mat = np.concatenate(agent_state_evo, axis=1)
    
    # Add the state at the end of the simulation.
    end_states.append(state_evo_mat[:, -1:])
    
    # Extract input evolution of agent.
    agent_input_evo = []
    for t in time_steps:
        agent_input_evo.append(closed_loop_evolution[t][agents.index(agent)]["u_opt"][0:agent.input_dim, 0:1])
    input_evo_mat = np.concatenate(agent_input_evo, axis=1)
    
    label_str = "Agent " + str(agent.id)  # For labelling the plot.
    
    ax_states_1.plot(time, state_evo_mat[0, :], label=label_str, marker='x', markersize=0)
    ax_states_1.set_ylabel('pos1')
    ax_states_1.set_xlabel('time in s')
    
    ax_states_2.plot(time, state_evo_mat[1, :], label=label_str, marker='x', markersize=0)
    ax_states_2.set_ylabel('pos2')
    ax_states_2.set_xlabel('time in s')
    
    # Plot evolution of the altitute.
    ax_states_3.plot(time, state_evo_mat[2, :], label=label_str, marker='x', markersize=0)
    ax_states_3.set_ylabel('altitute in m')
    ax_states_3.set_xlabel('time in s')
    
    ax_states_4.plot(time, np.degrees(state_evo_mat[3, :]), label=label_str, marker='x', markersize=0)
    ax_states_4.set_ylabel('x_4')
    ax_states_4.set_xlabel('time in seconds')
    
    ax_states_5.plot(time, np.degrees(state_evo_mat[4, :]), label=label_str, marker='x', markersize=0)
    ax_states_5.set_ylabel('x_5')
    ax_states_5.set_xlabel('time in seconds')
    
    ax_states_6.plot(time, state_evo_mat[5, :], label=label_str, marker='x', markersize=0)
    ax_states_6.set_ylabel('x_6')
    ax_states_6.set_xlabel('time in seconds')
    
    ax_states_7.plot(time, state_evo_mat[6, :], label=label_str, marker='x', markersize=0)
    ax_states_7.set_ylabel('x_7')
    ax_states_7.set_xlabel('time in seconds')
    
    ax_states_8.plot(time, state_evo_mat[7, :], label=label_str, marker='x', markersize=0)
    ax_states_8.set_ylabel('x_8')
    ax_states_8.set_xlabel('time in seconds')
    
    ax_states_9.plot(time, state_evo_mat[8, :], label=label_str, marker='x', markersize=0)
    ax_states_9.set_ylabel('x_9')
    ax_states_9.set_xlabel('time in seconds')
    
    ax_states_10.plot(time, state_evo_mat[9, :], label=label_str, marker='x', markersize=0)
    ax_states_10.set_ylabel('x_10')
    ax_states_10.set_xlabel('time in seconds')
    
    ax_u1.plot(time, input_evo_mat[0, :], label=label_str, marker='x', markersize=0)
    ax_u1.set_ylabel('u_1')
    ax_u1.set_xlabel('time in seconds')
    
    ax_u2.plot(time, input_evo_mat[1, :], label=label_str, marker='x', markersize=0)
    ax_u2.set_ylabel('u_2')
    ax_u2.set_xlabel('time in seconds')
    
    ax_u3.plot(time, input_evo_mat[2, :], label=label_str, marker='x', markersize=0)
    ax_u3.set_ylabel('u_3')
    ax_u3.set_xlabel('time in seconds')
        
    # Plot 2D evolution of position without altitute.
    ax2.plot(state_evo_mat[0, :], state_evo_mat[1, :], label=label_str, marker='x', markersize=0)
    
    # Plot 3D evolution of position.
    ax3.plot(state_evo_mat[0, :], state_evo_mat[1, :], state_evo_mat[2, :])

for vertices in state_constraint_vertices:
    ax2.fill(vertices[:,0], vertices[:,1],
             edgecolor='black', linewidth=1,
             facecolor=(0.05,0.05,0.05,0.1))  # draw polytope
    
# ax2.set_xlim([4.0,4.5])
# ax2.set_ylim([1.5, 2])

# %%
# Plot the evolution of the cooperation cost.
cooperation_cost_evolution = [0]*(last_simulation_time+1)

# The state is the output.
for t in range(0, last_simulation_time+1):
    for agent in agents:
        current_state = closed_loop_evolution[t][agents.index(agent)]["current_state"]
        current_output = agent.output_map(current_state, np.zeros((agent.input_dim,1)))
        for neighbour in agent.neighbours:
            neighbour_current_state = closed_loop_evolution[t][agents.index(neighbour)]["current_state"]
            neighbour_current_output = neighbour.output_map(neighbour_current_state, np.zeros((neighbour.input_dim,1)))
            cooperation_cost = bilat_coop_cost(yc1=current_output, yc2=neighbour_current_output)
            cooperation_cost_evolution[t] += float(cooperation_cost['V_ij^c'])
                 
# Plot evolution.
fig, ax = plt.subplots()
time_steps = range(0, last_simulation_time+1)
time = []
for t in time_steps:
    time.append(t*disc_step_size)
ax.plot(time, cooperation_cost_evolution, label='observed cooperation cost')
ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('time')
ax.set_ylabel('Cost for cooperation')
ax.set_title('Cost for cooperation')
ax.legend()
plt.show()

# Print the last distances between neighbours.
for agent in agents:
    print("agent ", agent.id, "------------")
    current_output = closed_loop_evolution[-1][agents.index(agent)]["current_state"][0:2]
    for neighbour in agent.neighbours:
            neighbour_output = closed_loop_evolution[t][agents.index(neighbour)]["current_state"][0:2]
            distance = np.linalg.norm(current_output - neighbour_output)
            print("agents", str(agent.id), "to", str(neighbour.id) + ":", "distance", "{:04.2e}".format(distance))


