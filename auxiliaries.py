import numpy as np
import casadi as cas
import cvxpy
import scipy
import warnings

class Agent:
    """A discrete-time system with individual states and inputs, i.e. without dynamic coupling to other agents.

    Attributes:
    - id (int): Integer identifier.
    - state_dim (int): Dimension of the state.
    - input_dim (int): Dimension of the input.
    - dynamics (casadi function): Discrete-time dynamics of the agent, i.e. x+ = f(x,u).
    - output_dim (int): Dimension of the output.
    - output_map (casadi function): Output map of the agents, i.e. y = h(x,u).
    - average_output_map (casadi function): Output map defining a different output used for averaging.
    - current_state (narray): Current state of the agent. Stored as a one-dimensional numpy array.
    - current_average_value (narray): Current average value of the agent.
    - t (int): Current time step associated with the current state of the agent.
    - neighbours (list of agents): Neighbours of the agent.
    - stage_cost (casadi function): Function defining the stage cost (for tracking) of the agent which takes 'x', 'u', 'xT', 'uT' and return 'l', i.e. l(x, u, xT, uT), which is either called in that order or using these names,
    e.g. stage_cost(x=x[0:n], u=u[0:q], xT=xT[0:n], uT=uT[0:q])['l']
    - state_constraints (dict): Containing linear inequality constraints (A*x <= b with keys 'A' and 'b'), equality constraints (Aeq*x <= b with keys 'Aeq' and 'beq')
    - input_constraints (dict): Containing linear inequality constraints (A*u <= b with keys 'A' and 'b'), equality constraints (Aeq*u <= b with keys 'Aeq' and 'beq')
    - cooperation_constraints (dict): Containing constraints on the cooperation output and reference state and input.
        - 'Ay' (np.array): Matrix defining a pointwise-in-time affine constraint on the cooperation output, i.e. Ay@yT(k) <= by.
        - 'by' (np.array): Vector defining a pointwise-in-time affine constraint on the cooperation output, i.e. Ay@yT(k) <= by.
        - 'Ax' (np.array): Matrix defining a pointwise-in-time affine constraint on the cooperation state, i.e. Ax@xT(k) <= bx.
        - 'bx' (np.array): Vector defining a pointwise-in-time affine constraint on the cooperation state, i.e. Ax@xT(k) <= bx.
        - 'Au' (np.array): Matrix defining a pointwise-in-time affine constraint on the cooperation input, i.e. Au@uT(k) <= bu.
        - 'bu' (np.array): Vector defining the constraint on the cooperation input, i.e. Au@uT <= bu.
        - 'function' (cas.Function): Function that takes the cooperation decision variables and returns the constraint.
        - 'upper_bound' (list[np.array]): List containing the upper bounds for the constraint.
        - 'lower_bound' (list[np.array]): List containing the lower bounds for the constraint.
    - data (dict): Containing arbitrary data, e.g. trajectories.
    - coupling_constraints (list[casadi.Function]): Containing coupling constraints with neighbours defined as casadi.Functions. The constraint is defined such that this function is smaller than or equal to zero. Define coupling constraint pointwise-in-time.
    - nonlinear_constraints (list[casadi.Function]): Defining nonlinear constraints local to the agent.

    Class variables:
    - _counter (int): This is a class-level counter that increments each time an instance is generated and is used to assign the id of the agent.
    """

    _counter = 0  # class-level counter

    def __init__(self, id = None, dynamics = None, output_map = None, initial_time = 0, initial_state = None, neighbours = None, box_state_constraints=None, box_input_constraints=None, stage_cost=None, average_output_map=None, current_average_value=None, offset_cost=None, nonlinear_constraints=None, data=None):
        """
        Initialise an agent.

        Args:
        - dynamics (casadi function): Discrete-time dynamics of the agent, i.e. x+ = f(x,u).
        - output_map (casadi Function): function from state and input to output, i.e. y = h(x,u) (default output is outputs = states)
        - initial_time (int): initial time for internal time-keeping. (default 0)
        - initial_state (numpy array): initial state at the initial time. (default 0)
        - neighbours (list with agents): neighbours of this agent. (default None)
        - box_state_constraints (array): Contains a lower (first column) and upper bound (second column) for each state variable (rows).
        - box_input_constraints (array): Contains a lower (first column) and upper bound (second column) for each input variable (rows).
        - stage_cost (casadi Function): function from state and input to reals.
        - average_output_map (casadi Function): function from state and input to output used in average constraints. (default None)
        - current_average_value (array): initial value of the value that should satisfy the average constraint. (default None)
        - offset_cost (casadi Function): function penalising deviation (offset) from a desirable reference (default None)
        - nonlinear_constraints (list[casadi.Function]): Defining nonlinear constraints local to the agent.
            Currently, only pointwise-in-time constraints on the state are allowed.
            The input must be named 'x' and the output 'g'. The constraint should be non-positive if and only if the state is feasible.
        """

        # Warn if id is provided, since this has been deprecated.
        if id is not None:
            warnings.warn(
                "The 'id' parameter is deprecated and will be ignored. "
                "Unique IDs are generated automatically.",
                DeprecationWarning
            )

        self.dynamics = dynamics

        if dynamics is not None:
            # Set values as provided.
            for var_name in dynamics.name_in():
                if var_name not in ['x', 'u']:
                    raise ValueError(f"Unknown input variable '{var_name}' in dynamics function.")
            if dynamics.name_out() != ['x+']:
                raise ValueError("The dynamics function should have a single output (vector) 'x+'.")
            if dynamics.size_in('x') != dynamics.size_out('x+'):
                raise ValueError("The dynamics function should have the same input and output dimension.")
            if dynamics.size_in('x')[1] != 1:
                raise ValueError("The state entering the dynamics function should be a column vector.")
            if dynamics.size_in('u')[1] != 1:
                raise ValueError("The input entering the dynamics function should be a column vector.")

            self.state_dim = dynamics.size_in('x')[0]
            self.input_dim = dynamics.size_in('u')[0]

            # Define a symbolic state, input and output.
            self._state = cas.SX.sym('x', self.state_dim)
            self._input = cas.SX.sym('u', self.input_dim)

        # Assign an id.
        Agent._counter += 1
        self.id = Agent._counter

        # Define the output map.
        if output_map is None and dynamics is not None:
            self.output_map = cas.Function('h', [self._state, self._input], [self._state], ['x', 'u'], ['y'])
            self._output = cas.SX.sym('y', self.output_dim)
        elif output_map is not None:
            self.output_map = output_map
            self._output = cas.SX.sym('y', self.output_dim)

        # Set constraints of the agent. If no box constraints are passed, the constraints are initialised to be empty.
        self.set_box_state_constraints(box_state_constraints=box_state_constraints)
        self.set_box_input_constraints(box_input_constraints=box_input_constraints)

        # Initialize dictionaries for constraints on the cooperation output, state, and input.
        self.cooperation_constraints = {'Ay': None, 'by': None, 'Ax': None, 'bx': None, 'Au': None, 'bu': None, 'function': None, 'upper_bound': None, 'lower_bound': None}

        # Set the initial time (internal clock) and the initial state.
        self.t = initial_time
        self.current_state = initial_state

        if neighbours is None:
            self.neighbours = []
        else:
            self.neighbours = neighbours

        self.stage_cost = stage_cost
        self.offset_cost = offset_cost
        self.average_output_map = average_output_map
        if current_average_value is not None:
            self.current_average_value = current_average_value

        self.nonlinear_constraints = nonlinear_constraints
        self.coupling_constraints = None

        if data is None:
            self.data = {}  # Initialise an empty dictionary.


    def __str__(self):
        return "Agent " + str(self.id)

    def __repr__(self):
        return "Agent() " + str(self.id)

    @property
    def cooperation_constraints(self):
        return self._cooperation_constraints

    @cooperation_constraints.setter
    def cooperation_constraints(self, constraints):
        required_keys = {'Ay', 'by', 'Ax', 'bx', 'Au', 'bu', 'function', 'upper_bound', 'lower_bound'}
        if not isinstance(constraints, dict):
            raise TypeError("cooperation_constraints must be a dictionary.")
        if not required_keys.issubset(constraints.keys()):
            missing_keys = required_keys - constraints.keys()
            raise ValueError(f"cooperation_constraints is missing required keys: {missing_keys}")
        self._cooperation_constraints = constraints

    @property
    def id(self):
        return self._id
    @id.setter
    def id(self, id_value):
        """Set the ID which is an integer."""
        if isinstance(id_value, int):
            self._id = id_value
        else:
            raise TypeError("Please specify an integer as the ID.")

    @property
    def output_map(self):
        return self._output_map
    @output_map.setter
    def output_map(self, value):
        size_out = value.size_out('y')
        if size_out[1] > 1:
            raise ValueError('The output map should return a column vector.')
        self._output_map = value

    @property
    def output_dim(self):
        return self._output_map.size_out('y')[0]
    @output_dim.setter
    def output_dim(self, value):
        warnings.warn('output_dim is determined by output_map and cannot be set manually.', UserWarning)

    @property
    def current_state(self):
        return self._current_state
    @current_state.setter
    def current_state(self, state):
        if state is not None:
            if not (isinstance(state, np.ndarray) or isinstance(state, cas.DM)):
                raise TypeError("State must be a numpy array or casadi.DM.")
            # Check the state dimension.
            # The state should be saved as a two-dimensional array, even if a one-dimensional would suffice. It will be transformed into a column vector.
            # This increases compatibility with casadi's DM type, which only knows two-dimensional shapes.
            if len(state.shape) == 1 and state.shape[0] == self.state_dim:
                raise ValueError(f"State assignment failed. One-dimensional arrays are not allowed. Received: {state.shape}; expected: ({self.state_dim}, 1) or (1, {self.state_dim}).")
            elif state.shape == (self.state_dim, 1):
                self._current_state = state
            elif state.shape == (1, self.state_dim):
                self._current_state = state.T
            else:
                raise ValueError(f"State assignment failed, e.g. due to wrong dimensions. Received: {state.shape}; expected: ({self.state_dim}, 1) or (1, {self.state_dim}).")
        else:
            self._current_state = None
    @current_state.deleter
    def current_state(self):
        raise AttributeError("Do not delete, set state to 0.")

    @property
    def current_reference(self):
        return self._current_reference
    @current_reference.setter
    def current_reference(self, reference):
        # TODO validation, in particular the reference should conform to the offset cost if applicable
        # and to either the state or the output of the system (maybe only the output makes sense)
        self._current_reference = reference

    @property
    def coupling_constraints(self):
        return self._coupling_constraints
    @coupling_constraints.setter
    def coupling_constraints(self, cstr_list):
        if cstr_list is not None and not isinstance(cstr_list, list):
            raise TypeError("coupling_constraints must be a list of dictionaries.")
        if cstr_list is not None:
            for cstr in cstr_list:
                if not isinstance(cstr, cas.Function):
                    raise TypeError("The 'function' value must be a casadi.Function.")
        self._coupling_constraints = cstr_list

    @property
    def current_average_value(self):
        return self._current_average_value
    @current_average_value.setter
    def current_average_value(self, average_value):
        # TODO: validation.
        if np.shape(average_value)[0] == np.size(average_value):
            self._current_average_value = average_value
        else:
            raise AttributeError("The current average value needs to be a column vector.")

    def set_box_state_constraints(self, box_state_constraints=None):
        """
        Set constraints of the agents.

        Keyword arguments:
        - box_state_constraints (list): Contains a lower (first entry, 'lb') and upper bound (second entry, 'ub') for the state vector ('x'), i.e. lb <= x <= ub element-wise.
        """
        # Define state constraints.
        self.state_constraints = {"A": np.empty, "b": np.empty}
        if box_state_constraints is not None:
            self.state_constraints["A"] = np.vstack((-np.eye(self.state_dim), np.eye(self.state_dim)))
            self.state_constraints["b"] = np.vstack((-box_state_constraints[0]*np.ones((self.state_dim,1)), box_state_constraints[1]*np.ones((self.state_dim,1))))

    def set_box_input_constraints(self, box_input_constraints=None):
        """
        Set box input constraints for the agent.

        Keyword arguments:
        - box_input_constraints (list): Contains a lower (first entry, 'lb') and upper bound (second entry, 'ub') for the input vector ('u'), i.e. lb <= u <= ub element-wise.
        """
        # Define input constraints.
        self.input_constraints = {"A": np.empty, "b": np.empty}
        if box_input_constraints is not None:
            self.input_constraints["A"] = np.vstack((-np.eye(self.input_dim), np.eye(self.input_dim)))
            self.input_constraints["b"] = np.vstack((-box_input_constraints[0]*np.ones((self.input_dim,1)), box_input_constraints[1]*np.ones((self.input_dim,1))))

class Quadrotor(Agent):
    """
    A quadrotor agent with a 10-dimensional state and 3-dimensional input.

    State:
    - z1, z2, z3: Position in x, y, z (m)
    - theta: Pitch angle (rad)
    - phi: Roll angle (rad)
    - v1, v2, v3: Linear velocities in x, y, z (m/s)
    - omega_theta, omega_phi: Angular velocities (rad/s)

    Input:
    - u_theta: Pitch control input (arbitrary units)
    - u_phi: Roll control input (arbitrary units)
    - u_thrust: Vertical thrust input (arbitrary units)

    Output:
    - z1, z2, z3: Position in space

    Attributes:
    - h (float): Sampling time.
    - g (float): Gravitational acceleration.
    - d_phi (float): Damping for angular displacements.
    - d_omega (float): Damping for angular velocity feedback.
    - k_motor (float): Motor gain from input to angular acceleration.
    - k_thrust (float): Thrust coefficient.
    - method (str): Discretization method ('RK4', 'RK2', 'Euler').
    """

    def __init__(self, h, g=9.81, d_phi=8.0, d_omega=10.0, k_motor=10.0, k_thrust=0.91, method='Euler'):
        self.h = h
        self.g = g
        self.d_phi = d_phi
        self.d_omega = d_omega
        self.k_motor = k_motor
        self.k_thrust = k_thrust
        self.method = method

        x = cas.MX.sym('x', 10)  # State: [z1, z2, z3, theta, phi, v1, v2, v3, omega_theta, omega_phi]
        u = cas.MX.sym('u', 3)   # Input: [u_theta, u_phi, u_thrust]

        def f(x, u):
            z1, z2, z3 = x[0], x[1], x[2]
            theta, phi = x[3], x[4]
            v1, v2, v3 = x[5], x[6], x[7]
            omega_theta, omega_phi = x[8], x[9]

            u_theta, u_phi, u_thrust = u[0], u[1], u[2]

            return cas.vertcat(
                v1,                                          # z1_dot
                v2,                                          # z2_dot
                v3,                                          # z3_dot
                -self.d_phi * theta + omega_theta,           # theta_dot
                -self.d_phi * phi + omega_phi,               # phi_dot
                self.g * cas.tan(theta),                     # v1_dot
                self.g * cas.tan(phi),                       # v2_dot
                -self.g + self.k_thrust * u_thrust,          # v3_dot
                -self.d_omega * theta + self.k_motor * u_theta,  # omega_theta_dot
                -self.d_omega * phi + self.k_motor * u_phi       # omega_phi_dot
            )

        # RK4 dynamics for simulation
        k1 = f(x, u)
        k2 = f(x + (h/2) * k1, u)
        k3 = f(x + (h/2) * k2, u)
        k4 = f(x + h * k3, u)
        x_next_RK4 = x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        dynamics_RK4 = cas.Function('dynamics_RK4', [x, u], [x_next_RK4], ['x', 'u'], ['x+'])

        # Select discretization method
        if method == 'RK4':
            x_next = x_next_RK4
        elif method == 'RK2':
            k1 = f(x, u)
            k2 = f(x + (h/2) * k1, u)
            x_next = x + h * k2
        elif method == 'Euler':
            x_next = x + h * f(x, u)
        else:
            raise ValueError("Unknown method. Choose 'RK4', 'RK2', or 'Euler'.")

        dynamics = cas.Function('dynamics', [x, u], [x_next], ['x', 'u'], ['x+'])

        # Define output map (position)
        output_map = cas.Function('output', [x, u], [x[0:3]], ['x', 'u'], ['y'])

        super().__init__(dynamics=dynamics, output_map=output_map)
        self.dynamics_RK4 = dynamics_RK4

    def compute_jacobians(self, xval=None, uval=None) -> tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians of the dynamics."""
        x = cas.MX.sym('x', 10)
        u = cas.MX.sym('u', 3)
        dfdx = cas.Function('dfdx', [x, u], [cas.jacobian(self.dynamics(x, u), x)], ['x', 'u'], ['dfdx'])
        dfdu = cas.Function('dfdu', [x, u], [cas.jacobian(self.dynamics(x, u), u)], ['x', 'u'], ['dfdu'])

        if xval is not None and uval is not None:
            return dfdx(xval, uval), dfdu(xval, uval)
        return dfdx, dfdu

def compute_terminal_ingredients_for_quadrotor(agent:Agent, grid_resolution:int, num_decrease_samples:int, alpha:float, alpha_tol:float = 1e-8, references_are_equilibria:bool=False, compute_size_for_decrease:bool=True, compute_size_for_constraints:bool=True, epsilon:float=1.0, verbose:int=1, solver:str='CLARABEL'):
    """Design terminal ingredients for the 10-state quadrotor."""

    if not hasattr(agent, 'terminal_ingredients'):
        agent.terminal_ingredients = {}  # Initialize the dictionary.
        
    if references_are_equilibria:
        # Since the linearization is independent of the reference, we can compute the terminal matrices by standard design.
        # If terminal ingredients are designed for equilibria, then the LPV parametrisation consists only of a static matrix, respectively.
        # Compute the matrices that define the LPV system.
        h = agent.h
        d_phi = agent.d_phi
        d_omega = agent.d_omega
        k_motor = agent.k_motor
        k_thrust = agent.k_thrust
        g = agent.g
        A0 = np.array([
            [1., 0., 0., 0., 0.,  h, 0., 0., 0., 0.], # z1
            [0., 1., 0., 0., 0., 0.,  h, 0., 0., 0.], # z2
            [0., 0., 1., 0., 0., 0., 0.,  h, 0., 0.], # z3
            [0., 0., 0., 1 - h*d_phi, 0., 0., 0., 0.,  h, 0.], # phi1
            [0., 0., 0., 0., 1 - h*d_phi, 0., 0., 0., 0.,  h], # phi2
            [0., 0., 0., h*g, 0., 1., 0., 0., 0., 0.], # v1
            [0., 0., 0., 0., h*g, 0., 1., 0., 0., 0.], # v2
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], # v3
            [0., 0., 0., -h*d_omega, 0., 0., 0., 0., 1., 0.], # omega1
            [0., 0., 0., 0., -h*d_omega, 0., 0., 0., 0., 1.]  # omega2       
        ])
        B0 = np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., h*k_thrust],
            [h*k_motor, 0., 0.],
            [0., h*k_motor, 0.]   
        ])
        A_LPV = {'static': A0}
        B_LPV = {'static': B0}
        get_lpv_par = None  # No need to compute the LPV parameters.
        get_next_points = None  # No need to compute the next points.
        
        # Create a 'grid' with one equilibrium in order to check the decrease condition, if desired.
        x_lbs, x_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Ax'], agent.cooperation_constraints['bx'])
        if compute_size_for_decrease:
            grid = {'xT': [np.vstack([x_lbs[0], x_lbs[1], x_lbs[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])], 'uT': [np.vstack([0.0, 0.0, agent.g/agent.k_thrust])]}
        if compute_size_for_constraints:
            if grid_resolution < 2:
                grid_resolution = 2
            # Compute a grid at the boundary of the constraints to compute the size for constraint satisfaction.
            x_samples = [np.linspace(x_lbs[i], x_ubs[i], grid_resolution) for i in range(3)]  # Only the position can vary for an equilibrium.
            x_mesh = np.meshgrid(*x_samples, indexing='ij')  # 'ij' ensures Cartesian indexing.
            # Flatten each dimension.
            x_flat = [axis.ravel() for axis in x_mesh]
            # Combine the state dimensions into one list.
            xT_list = np.column_stack(x_flat).tolist()
            [xT.extend([0.0]*7) for xT in xT_list]  # Add zeros for the velocities and angles.
            if compute_size_for_decrease:
                grid['xT'] += [np.vstack(xT) for xT in xT_list] 
                grid['uT'] += [np.vstack([0.0, 0.0, agent.g/agent.k_thrust]) for xT in xT_list]
            else:
                grid = {'xT': [np.vstack(xT) for xT in xT_list] , 'uT': [np.vstack([0.0, 0.0, agent.g/agent.k_thrust]) for xT in xT_list]}
            agent.terminal_ingredients['grid_resolution'] = grid_resolution
            
    else:        
        if agent.cooperation_constraints['Au'] is None or agent.cooperation_constraints['bu'] is None:
            raise ValueError('Affine constraints for the cooperation input are not defined in agent.cooperation_constraints!')
        if agent.cooperation_constraints['Ax'] is None or agent.cooperation_constraints['bx'] is None:
            raise ValueError('Affine constraints for the cooperation input are not defined in agent.cooperation_constraints!')
                
        n = agent.state_dim
        q = agent.input_dim
        
        # The LPV parameterisation is explicitly designed for a discrete-time model achieved by Euler discretisation.
        # Raise an error if this is not the chosen discretisation method.
        if agent.method != 'Euler':
            raise NotImplementedError('Terminal ingredients for trajectories can only be computed for Euler discretization!')

        xTsym = cas.MX.sym('xT', n, 1)
        uTsym = cas.MX.sym('uT', q, 1)
        
        get_lpv_par = cas.Function(
            f'A{agent.id}_lpv_parameter',
            [xTsym, uTsym],
            [
                1 / ( cas.cos(xTsym[3])**2 ),
                1 / ( cas.cos(xTsym[4])**2 )
            ],
            [xTsym.name(), uTsym.name()],
            ['par1', 'par2']
        )
                
        # Compute the matrices that define the LPV system.
        h = agent.h
        d_phi = agent.d_phi
        d_omega = agent.d_omega
        k_motor = agent.k_motor
        k_thrust = agent.k_thrust
        g = agent.g
        A0 = np.array([
            [1., 0., 0., 0., 0.,  h, 0., 0., 0., 0.], # z1
            [0., 1., 0., 0., 0., 0.,  h, 0., 0., 0.], # z2
            [0., 0., 1., 0., 0., 0., 0.,  h, 0., 0.], # z3
            [0., 0., 0., 1 - h*d_phi, 0., 0., 0., 0.,  h, 0.], # phi1
            [0., 0., 0., 0., 1 - h*d_phi, 0., 0., 0., 0.,  h], # phi2
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], # v1
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], # v2
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], # v3
            [0., 0., 0., -h*d_omega, 0., 0., 0., 0., 1., 0.], # omega1
            [0., 0., 0., 0., -h*d_omega, 0., 0., 0., 0., 1.]  # omega2       
        ])
        B0 = np.array([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., h*k_thrust],
            [h*k_motor, 0., 0.],
            [0., h*k_motor, 0.]   
        ])
        A1 = np.zeros((10, 10))
        A1[5, 3] = h*g
        B1 = np.zeros((10, 3))
        A2 = np.zeros((10, 10))
        A2[6, 4] = h*g
        B2 = np.zeros((10, 3))	
        A_LPV = {'static': A0, 'par1': A1, 'par2': A2}
        B_LPV = {'static': B0, 'par1': B1, 'par2': B2} 

        # The LPV parameterisation depends only on the fourth and fifth states (pitch angle and roll angle)
        # and the constraints are assumed to be polytopic.
        # Hence, we can convexify the design. 
        
        # Compute the polytopic bounds on the constraints.
        x_lbs, x_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Ax'], agent.cooperation_constraints['bx'])
        
        # Compute the bounds on the LPV parameters.
        # The smallest absolute bound results in the largest bound on the LPV parameters.
        par1_max = np.max([1 / ( np.cos(x_lbs[3])**2 ), 1 / ( np.cos(x_ubs[3])**2 )])
        par1_min = 1.0  # The minimum of 1/cos^2(x) is 1.
        par2_max = np.max([1 / ( np.cos(x_lbs[4])**2 ), 1 / ( np.cos(x_ubs[4])**2 )])
        par2_min = 1.0  # The minimum of 1/cos^2(x) is 1.
        
        # Create the vertices.
        vertices = []
        vertices.append({'par1': par1_min, 'par2': par2_min})
        vertices.append({'par1': par1_min, 'par2': par2_max})
        vertices.append({'par1': par1_max, 'par2': par2_min})
        vertices.append({'par1': par1_max, 'par2': par2_max})
        grid = {'vertices': vertices}
        
        # Sample the reference space.
        # We use Latin Hypercube sampling since the ten dimensional state space is too large to sample uniformly.
        # The number of samples is given by the grid resolution in an abuse of naming.
        grid['reference_points'] = generate_lhs_reference_grid(agent, num_samples = int(grid_resolution), seed=42)

        agent.terminal_ingredients['grid_resolution'] = None
        
        def get_next_points(agent: Agent, grid_point: dict) -> list:
            """
            Compute the next reference point (xT+, uT+) from the current (xT, uT)
            for use in terminal cost decrease validation.
            
            Ensures that xT+ satisfies cooperation state constraints.
            Reuses uT as a valid future reference input.
            """
            xT = grid_point['xT']
            uT = grid_point['uT']

            # Propagate one step using the agent's dynamics.
            xT_next = agent.dynamics(x=xT, u=uT)[agent.dynamics.name_out()[0]]
            xT_next = np.array(xT_next)

            # Check cooperation state constraints Ax x <= b.
            Ax = agent.cooperation_constraints['Ax']
            bx = agent.cooperation_constraints['bx']
            if Ax is not None and bx is not None:
                if not np.all(Ax @ xT_next <= bx + 1e-8):
                    return []  # Skip infeasible transitions.

            return [{'xT': xT_next, 'uT': uT}]
       
    agent.terminal_ingredients.update({'get_lpv_par': get_lpv_par, 
                                       'A_LPV': A_LPV, 
                                       'B_LPV': B_LPV, 
                                       'Q':agent.stage_cost_weights['Q'], 
                                       'R':agent.stage_cost_weights['R'], 
                                       'get_next_points': get_next_points})

    prob = compute_generic_terminal_ingredients(
        agent, 
        grid, 
        num_decrease_samples = num_decrease_samples, 
        alpha = alpha, 
        compute_size_for_decrease = compute_size_for_decrease,
        alpha_tol = alpha_tol,
        compute_size_for_constraints = compute_size_for_constraints, 
        epsilon = epsilon, 
        verbose = verbose, 
        solver = solver)
    
    
def compute_generic_terminal_ingredients(agent:Agent, grid:dict, num_decrease_samples:int, alpha:float, alpha_tol:float = 1e-8, compute_size_for_decrease:bool=True, compute_size_for_constraints:bool=True, epsilon:float=1.0, parameter_threshold:float=1e-10, verbose:int=1, solver:str='MOSEK') -> cvxpy.Problem:
    """Compute generic terminal ingredients following the scheme proposed in [1] for quadratic stage costs.

    The quadratic stage cost has the form: (x - xT).T @ Q @ (x - xT) + (u - uT).T @ R @ (u - uT)
    Multiple LMIs are set up and solved using CVXPY.
    The computed terminal ingredients are saved to the attribute 'terminal_ingredients' of the agent.
    This attribute is overwritten if present.
    Currenlty, supports only polytopic constraints for the agent's state and input and cooperation reference.

    Requires packages 'cvxpy', 'scipy', and 'casadi'.

    [1] 2020 - J. Koehler et al. - A Nonlinear Model Predictive Control Framework Using Reference Generic Terminal Ingredients - IEEE TAC. doi: 10.1109/TAC.2019.2949350

    Arguments:
        - agent (Agent): Agent for which the terminal ingredients are designed.
            Must have the attribute terminal_ingredients (dict) with entries:
            - 'get_lpv_par' (casadi.Function): A function that takes a point on the cooperative trajectory (cas.MX.sym) called 'xT' and 'uT' and returns the parameters (cas.MX.sym) used in the quasi-LPV description, cf. [(11), (12); 1]. For example, 'thetas = get_lpv_par(xT=xT[2], uT=uT[2])', where thetas (dict) contains as keys the variables' names and as values the numerical value. Note that the individual thetas are scalars. If 'get_lpv_par' is set to None, then it is assumed that
            the LPV description is static, and the terminal matrix design is standard.
            - 'A_LPV' (dict): A dictionary containing as keys the names of the variables 'get_lpv_par' returns, and as respective values the matrix that is multiplied with that specific parameter to get the LPV dynamic matrix A, cf. [(11), 1]. Must also contain 'static' which is the static component (A0).
            - 'B_LPV' (dict): A dictionary containing as keys the names of the variables 'get_lpv_par' returns, and as respective values the matrix that is multiplied with that specific parameter to get the LPV dynamic matrix B, cf. [(11), 1]. Must also contain 'static' which is the static component (B0).
            - Q (np.ndarray): Weighting matrix for the state in the quadratic stage cost.
            - R (np.ndarray): Weighting matrix for the input in the quadratic stage cost.
            - 'get_next_points' (function): A function that returns the next points from a point on the grid. The function will be called using the agent as the first positional argument, and then keyword arguments; using the names of the variables in the grid and a value. If 'get_next_points' is not defined, then it is assumed that terminal ingredients for equilibria should be computed.
            For example, for a given index 'idx' to select the point on the grid, 'function({key: grid[key][idx] for key in grid})' is performed. Must return the next grid points, which is a list containing dictionaries with the same keys as the grid and the value of the points as values.
            If compute_size_for_constraints is True, then agent must have the attribute cooperation_constraints (dict) with entries:
            - 'Ax' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the reference state: Ax <= b
            - 'bx' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the reference state: Ax <= b
            - 'Au' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the reference input: Au <= b
            - 'bu' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the reference input: Au <= b
            If compute_size_for_constraints is True, then agent must have the attribute state_constraints (dict) and input_constraints (dict) with entries:
            - 'A' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the state or input: Az <= b
            - 'b' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the state or input: Az <= b
        - grid (dict): Containing a grid for the variables of the reference. Each key must correspond to the variable name in 'get_lpv_par' ('xT' and 'uT'), which will be explicitly called using these names. The values should be lists containing the respective part of the grid point. If 'get_lpv_par' is None, i.e. the LPV parametrization is static, the grid is ignored for the design of the terminal matrices, but it is required for the computation of the terminal set size, if 'compute_size_for_decrease' is set to True.
        - num_decrease_samples (int): Number of samples that are taken to check the decrease condition in the terminal set in order to determine the terminal set size.
        - alpha (float): A first guess and upper bound for the terminal set size.
        - alpha_tol (float): Tolerance of the terminal set size. If no terminal set size larger than or equal to this value can be found, the method fails. (default is 1e-8)
        - compute_size_for_decrease (bool): Whether to compute the terminal set size such that the decrease condition is satisfied. (default is True)
        - compute_size_for_constraints (bool): Whether to compute the terminal set size such that state and input constraints are satisfied. Since this is only supported for polytopic constraints on the state, input and cooperation reference, setting this to False skips that step and allows for manual adjustment of the terminal set size (the decrease condition is always ensured on the samples). (default is True)
        - epsilon (float): Tightening of the Lyapunov decrease equation the terminal ingredients need to satisfy, cf. [(10), 1]. (defaults to 1.0)
        - parameter_threshold (float): Threshold for the parameters of the LPV description. If the parameters are below this threshold, it is set to 0.0 to improve numerical stability.
            Note that this corresponds to solving the LMIs with the decision matrix corresponding to this parameter set to zero. Hence, a feasible solution to the problem is admissible. (default is 1e-10).
        - verbose (int): 0: No printing; 1: Printing of solution stats; 2: Solver set to verbose (default is 1)
        - solver (str): Solver that is used to solve the problem, e.g. 'CLARABEL', 'MOSEK', 'OSQP', 'SCS' (default is 'MOSEK')

    Returns:
        - (cvxpy.Problem) Solution object returned by solving the optimisation problem.
            The following is added to the agent's attribute 'terminal_ingredients':
                - 'X': A list of matrices that are multiplied with the parameters of the quasi-LPV description to obtain the terminal cost matrix, cf. [Prop. 1; 1].
                - 'Y': A list of matrices that are multiplied with the parameters of the quasi-LPV description used to obtain the terminal controller matrix, cf. [Prop. 1; 1].
                - 'size': A scalar determining the terminal set size, cf. [Sec. III.C; 1].
    """
    import cvxpy

    # Extract the functions.
    get_lpv_par = agent.terminal_ingredients['get_lpv_par']
    if 'get_next_points' in agent.terminal_ingredients:
        get_next_points = agent.terminal_ingredients['get_next_points']
    else:
        get_next_points = None

    # Extract the state dimension of the agent.
    n = agent.state_dim
    q = agent.input_dim
    # Extract the matrices of the stage cost.
    Q = agent.terminal_ingredients['Q']
    R = agent.terminal_ingredients['R']
    # Extract the dictionaries defining the LPV matrices.
    A_LPV = agent.terminal_ingredients['A_LPV']
    B_LPV = agent.terminal_ingredients['B_LPV']

    Qepsilonroot = scipy.linalg.sqrtm(Q + epsilon*np.eye(n))
    Rroot = scipy.linalg.sqrtm(R)
    # Ensure these matrices are symmetric.
    Qepsilonroot = cvxpy.Constant((Qepsilonroot + Qepsilonroot.T) / 2)
    Rroot = cvxpy.Constant((Rroot + Rroot.T) / 2)

    if 'vertices' in grid:
        # Vertices are supplied in the grid, so a convexification approach is assumed.

        # Create decision variables.
        X_min = cvxpy.Variable((n, n), name='Xmin', PSD=True)
        lambdas = {par_name: cvxpy.Variable((2*n, 2*n), name=par_name, PSD=True) for par_name in get_lpv_par.name_out()}
        X_conc = {par_name : cvxpy.Variable((n, n), name=f'X_{par_name}') for par_name in get_lpv_par.name_out()}
        Y_conc = {par_name : cvxpy.Variable((q, n), name=f'Y_{par_name}') for par_name in get_lpv_par.name_out()}

        X_dict = {'static': cvxpy.Variable((n, n), name='X0')}
        Y_dict = {'static': cvxpy.Variable((q, n), name='Y0')}
        if get_lpv_par:
            # Create a decision variable per parameter supplied by 'get_lpv_par'.
            for name_out in get_lpv_par.name_out():
                X_dict[name_out] = cvxpy.Variable((n, n), name=f'X_{name_out}')
                Y_dict[name_out] = cvxpy.Variable((q, n), name=f'Y_{name_out}')

        constraints = []
        # Add LMIs per vertex.
        for vertex in grid['vertices']:
            # Compute the dynamic matrices at the current vertex.
            A = A_LPV['static']
            B = B_LPV['static']
            # Also construct the decision variables.
            X = X_dict['static']
            Y = Y_dict['static']
            for par_name, par in vertex.items():
                A = A + float(par) * A_LPV[par_name]
                B = B + float(par) * B_LPV[par_name]
                X = X + float(par) * X_dict[par_name]
                Y = Y + float(par) * Y_dict[par_name]

            # Add constraints.
            constraints.append(X >> 0)
            constraints.append(X >> X_min)
            constraints.append(X == X.T)

            # Compute the RHS matrix of the LMI.
            lambda_sum = cvxpy.Constant(np.zeros((2*n, 2*n)))
            for name_out in get_lpv_par.name_out():
                lambda_sum = lambda_sum + vertex[name_out]**2 * lambdas[name_out]

            RHS = cvxpy.bmat([
                [lambda_sum,                                cvxpy.Constant(np.zeros((2*n, n + q)))],
                [cvxpy.Constant(np.zeros((n + q, 2*n))),    cvxpy.Constant(np.zeros((n + q, n + q)))]
            ])

            # The next point can be all vertices.
            for next_vertex in grid['vertices']:
                X_next = X_dict['static']
                for par_name, par in next_vertex.items():
                    X_next = X_next + float(par) * X_dict[par_name]

                LMI = cvxpy.bmat([
                    [X,                     X@A.T + Y.T@B.T,                    Qepsilonroot@X,                     (Rroot@Y).T],
                    [(X@A.T + Y.T@B.T).T,   X_next,                             cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.zeros((n,q)))],
                    [(Qepsilonroot@X).T,    cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.eye(n)),          cvxpy.Constant(np.zeros((n,q)))],
                    [Rroot@Y,               cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.eye(q))]
                    ])

                constraints.append(LMI >> RHS)
                constraints.append(X_next >> 0)
                constraints.append(X_next >> X_min)
                constraints.append(X_next == X_next.T)

        # Add the multi-concavity constraints.
        for par_name, par in vertex.items():
            LMI_conc = cvxpy.bmat([
                [ cvxpy.Constant(np.zeros((n, n))),                                     (A_LPV[par_name]@X_conc[par_name] + B_LPV[par_name]@Y_conc[par_name]).T ],
                [ A_LPV[par_name]@X_conc[par_name] + B_LPV[par_name]@Y_conc[par_name],  cvxpy.Constant(np.zeros((n, n)))]
            ])
            constraints.append(lambdas[par_name] >> LMI_conc)
    else:
        # Grid points are supplied in the grid.

        # Create decision variables.
        X_min = cvxpy.Variable((n, n), name='Xmin', PSD=True)

        X_dict = {'static': cvxpy.Variable((n, n), name='X0')}
        Y_dict = {'static': cvxpy.Variable((q, n), name='Y0')}
        if get_lpv_par:
            # Create a decision variable per parameter supplied by 'get_lpv_par'.
            for name_out in get_lpv_par.name_out():
                X_dict[name_out] = cvxpy.Variable((n, n), name=f'X_{name_out}')
                Y_dict[name_out] = cvxpy.Variable((q, n), name=f'Y_{name_out}')

        # Initialise all decision variables with zero, except for X0 and Y0
        for name, var in X_dict.items():
            if name != 'static':
                X_dict[name].value = np.zeros((n, n))  # Set to zero except for 'static'

        for name, var in Y_dict.items():
            if name != 'static':
                Y_dict[name].value = np.zeros((q, n))  # Set to zero except for 'static'

        constraints = []

        # Compute the length of the grid.
        if grid:
            grid_length = len(grid[next(iter(grid))])

        if grid and get_lpv_par:

            for idx in range(grid_length):
                grid_point = {key: grid[key][idx] for key in grid}  # Extract a grid point.
                thetas = get_lpv_par.call(grid_point)  # Get the LPV parameters for the grid points.
                thetas = {key: cvxpy.Constant(float(thetas[key])) if abs(thetas[key]) > parameter_threshold else 0.0 for key in thetas}  # Transform the values into scalars.

                # Compute the dynamic matrices at the current grid point.
                A = agent.terminal_ingredients['A_LPV']['static']
                B = agent.terminal_ingredients['B_LPV']['static']
                for key in thetas:
                    A = A + thetas[key]*agent.terminal_ingredients['A_LPV'][key]
                    B = B + thetas[key]*agent.terminal_ingredients['B_LPV'][key]

                # Construct the parametrisation from the decision variables.
                X = X_dict['static']
                Y = Y_dict['static']
                for theta in thetas:
                    X = X + thetas[theta]*X_dict[theta]
                    Y = Y + thetas[theta]*Y_dict[theta]

                # Add constraints.
                constraints.append(X >> 0)
                constraints.append(X >> X_min)
                constraints.append(X == X.T)

                # Compute the next points from this grid point.
                if get_next_points is not None:
                    next_grid_points = get_next_points(agent, grid_point)

                    for next_point in next_grid_points:
                        next_thetas = get_lpv_par.call(next_point)  # Get the LPV parameters for the next grid points.
                        next_thetas = {key: cvxpy.Constant(float(next_thetas[key])) if abs(next_thetas[key]) > parameter_threshold else 0.0 for key in next_thetas}  # Transform the values into scalars.
                        X_next = X_dict['static']
                        for next_theta in next_thetas:
                            X_next = X_next + next_thetas[next_theta]*X_dict[next_theta]
                        # constraints.append(X_next >> 0)
                        constraints.append(X_next >> X_min)
                        constraints.append(X_next == X_next.T)

                        LMI = cvxpy.bmat([
                            [X,                     X@A.T + Y.T@B.T,                    Qepsilonroot@X,                     (Rroot@Y).T],
                            [(X@A.T + Y.T@B.T).T,   X_next,                             cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.zeros((n,q)))],
                            [(Qepsilonroot@X).T,    cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.eye(n)),          cvxpy.Constant(np.zeros((n,q)))],
                            [Rroot@Y,               cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.eye(q))]
                            ])

                        constraints.append(LMI >> 0)
                else:
                    # If no next reference points are provided, then the terminal ingredients are computed for equilibria.
                    LMI = cvxpy.bmat([
                            [X,                     X@A.T + Y.T@B.T,                    Qepsilonroot@X,                     (Rroot@Y).T],
                            [(X@A.T + Y.T@B.T).T,   X,                                  cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.zeros((n,q)))],
                            [(Qepsilonroot@X).T,    cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.eye(n)),          cvxpy.Constant(np.zeros((n,q)))],
                            [Rroot@Y,               cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.eye(q))]
                            ])

                    constraints.append(LMI >> 0)
        else:
            # The problem is static, i.e. A and B are the same for all parameters defined by the grid points.

            # Get the LPV matrices, which are static.
            A = agent.terminal_ingredients['A_LPV']['static']
            B = agent.terminal_ingredients['B_LPV']['static']

            # Xmin can be directly used as the decision variable.
            X_dict['static'] = X_min
            X = X_dict['static']
            Y = Y_dict['static']

            # Set up the LMI.
            LMI = cvxpy.bmat([
                    [X,                     X@A.T + Y.T@B.T,                    Qepsilonroot@X,                     (Rroot@Y).T],
                    [(X@A.T + Y.T@B.T).T,   X,                                  cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.zeros((n,q)))],
                    [(Qepsilonroot@X).T,    cvxpy.Constant(np.zeros((n,n))),    cvxpy.Constant(np.eye(n)),          cvxpy.Constant(np.zeros((n,q)))],
                    [Rroot@Y,               cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.zeros((q,n))),    cvxpy.Constant(np.eye(q))]
                    ])
            # Add the LMI to the constraints.
            constraints.append(LMI >> 0)

    # Form objective.
    obj = cvxpy.Minimize(-cvxpy.log_det(X_min))

    # Form and solve the problem.
    prob = cvxpy.Problem(obj, constraints)
    # Try the solver first, then try MOSEK.
    if solver != 'MOSEK':
        try:
            prob.solve(solver=solver, verbose=(verbose > 1))  # Get the optimal value.
        except:
            if verbose > 0:
                print(f"Solver {solver} failed. Trying MOSEK.")
            solver = 'MOSEK'
            prob.solve(solver=solver, verbose=(verbose > 1), warm_start=True)  # Get the optimal value.
    else:
        prob.solve(solver=solver, verbose=(verbose > 1), warm_start=True)

    if verbose > 0:
        print(f"Solving for Agent {agent.id} ------------------------------------")
        print(f"Problem with {len(constraints)} LMI constraints.")
        print("status:", prob.status)
        print("solver:", prob.solver_stats.solver_name)
        print("optimal value", prob.value)
        print("X_min \n", X_min.value)
        # Check if the solution is positive definite and compute its condition number.
        try:
            np.linalg.cholesky(X_min.value)
            pos_def = 'positive definite'
        except:
            pos_def = 'not positive definite'
        print(f"Solution is {pos_def} with minimum eigenvalue {min(np.linalg.eigvalsh(np.linalg.inv(X_min.value)))}")

    # Validate the solution.
    _validate_generic_terminal_ingredients_solution(agent, grid, X_dict, Y_dict, X_min, epsilon)

    agent.terminal_ingredients['X'] = {theta_name: X_dict[theta_name].value for theta_name in X_dict}
    agent.terminal_ingredients['Y'] = {theta_name: Y_dict[theta_name].value for theta_name in Y_dict}

    if compute_size_for_decrease:
        alpha1 = compute_terminal_set_size_cost_decrease(agent, grid, alpha, num_decrease_samples, alpha_tol=alpha_tol, verbose=verbose)
    else:
        alpha1 = alpha
        warnings.warn("Terminal set size does not ensure cost decrease. Make adjustments as needed.", UserWarning)

    if compute_size_for_constraints:
        terminal_set_size = compute_terminal_set_size_constraint_satisfaction(agent, grid, alpha_tol, verbose, solver)
        terminal_set_size = min([alpha1, terminal_set_size])
        agent.terminal_ingredients['size'] = terminal_set_size  # Transfer the terminal set size to the agent.
    else:
        terminal_set_size = alpha1
        warnings.warn("Terminal set size does not ensure constraint satisfaction. Make adjustments as needed.", UserWarning)
        # Write the terminal set size on the agent.
        agent.terminal_ingredients['size'] = terminal_set_size

    return prob

def _validate_generic_terminal_ingredients_solution(agent, grid, X_dict, Y_dict, X_min, epsilon):
    """
    Validates the solution of compute_generic_terminal_ingredients by checking whether the LMI constraints are satisfied.
    The parameter_threshold in compute_generic_terminal_ingredients is ignored, i.e. also parameters below this threshold are considered.

    Parameters:
        agent: The agent object containing problem parameters.
        grid: The dictionary of grid points.
        X_dict: Dictionary of solved X decision variable values.
        Y_dict: Dictionary of solved Y decision variable values.
        X_min: The minimum X matrix.
        epsilon: Small positive number added to Q.

    Returns:
        validity: Boolean indicating if the solution is valid.
        failed_points: List of grid points where constraints were violated.
    """
    if 'vertices' in grid:
        # This method is not yet implemented for the convexification approach.
        # Warn the user and return that the solution is valid.
        warnings.warn("Validation for convexification approach is not implemented yet. Skipping validation.", UserWarning)
        return True, []

    get_lpv_par = agent.terminal_ingredients['get_lpv_par']
    get_next_points = agent.terminal_ingredients.get('get_next_points', None)

    n = agent.state_dim
    q = agent.input_dim
    Q = agent.terminal_ingredients['Q']
    R = agent.terminal_ingredients['R']

    X_min = X_min.value
    X_min = (X_min + X_min.T)/2

    try:
        scipy.linalg.cholesky(X_min)
    except np.linalg.LinAlgError:
        raise ValueError("X_min is not positive definite!")

    lambda_min = np.min(np.linalg.eigvalsh(X_min))
    # Check X > X_min against a shifted X_min which is positive definite.
    X_min_shifted = X_min - (0.5 * lambda_min) * np.eye(X_min.shape[0])
    try:
        scipy.linalg.cholesky(X_min_shifted)
    except np.linalg.LinAlgError:
        raise ValueError("X_min_shifted is not positive definite!")

    # Compute the length of the grid
    if grid:
        grid_length = len(grid[next(iter(grid))])

    # Iterate over grid points
    if grid and get_lpv_par:
        for idx in range(grid_length):
            grid_point = {key: grid[key][idx] for key in grid}  # Extract a grid point
            thetas = get_lpv_par.call(grid_point)  # Get LPV parameters
            thetas = {key: float(thetas[key]) for key in thetas}

            # Compute dynamic matrices
            A = agent.terminal_ingredients['A_LPV']['static']
            B = agent.terminal_ingredients['B_LPV']['static']
            for key in thetas:
                A = A + thetas[key] * agent.terminal_ingredients['A_LPV'][key]
                B = B + thetas[key] * agent.terminal_ingredients['B_LPV'][key]

            # Compute Q and R square roots
            Qepsilonroot = scipy.linalg.sqrtm(Q + epsilon * np.eye(n))
            Rroot = scipy.linalg.sqrtm(R)
            Qepsilonroot = (Qepsilonroot + Qepsilonroot.T) / 2
            Rroot = (Rroot + Rroot.T) / 2

            # Construct decision variables at this grid point
            X = X_dict['static'].value
            Y = Y_dict['static'].value
            for theta in thetas:
                X = X + thetas[theta] * X_dict[theta].value
                Y = Y + thetas[theta] * Y_dict[theta].value

            X = (X + X.T)/2

            # Check if X > X_min (positive definite difference)
            try:
                scipy.linalg.cholesky(X - X_min_shifted)
            except np.linalg.LinAlgError:
                raise ValueError(f"Constraint X > X_min failed at grid point {grid_point}")

            # Compute next points if provided
            if get_next_points is not None:
                next_grid_points = get_next_points(agent, grid_point)
                for next_point in next_grid_points:
                    next_thetas = get_lpv_par.call(next_point)
                    next_thetas = {key: float(next_thetas[key]) for key in next_thetas}

                    X_next = X_dict['static'].value
                    for next_theta in next_thetas:
                        X_next = X_next + next_thetas[next_theta] * X_dict[next_theta].value
                    X_next = (X_next + X_next.T)/2

                    try:
                        scipy.linalg.cholesky(X_next - X_min_shifted)
                    except np.linalg.LinAlgError:
                        raise ValueError(f"Constraint X_next > X_min failed at grid point {grid_point}")

                    # Form LMI
                    LMI = np.block([
                        [X, X @ A.T + Y.T @ B.T, Qepsilonroot @ X, (Rroot @ Y).T],
                        [(X @ A.T + Y.T @ B.T).T, X_next, np.zeros((n, n)), np.zeros((n, q))],
                        [(Qepsilonroot @ X).T, np.zeros((n, n)), np.eye(n), np.zeros((n, q))],
                        [Rroot @ Y, np.zeros((q, n)), np.zeros((q, n)), np.eye(q)]
                    ])

                    # Check if LMI is positive definite up to a tolerance using the Cholesky decomposition.
                    try:
                        scipy.linalg.cholesky(LMI + 1e-8 * np.eye(LMI.shape[0]))
                    except:
                        raise ValueError(f"Constraint LMI failed at grid point {grid_point} with minimal eigenvalue {np.min(np.linalg.eigvalsh(LMI))}")

            else:
                # Check equilibrium case
                LMI = np.block([
                    [X, X @ A.T + Y.T @ B.T, Qepsilonroot @ X, (Rroot @ Y).T],
                    [(X @ A.T + Y.T @ B.T).T, X, np.zeros((n, n)), np.zeros((n, q))],
                    [(Qepsilonroot @ X).T, np.zeros((n, n)), np.eye(n), np.zeros((n, q))],
                    [Rroot @ Y, np.zeros((q, n)), np.zeros((q, n)), np.eye(q)]
                ])

                # Check if LMI is positive definite up to a tolerance using the Cholesky decomposition.
                try:
                    scipy.linalg.cholesky(LMI + 1e-8 * np.eye(LMI.shape[0]))
                except:
                    raise ValueError(f"Constraint LMI failed at grid point {grid_point} with minimal eigenvalue {np.min(np.linalg.eigvalsh(LMI))}")

    else:
        # Single equilibrium case
        A = agent.terminal_ingredients['A_LPV']['static']
        B = agent.terminal_ingredients['B_LPV']['static']

        Qepsilonroot = scipy.linalg.sqrtm(Q + epsilon * np.eye(n))
        Rroot = scipy.linalg.sqrtm(R)

        X = X_dict['static'].value
        Y = Y_dict['static'].value

        # Check if X > X_min
        try:
                scipy.linalg.cholesky(X - X_min_shifted)
        except np.linalg.LinAlgError:
            raise ValueError("Constraint X > X_min failed in equilibrium case")

        LMI = np.block([
            [X, X @ A.T + Y.T @ B.T, Qepsilonroot @ X, (Rroot @ Y).T],
            [(X @ A.T + Y.T @ B.T).T, X, np.zeros((n, n)), np.zeros((n, q))],
            [(Qepsilonroot @ X).T, np.zeros((n, n)), np.eye(n), np.zeros((n, q))],
            [Rroot @ Y, np.zeros((q, n)), np.zeros((q, n)), np.eye(q)]
        ])

        # Check if LMI is positive definite up to a tolerance using the Cholesky decomposition.
        # Try again with a shifted LMI since it does not have to hold strictly.
        try:
            scipy.linalg.cholesky(LMI + 1e-8 * np.eye(LMI.shape[0]))
        except:
            raise ValueError(f"LMI not feasible with {np.min(np.linalg.eigvalsh(LMI))}.")


def generate_reference_grid(agent, resol: int):
    """
    Generate reference grid points for the state and input, based on cooperation constraints.

    Args:
        agent: Agent with cooperation constraints defined.
        resol: Number of samples per dimension for state and input (uniform).

    Returns:
        dict with keys 'xT' and 'uT', each a list of numpy column vectors.
    """
    # Get bounds from cooperation constraints.
    x_lbs, x_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Ax'], agent.cooperation_constraints['bx'])
    u_lbs, u_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Au'], agent.cooperation_constraints['bu'])

    n = agent.state_dim
    m = agent.input_dim

    if len(x_lbs) != n or len(x_ubs) != n:
        raise ValueError(f"State bounds length mismatch with state dimension.")
    if len(u_lbs) != m or len(u_ubs) != m:
        raise ValueError(f"Input bounds length mismatch with input dimension.")

    # Generate evenly spaced grid points per dimension.
    x_axes = [np.linspace(x_lbs[i], x_ubs[i], resol) for i in range(n)]
    u_axes = [np.linspace(u_lbs[i], u_ubs[i], resol) for i in range(m)]

    # Mesh and flatten.
    x_mesh = np.meshgrid(*x_axes, indexing='ij')
    u_mesh = np.meshgrid(*u_axes, indexing='ij')

    x_coords = np.column_stack([x.ravel() for x in x_mesh])
    u_coords = np.column_stack([u.ravel() for u in u_mesh])

    # Generate Cartesian product of x and u.
    xT = []
    uT = []

    for x in x_coords:
        x_col = np.vstack(x.reshape(-1, 1))  # shape (n, 1)
        for u in u_coords:
            u_col = np.vstack(u.reshape(-1, 1))  # shape (m, 1)
            xT.append(x_col)
            uT.append(u_col)

    return {'xT': xT, 'uT': uT}


def generate_lhs_reference_grid(agent, num_samples: int, seed: int = None):
    """
    Generate reference points using Latin Hypercube Sampling (LHS) from the cooperation constraint set.

    Args:
        agent: The agent with cooperation constraints defined.
        num_samples: Number of total reference points (xT, uT) to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        dict with keys 'xT' and 'uT', each a list of column vectors (np.ndarray with shape (n, 1) or (m, 1)).
    """
    from scipy.stats import qmc

    if seed is not None:
        np.random.seed(seed)

    # Get bounds for state and input references.
    x_lbs, x_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Ax'], agent.cooperation_constraints['bx'])
    u_lbs, u_ubs = get_bounds_of_affine_constraint(agent.cooperation_constraints['Au'], agent.cooperation_constraints['bu'])

    n = agent.state_dim
    m = agent.input_dim

    dim = n + m  # Total number of variables in each sample.

    # Generate LHS samples in [0, 1]^dim and scale to the correct ranges.
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    sample = sampler.random(n=num_samples)  # Shape: (num_samples, dim)

    # Rescale to the [lb, ub] bounds.
    x_bounds = np.column_stack((x_lbs, x_ubs))
    u_bounds = np.column_stack((u_lbs, u_ubs))
    bounds = np.vstack((x_bounds, u_bounds))  # Shape: (n+m, 2)

    scaled_samples = qmc.scale(sample, bounds[:, 0], bounds[:, 1])  # (num_samples, dim)

    # Split into state and input lists of column vectors.
    xT = [scaled_samples[i, :n].reshape(-1, 1) for i in range(num_samples)]
    uT = [scaled_samples[i, n:].reshape(-1, 1) for i in range(num_samples)]

    return {'xT': xT, 'uT': uT}

def compute_terminal_set_size_cost_decrease(agent:Agent, grid:dict, alpha:float, num_decrease_samples:int, alpha_tol:float = 1e-8, change_in_cost_tol:float = 1e-8, verbose:int=1) -> float:
    """Compute the terminal set size such that the terminal cost decreases, cf [Sec. III.C; 1].

    This is based on gridding the references and evaluating the cost decrease condition on the grid.
    If the cost does not decrease, the terminal set size is reduced until alpha_tol, at which point the method fails.

    [1] 2020 - J. Koehler et al. - A Nonlinear Model Predictive Control Framework Using Reference Generic Terminal Ingredients - IEEE TAC. doi: 10.1109/TAC.2019.2949350

    Arguments:
        - agent (Agent): Agent for which the terminal ingredients are designed.
        Must have the attribute terminal_ingredients (dict) with entries:
            - 'get_lpv_par' (casadi.Function or None): A function that takes a point on the cooperative trajectory (cas.MX.sym) called 'xT' and 'uT' and returns the parameters (cas.MX.sym) used in the quasi-LPV description, cf. [(11), (12); 1]. For example, 'thetas = get_lpv_par(xT=xT[2], uT=uT[2])', where thetas (dict) contains as keys the variables' names and as values the numerical value. Note that the individual thetas are scalars. If set to None, then it is assumed that the LPV description is static.
            - 'X': A list of matrices that are multiplied with the parameters of the quasi-LPV description to obtain the terminal cost matrix, cf. [Prop. 1; 1].
            - 'Y': A list of matrices that are multiplied with the parameters of the quasi-LPV description used to obtain the terminal controller matrix, cf. [Prop. 1; 1].
            For example, for a given index 'idx' to select the point on the grid, 'function({key: grid[key][idx] for key in grid})' is performed. Must return the next grid points, which is a list containing dictionaries with the same keys as the grid and the value of the points as values.
        - alpha (float): A first guess and upper bound for the terminal set size.
        - num_decrease_samples (int): Number of samples that are taken to check the decrease condition in the terminal set in order to determine the terminal set size.
        - grid (dict): Containing a grid for the variables of the reference. Each key must correspond to the variable name in 'get_lpv_par' ('xT' and 'uT'), which will be explicitly called using these names. The values should be lists containing the respective part of the grid point.
        - alpha_tol (float): Tolerance of the terminal set size. If no terminal set size larger than or equal to this value can be found, the method fails. (default is 1e-8)
        - change_in_cost_tol (float): Tolerance for the change in cost. If the change in cost is larger than this value, the terminal set size is reduced. (default is 1e-8)
        - verbose (int): 0: No printing; 1: Printing of solution stats; 2: Solver set to verbose (default is 1)

    Returns:
        - (float) The computed terminal set size
    """
    if 'vertices' in grid:
        grid = grid['reference_points']  # Extract the reference points from the grid.

    grid_length = len(grid[next(iter(grid))])  # Compute the length of the grid.

    if type(num_decrease_samples) is float:
        num_decrease_samples = int(num_decrease_samples)
    elif type(num_decrease_samples) is not int:
        raise TypeError(f"num_decrease_samples must be int, but is {type(num_decrease_samples)}!")

    # Extract functions.
    get_lpv_par = agent.terminal_ingredients['get_lpv_par']
    if 'get_next_points' in agent.terminal_ingredients:
        get_next_points = agent.terminal_ingredients['get_next_points']
    else:
        get_next_points = None

    ## Compute the terminal set size using [Alg. 1, 1]:
    # Initialize binary search bounds.
    alpha_low = alpha_tol/2  # Lower bound on feasible alpha; allow it to fall below the tolerance at which point the method fails.
    alpha_high = alpha     # Initial upper bound.
    tolerance = alpha_tol        # Desired precision on alpha.

    # Iterate over all grid points.
    for idx in range(grid_length):
        grid_point = {key: grid[key][idx] for key in grid}  # Extract a grid point.

        # Convert list entries to numpy arrays if needed.
        for key in grid_point:
            if isinstance(grid_point[key], list):
                grid_point[key] = np.vstack(grid_point[key])

        # Compute the terminal cost and control matrices.
        X = agent.terminal_ingredients['X']
        Y = agent.terminal_ingredients['Y']
        P = X['static'].copy()
        K = Y['static'].copy()
        if get_lpv_par is not None:
            thetas = get_lpv_par.call(grid_point)
            thetas = {key: np.array(thetas[key]).item() for key in thetas}
            for theta_name in thetas:
                P += thetas[theta_name] * X[theta_name]
                K += thetas[theta_name] * Y[theta_name]
        P = (P + P.T) / 2  # Ensure symmetry.

        # Validate that P is positive definite.
        try:
            np.linalg.cholesky(P)
        except:
            raise ValueError(f"Ill-defined terminal ingredients: terminal cost matrix is not positive definite for {grid_point['xT']} and {grid_point['uT']}.")

        # Invert P to get the ellipsoid definition.
        P_inv = np.linalg.inv(P)
        L = np.linalg.cholesky(P_inv)  # Cholesky factor for sampling.
        n = P_inv.shape[0]

        # Compute the terminal control gain K * P^{-1}.
        K = K @ P_inv
        L_inv = np.linalg.inv(L)

        # Determine next reference points if applicable.
        if get_next_points is not None:
            next_grid_points = get_next_points(agent, grid_point)
            if not next_grid_points:
                continue  # Skip if no next points are provided.
        else:
            next_grid_points = None

        # Binary search loop for terminal set size.
        while alpha_high - alpha_low > tolerance:
            alpha_candidate = 0.5 * (alpha_low + alpha_high)
            all_samples_ok = True  # Assume all samples pass until one fails.

            for _ in range(num_decrease_samples):
                if not all_samples_ok:
                    break
                # Sample from inside the ellipsoid of size alpha_candidate.
                v = np.random.randn(n)
                v /= np.linalg.norm(v)
                scale = np.random.uniform(0, 1) ** (1.0 / n)
                xdelta = np.sqrt(alpha_candidate) * (L_inv.T @ (scale * v))
                xdelta = xdelta.reshape((-1, 1))
                xT = grid_point['xT']
                x = xT + xdelta
                uT = grid_point['uT']
                kf = uT + K @ (x - xT)

                # Simulate one step forward.
                xnext = agent.dynamics(x=x, u=kf)[agent.dynamics.name_out()[0]]

                if next_grid_points is not None:
                    for next_grid_point in next_grid_points:
                        for key in next_grid_point:
                            if isinstance(next_grid_point[key], list):
                                next_grid_point[key] = np.vstack(next_grid_point[key])
                        xTnext = next_grid_point['xT']
                        Pn = X['static'].copy()
                        if get_lpv_par is not None:
                            thetas = get_lpv_par.call(next_grid_point)
                            thetas = {key: np.array(thetas[key]).item() for key in thetas}
                            for theta_name in thetas:
                                Pn += thetas[theta_name] * X[theta_name]
                        Pn = (Pn + Pn.T) / 2
                        try:
                            np.linalg.cholesky(Pn)
                        except:
                            raise ValueError(f"Ill-defined terminal ingredients: terminal cost matrix is not positive definite for {next_grid_point['xT']} and {next_grid_point['uT']}.")
                        Pn_inv = np.linalg.inv(Pn)
                        delta_cost = (xnext - xTnext).T @ Pn_inv @ (xnext - xTnext) - (x - xT).T @ P_inv @ (x - xT)
                        stage_cost_val = agent.stage_cost(x=x, u=kf, xT=xT, uT=uT)[agent.stage_cost.name_out()[0]]
                        change_in_cost = delta_cost + stage_cost_val
                        if change_in_cost > change_in_cost_tol:
                            all_samples_ok = False
                            break
                else:
                    delta_cost = (xnext - xT).T @ P_inv @ (xnext - xT) - (x - xT).T @ P_inv @ (x - xT)
                    stage_cost_val = agent.stage_cost(x=x, u=kf, xT=xT, uT=uT)[agent.stage_cost.name_out()[0]]
                    change_in_cost = delta_cost + stage_cost_val
                    if change_in_cost > change_in_cost_tol:
                        all_samples_ok = False
                        break  # Stop sampling if one violation is found.

            if all_samples_ok:
                alpha_low = alpha_candidate  # Try larger size.
            else:
                alpha_high = alpha_candidate  # Too large, shrink.

        alpha1 = alpha_low  # Best feasible alpha after binary search.

        if verbose >= 2:
            print(f'Point {idx} of {grid_length}: current terminal set size = {alpha1}')

    if alpha1 < alpha_tol:
        raise RuntimeError(f"Could not compute terminal set size that ensures decrease ≥ {alpha_tol}. Best candidate: {alpha1}.")
    if verbose > 0:
        print(f"Computed terminal set size {alpha1} with {num_decrease_samples} samples.")
    return alpha1


def compute_terminal_set_size_constraint_satisfaction(agent:Agent, grid:dict, alpha_tol:float = 1e-8, verbose:int=1, solver:str='CLARABEL'):
    """Compute the terminal set size that satisfies the constraints, cf [Sec. III.C; 1].

    This is only supported for polytopic constraints on the state, input and cooperation reference.

    [1] 2020 - J. Koehler et al. - A Nonlinear Model Predictive Control Framework Using Reference Generic Terminal Ingredients - IEEE TAC. doi: 10.1109/TAC.2019.2949350

    Arguments:
        - agent (Agent): Agent for which the terminal ingredients are designed.
        Must have the attribute terminal_ingredients (dict) with entries:
            - 'get_lpv_par' (casadi.Function): A function that takes a point on the cooperative trajectory (cas.MX.sym) called 'xT' and 'uT' and returns the parameters (cas.MX.sym) used in the quasi-LPV description, cf. [(11), (12); 1]. For example, 'thetas = get_lpv_par(xT=xT[2], uT=uT[2])', where thetas (dict) contains as keys the variables' names and as values the numerical value. Note that the individual thetas are scalars. If 'get_lpv_par' is set to None, then it is assumed that the LPV description is static.
            - 'X': A list of matrices that are multiplied with the parameters of the quasi-LPV description to obtain the terminal cost matrix, cf. [Prop. 1; 1].
            - 'Y': A list of matrices that are multiplied with the parameters of the quasi-LPV description used to obtain the terminal controller matrix, cf. [Prop. 1; 1].
            For example, for a given index 'idx' to select the point on the grid, 'function({key: grid[key][idx] for key in grid})' is performed. Must return the next grid points, which is a list containing dictionaries with the same keys as the grid and the value of the points as values.
        Furthermore the agent must have the attribute cooperation_constraints (dict) with entries:
            - 'Ax' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the reference state: Ax <= b
            - 'bx' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the reference state: Ax <= b
            - 'Au' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the reference input: Au <= b
            - 'bu' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the reference input: Au <= b
        In addition, the agent must have the attribute state_constraints (dict) and input_constraints (dict) with entries:
            - 'A' (np.ndarray): Defining the left-hand side of pointwise-in-time polytopic constraints on the state or input: Az <= b
            - 'b' (np.ndarray): Defining the right-hand side of pointwise-in-time polytopic constraints on the state or input: Az <= b
        - grid (dict): Containing a grid for the variables of the reference. Each key must correspond to the variable name in 'get_lpv_par' ('xT' and 'uT'), which will be explicitly called using these names. The values should be lists containing the respective part of the grid point.
        - alpha_tol (float): Tolerance of the terminal set size. If no terminal set size larger than or equal to this value can be found, the method fails. (default is 1e-8)
        - verbose (int): 0: No printing; 1: Printing of solution stats; 2: Solver set to verbose (default is 1)
        - solver (str): Solver that is used to solve the problem, e.g. 'CLARABEL', 'MOSEK', 'OSQP', 'SCS' (default is 'CLARABEL')

    Returns:
        - (float) The computed terminal set size.
    """
    import cvxpy

    if 'vertices' in grid:
        grid = grid['reference_points']  # Extract the reference points from the grid.

    # Extract the functions.
    get_lpv_par = agent.terminal_ingredients['get_lpv_par']
    # Compute the length of the grid.
    grid_length = len(grid[next(iter(grid))])

    ## Compute the largest terminal set size such that the constraints are satisfied, cf. [(18), 1]:
    L_r_1 = np.hstack((agent.state_constraints["A"], np.zeros((agent.state_constraints["b"].shape[0], agent.input_dim))))
    L_r_2 = np.hstack([np.zeros((agent.input_constraints["b"].shape[0], agent.state_dim)), agent.input_constraints["A"]])
    L_r = np.vstack([L_r_1, L_r_2])

    l_r = np.vstack([agent.state_constraints["b"], agent.input_constraints["b"]])

    alpha2 = cvxpy.Variable(1)
    alpha2.value = np.array([1e-6])
    obj_tss = cvxpy.Maximize(alpha2)
    constraints_tss = [alpha2 >= alpha_tol]

    for idx in range(grid_length):
        grid_point = {key: grid[key][idx] for key in grid}  # Extract a grid point.

        r = np.vstack([np.vstack(grid_point['xT']), np.vstack(grid_point['uT'])])

        X = agent.terminal_ingredients['X']
        Y = agent.terminal_ingredients['Y']
        # Compute the terminal cost matrix and the terminal control matrix.
        P = X['static'].copy()
        K = Y['static'].copy()

        if get_lpv_par is not None:
            thetas = get_lpv_par.call(grid_point)  # Get the LPV parameters for the grid points.
            thetas = {key: np.array(thetas[key]).item() for key in thetas}  # Transform the values into scalars.
            for theta_name in thetas:
                P = P + thetas[theta_name]*X[theta_name]
                K = K + thetas[theta_name]*Y[theta_name]
        P = 0.5*(P + P.T)

        # Compute the terminal control matrix.
        Pf = np.linalg.inv(P)
        Pf = 0.5*(Pf + Pf.T)
        K = K@Pf

        mult = scipy.linalg.sqrtm(P)@np.hstack([np.eye(agent.state_dim), K.T])

        for j in range(l_r.shape[0]):
            right_hand_side = (l_r[j] - L_r[j,:]@r)**2
            # Note that P is not inverted since the square root of the inverse of the terminal weight matrix is needed.
            left_hand_side = cvxpy.power(cvxpy.norm( mult @L_r[j,:].T), 2)
            constraints_tss.append(left_hand_side*alpha2 <= right_hand_side)

    # Form and solve the problem.
    prob_tss = cvxpy.Problem(obj_tss, constraints_tss)
    # Try the solver. If it fails, try MOSEK.
    if solver != 'MOSEK':
        try:
            prob_tss.solve(solver=solver)
        except:
            if verbose > 0:
                print(f'Solver {solver} failed. Trying MOSEK.')
            prob_tss.solve(solver='MOSEK', warm_start=True)
    else:
        prob_tss.solve(solver='MOSEK', warm_start=True)
    if verbose > 0:
        print(f"Computed terminal set size {prob_tss.value} for constraint satisfaction.")
        print("status:", prob_tss.status)
        print("solver:", prob_tss.solver_stats.solver_name)
        print("optimal value", prob_tss.value)

    return alpha2.value


def save_generic_terminal_ingredients(agent, filepath) -> None:
    """Save the generic terminal ingredients of an agent to a dill file."""
    import dill
    terminal_ingredients = agent.terminal_ingredients
    if not filepath.endswith(".pkl"):
        filepath += ".pkl"
    with open(filepath, "wb") as file:
        dill.dump(terminal_ingredients, file)


def load_generic_terminal_ingredients(agent, filepath):
    """Load the generic terminal ingredients (X, Y, terminal set size) of an agent from a dill file."""
    import dill
    # Check if the path ends with ".npz" and append if necessary
    if not filepath.endswith(".pkl"):
        filepath += ".pkl"
    with open(filepath, "rb") as file:
        terminal_ingredients = dill.load(file)
    if hasattr(agent, 'terminal_ingredients'):
        agent.terminal_ingredients.update(terminal_ingredients)
    else:
        agent.terminal_ingredients = terminal_ingredients


def get_bounds_of_affine_constraint(A, b, solver='CLARABEL'):
    """
    Find per-dimension lower and upper bounds for x in R^n
    subject to A x <= b. We assume this defines a non-empty,
    bounded (closed) polytope.

    Requires cvxpy.

    Parameters
    ----------
    - A (np.ndarray (shape = (m, n))):
        Matrix in the inequality constraints A x <= b.
    - b (np.ndarray (shape = (m,))) :
        Vector in the inequality constraints.
    - solver (str): Solver used in cvxpy. (default is 'CLARABEL')

    Returns
    -------
    - lower_bounds (np.ndarray of shape (n,)) :
        The minimal feasible value of x[i] for each dimension i.
    - upper_bounds (np.ndarray of shape (n,)):
        The maximal feasible value of x[i] for each dimension i.
    """
    import cvxpy

    A = np.asarray(A)
    b = np.asarray(b).flatten()
    m, n = A.shape

    lower_bounds = np.zeros(n)
    upper_bounds = np.zeros(n)

    # For each dimension i, solve two linear problems:
    #   (1) minimize x[i] subject to A x <= b
    #   (2) maximize x[i] subject to A x <= b
    for i in range(n):
        x_var = cvxpy.Variable(n, name="x")

        constraints = [A @ x_var <= b]

        # -----------------------
        # 1) Minimize x[i]
        # -----------------------
        objective_min = cvxpy.Minimize(x_var[i])
        prob_min = cvxpy.Problem(objective_min, constraints)
        result_min = prob_min.solve(solver=solver)

        if prob_min.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
            raise ValueError(
                f"Minimization for dimension {i} failed with status: {prob_min.status}"
            )
        lower_bounds[i] = x_var[i].value

        # -----------------------
        # 2) Maximize x[i]
        # -----------------------
        x_var2 = cvxpy.Variable(n, name="x2")
        constraints2 = [A @ x_var2 <= b]
        objective_max = cvxpy.Maximize(x_var2[i])
        prob_max = cvxpy.Problem(objective_max, constraints2)
        result_max = prob_max.solve(solver=solver)

        if prob_max.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
            raise ValueError(
                f"Maximization for dimension {i} failed with status: {prob_max.status}"
            )
        upper_bounds[i] = x_var2[i].value

    return lower_bounds, upper_bounds

def MPC_for_cooperation(agent, horizon=1, terminal_ingredients={'type': 'equality'}, warm_start=None, solver=None):
    """Implementation of the MPC problem that each agents solves. The problem is set up in casadi and solved.
    Returns the objective value, optimal input sequence, cooperative output, and cooperative equilibrium as well as the dictionary containing the result of ipopt.

    The explicit relation of the cooperative output and the cooperative equilibrium is not used. Instead, it is defined implicitely by adding a suitable constraint to the optimisation problem.

    Args:
        agent (mkmpc object): Current agent.
        horizon (int): Prediction horizon of the MPC optimisation problem.
        terminal_ingredients (dict): Dictionary that defines the terminal ingredients.
            Defaults to a terminal equality constraint.
        warm_start (list of arrays): Warm start for the MPC problem. The ordering should be:
            sequence of inputs, sequence of states, cooperation input, cooperation state, cooperation output
        ordering corresponds to ordering of 'neighbours' attribute of the agent. If None is provided, the cooperation outputs of the neighbours (i.e. of the objects) are taken, which is the intended behaviour.
            This attribute servers mainly for initialisation or tests. Defaults to None.
        solver (str): One of casadis solver options. If None is provided, IPOPT is used.

    Returns:
        dict: Solution of the MPC's optimisation problem.
    """
    # Initialise needed objects.
    objective = cas.MX(0)
    
    inequality_constraints = []
    equality_constraints = []

    # Shorthands.
    n = agent.state_dim
    q = agent.input_dim

    # Create decision variables.
    # Time steps are stacked beneath each other.
    u = cas.MX.sym('u', q*horizon, 1)
    x = cas.MX.sym('x', n*horizon, 1)
    uc = cas.MX.sym('xc', q, 1)
    xc = cas.MX.sym('xc', n, 1)
    yc = cas.MX.sym('yc', agent.output_dim, 1)
    
    # Set the standard constraints, i.e. adherence to dynamics, state and input constraints (also nonlinear) and fixing the initial state.
    (ineq_constraints, _, _, eq_constraints, _, _, ineq_constraints_names, eq_constraints_names) = get_standard_MPC_constraints(agent, u, x, horizon, 0)
    # Add these constraints.
    inequality_constraints.extend(ineq_constraints)
    # agent.local_inequality_constraints_names.extend(ineq_constraints_names)
    equality_constraints.extend(eq_constraints)
    # agent.local_equality_constraints_names.extend(eq_constraints_names)

    # Add a constraint that links the cooperation equilibrium to the cooperation output.
    equality_constraints.append(agent.output_map(xc, uc) - yc)
    # Add a constraint that enforces the cooperation equilibrium to be an equilibrium.
    equality_constraints.append(agent.dynamics(xc, uc) - xc)

    # Add constraints for the admissible cooperation references.
    inequality_constraints.append(agent.cooperation_constraints['Ay']@yc - agent.cooperation_constraints['by'])
    
    inequality_constraints.append(agent.cooperation_constraints['Ax']@xc - agent.cooperation_constraints['bx'])
    
    inequality_constraints.append(agent.cooperation_constraints['Au']@uc - agent.cooperation_constraints['bu'])
    
    ## Create the objective function:
    # Sum up the stage cost.
    stage_cost = agent.stage_cost
    objective += stage_cost(x=agent.current_state, u=u[0 : q, 0], xT=xc, uT=uc)['l']
    for k in range(1, horizon):
        objective += stage_cost(x[(k-1)*n : k*n, 0], u[k*q : (k+1)*q, 0], xc, uc)

    # Add the cooperative cost.
    for neigh_agent in agent.neighbours:
        objective += agent.bilat_coop_cost(yc, neigh_agent.current_cooperation_output)
        objective += agent.bilat_coop_cost(neigh_agent.current_cooperation_output, yc)

    # Add terminal ingredients.
    if 'type' in terminal_ingredients:
        if terminal_ingredients['type'] == 'standard':
            raise NotImplementedError("A standard terminal cost is not implemented yet.")
        elif terminal_ingredients['type'] == 'equality':
            # Use a terminal equality constraint.
            equality_constraints.append(agent.dynamics(x[horizon*n : (horizon+1)*n, 0], uc) - xc)
        elif terminal_ingredients['type'] == 'generalized':
            get_lpv_par = agent.terminal_ingredients['get_lpv_par']
            if get_lpv_par is None:
                thetas = []
            else:
                raise NotImplementedError("get_lpv_par not None is not implemented yet.")
            X = agent.terminal_ingredients['X']
            # Compute the terminal cost matrix.
            P = X['static'].copy()
            for theta_name in thetas:
                if np.linalg.norm(X[theta_name]) < 1e-10:
                    # Ignore matrices close to zero.
                    continue
                P = P + thetas[theta_name]*X[theta_name]
            if thetas:
                # Use CasADi to compute the inverse of P if it is parameter dependent.
                P = cas.inv(P)  # Invert P to get the terminal cost matrix.
            else:
                P = np.linalg.inv(P)
            # Ensure that P is symmetric.
            P = 0.5*(P + P.T)

            terminal_cost = (xc - x[(horizon-1)*n : horizon*n, 0]).T@ P @(xc - x[(horizon-1)*n : horizon*n, 0])
            objective += terminal_cost

            # Add the terminal constraint.
            inequality_constraints.append(terminal_cost - agent.terminal_ingredients['size'])
        else:
            raise ValueError('Unkown type of terminal cost. Allowed are "equality", "standard", "generalized".')
    else:
        raise AttributeError('Unknown specification of terminal cost.')

    # Create optimisation object.
    nlp = {'x':cas.vertcat(u, x, uc, xc, yc), 'f':objective, 'g':cas.vertcat(*equality_constraints, *inequality_constraints)}
    lec_shape = cas.vertcat(*equality_constraints).shape
    lic_shape = cas.vertcat(*inequality_constraints).shape
    lower_bound = np.vstack([np.zeros(lec_shape),
                            -np.inf*np.ones(lic_shape)
                            ])
    upper_bound = np.vstack([np.zeros(lec_shape),
                            np.zeros(lic_shape)
                            ])

    solver_options = {}
    if solver is None or solver == 'ipopt':
        solver_options["fixed_variable_treatment"] = 'make_constraint'
        solver_options["print_level"] = 0
        solver_options["print_user_options"] = 'yes'
        solver_options["linear_solver"] = 'ma97'
        #solver_options["nlp_scaling_method"] = 'none'
        #solver_options["print_options_documentation"] = 'yes'
        nlp_options = {'ipopt': solver_options, 'print_time': 0}
        S = cas.nlpsol('S', 'ipopt', nlp, nlp_options)
        # Solve the optimisation problem.
        if warm_start is None:
            r = S(lbg=lower_bound, ubg=upper_bound)
        else:
            r = S(x0=warm_start, lbg=lower_bound, ubg=upper_bound)
    else:
        error_str = "Solver " + str(solver) + " is not implemented yet."
        raise NotImplementedError(error_str)

    # Extract the solution.
    objective_function = r['f']
    opt_sol = r['x']

    # Extract the optimal input sequence.
    u_opt = np.copy(opt_sol[0 : q*horizon])
    # Reshape the optimal input sequence.
    u_opt = np.reshape(u_opt, (q, horizon), order='F')

    # Extract the optimal predicted state sequence.
    x_opt = np.copy(opt_sol[q*horizon : q*horizon + n*horizon])
    # Reshape the optimal predicted state sequence.
    x_opt = np.reshape(x_opt, (n, horizon), order='F')

    # Extract the optimal cooperation input.
    uc_opt = np.copy(opt_sol[q*horizon + n*horizon : q*horizon + n*horizon + q])

    # Extract the optimal cooperation state.
    xc_opt = np.copy(opt_sol[q*horizon + n*horizon + q : q*horizon + n*horizon + q + n])

    # Extract the optimal cooperation output.
    yc_opt = np.copy(opt_sol[q*horizon + n*horizon + q + n : q*horizon + n*horizon + q + n + agent.output_dim])

    # Return a dictionary with the respective values.
    return {"objective_function": objective_function,
            "u_opt": u_opt,
            "x_opt": x_opt,
            "uc_opt": uc_opt,
            "xc_opt": xc_opt,
            "yc_opt": yc_opt,
            "ipopt_sol": r}
    
    
def get_standard_MPC_constraints(agent, u, x, horizon, t=None, tol=1e-8):
    """Return the standard MPC constraints. Inequality constraints are upper bounded by zero.

    Deprecated call with x from x(0) to x(N) (i.e. containing x(0)) is passed on to deprecated function.

    Checks if the initial state is feasible up to the tolerance with respect to affine and nonlinear constraints on the state.

    The standard constraints are:
    * initial condition
    * dynamics
    * pointwise state and input constraints

    Parameters:
    - agent (mkmpc object): Agent solving the optimisation problem.
    - u (casadi MX): Decision variable, the input sequence vertically stacked (at least) from u(0) to u(horizon-1).
    - x (casadi MX): Decision variable, the state sequence vertically stacked (at least) from x(1) to x(horizon).
    - horizon (int): Prediction horizon of the MPC optimisation problem.
    - t (int): Optional current time step used for warnings. (default is None)
    - tol (float): Tolerance for checking of initial state's feasibility.

    Returns:
    In a tuple with the following order
    - constraints (list): Contains the MX equation that defines the constraints.
    - constraints_lb (list): Contains the lower bound to the MX equation at the same index in 'constraints'. For inequality constraints this is always -inf.
    - constraints_ub (list): Contains the upper bound to the MX equation at the same index in 'constraints'. If there is an upper bound, it is set to 0 and the value of the constraint     shifted.
    - eq_constraints (list): Contains the MX equation that defines the equality constraints.
    - eq_constraints (list): Contains the lower bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - eq_constraints (list): Contains the upper bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - ineq_constraints_names (list): Contains the names of the inequality constraints.
    - eq_constraints_names (list): Contains the names of the equality constraints.
    """
    # Define shorthands for state and input dimensions.
    n = agent.state_dim
    q = agent.input_dim

    if x.shape[0] > n*horizon:
        raise ValueError(f"The state decision variable should have shape ({n*horizon}, 1), but has shape {x.shape}.")

    # Collect the constraints and a corresponding upper and lower bound in a list.
    # Same indices should correspond to the same scalar constraint.
    ineq_constraints = []
    ineq_constraints_lb = []
    ineq_constraints_ub = []
    ineq_constraints_names = []
    eq_constraints = []
    eq_constraints_lb = []
    eq_constraints_ub = []
    eq_constraints_names = []

    # Get the constraints for the initial condition.
    eq_cstr, eq_cstr_lb, eq_cstr_ub = get_ic_MPC_constraints(agent, u, x, t, tol)
    eq_constraints.extend(eq_cstr)
    eq_constraints_lb.extend(eq_cstr_lb)
    eq_constraints_ub.extend(eq_cstr_ub)
    eq_constraints_names.extend(["initial condition"]*sum([c.shape[0] for c in eq_cstr_lb]))

    # Get the constraints for the dynamics as well as state and input constraints.
    cstr, cstr_lb, cstr_ub, eq_cstr, eq_cstr_lb, eq_cstr_ub, ineq_names, eq_names = get_system_MPC_constraints(agent, u, x, horizon)
    ineq_constraints.extend(cstr)
    ineq_constraints_lb.extend(cstr_lb)
    ineq_constraints_ub.extend(cstr_ub)
    ineq_constraints_names.extend(ineq_names)
    eq_constraints.extend(eq_cstr)
    eq_constraints_lb.extend(eq_cstr_lb)
    eq_constraints_ub.extend(eq_cstr_ub)
    eq_constraints_names.extend(eq_names)

    return ineq_constraints, ineq_constraints_lb, ineq_constraints_ub, eq_constraints, eq_constraints_lb, eq_constraints_ub, ineq_constraints_names, eq_constraints_names


def get_ic_MPC_constraints(agent, u, x, t=None, tol=1e-8):
    """Return the MPC constraints on the initial condition, this is a equality constraint.

    Checks if the current state is feasible with respect to the agents affine and nonlinear state constraints.

    By convention, equality constraints are always assumed to be equal to zero.

    Parameters:
    - agent (mkmpc object): Agent solving the optimization problem.
    - u (casadi MX): Decision variable, the input sequence vertically stacked (at least) from u(0) to u(horizon-1).
    - x (casadi MX): Decision variable, the state sequence vertically stacked (at least) from x(1) to x(horizon).

    Returns:
    In a tuple with the following order
    - constraints (list): Contains the MX equation that defines the constraints.
    - constraints_lb (list): Contains the lower bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - constraints_ub (list): Contains the upper bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - t (int): Optional time step at which the initial condition is evaluated. Used for warnings. (default is None)
    - tol (float): Tolerance for checking of initial state's feasibility.
    """
    # Define shorthands for state and input dimensions.
    n = agent.state_dim
    q = agent.input_dim

    # Collect the constraints and a corresponding upper and lower bound in a list.
    # Same indices should correspond to the same scalar constraint.
    constraints = []
    constraints_lb = []
    constraints_ub = []

    infeasible = False
    residual = 0.0

    # Check if the initial state is feasible.
    if "A" in agent.state_constraints and "b" in agent.state_constraints:
        A = agent.state_constraints["A"]
        b = agent.state_constraints["b"]
        cstr = A@agent.current_state - b
        if np.max(cstr) > tol:
            residual = np.max([residual, np.max(cstr)])
            infeasible = True

    if agent.nonlinear_constraints is not None:
        for cstr in agent.nonlinear_constraints:
            if len(cstr.name_out()) != 1 and cstr.name_out()[0] != 'g':
                raise ValueError(f"Nonlinear constraints must have exactly one output named 'g'. Check Agent {agent.id}.")
            if len(cstr.name_in()) == 1 and cstr.name_in()[0] == 'x':
                cstr_value = cstr(x=agent.current_state)['g']
                if np.max(cstr_value) > tol:
                    infeasible = True
                    residual = np.max([residual, np.max(cstr_value)])
            else:
                raise NotImplementedError(f"Nonlinear constraints depending on {cstr.name_in()} are not implemented.")

    # Set the initial condition as a constraints.
    # Note that the first entry of x contains the first predicted state.
    constraints.append(agent.dynamics(agent.current_state, u[0 : q, 0]) - x[0 : n, 0])
    # These constraints are equality constraints.
    constraints_lb.append(np.zeros((n,1)))
    constraints_ub.append(np.zeros((n,1)))

    if infeasible:
        if t is not None:
            warnings.warn(f"Initial state appears infeasible at t={t} with residual {residual:.3e}.", RuntimeWarning)
        else:
            warnings.warn("Initial state appears infeasible with residual {residual:.3e}", RuntimeWarning)
    return constraints, constraints_lb, constraints_ub


def get_system_MPC_constraints(agent, u, x, horizon):
    """Return the MPC constraints on dynamics and pointwise state and input constraints.

    Inequality constraints are upper bounded by zero.
    Equality constraints are equal to zero.

    Parameters:
    - agent (mkmpc object): Agent solving the optimisation problem.
    - u (casadi MX): Decision variable, the input sequence vertically stacked (at least) from u(0) to u(horizon-1).
    - x (casadi MX): Decision variable, the state sequence vertically stacked (at least) from x(1) to x(horizon).
    - horizon (int): Prediction horizon of the MPC optimisation problem.

    Returns:
    In a tuple with the following order
    - ineq_constraints (list): Contains the MX equation that defines the ineqquality constraints.
    - ineq_constraints_lb (list): Contains the lower bound to the MX equation at the same index in 'constraints'. For inequality constraints this is always -inf.
    - ineq_constraints_ub (list): Contains the upper bound to the MX equation at the same index in 'constraints'. If there is an upper bound, it is set to 0 and the value of the constraint shifted.
    - eq_constraints (list): Contains the MX equation that defines the equality constraints.
    - eq_constraints (list): Contains the lower bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - eq_constraints (list): Contains the upper bound, i.e. 0, to the MX equation at the same index in 'constraints'.
    - ineq_names (list): Contains the names of the inequality constraints.
    - eq_names (list): Contains the names of the equality constraints.
    """
    # Define shorthands for state and input dimensions.
    n = agent.state_dim
    q = agent.input_dim

    # Collect the constraints and a corresponding upper and lower bound in a list.
    # Same indices should correspond to the same scalar constraint.
    ineq_constraints = []
    ineq_constraints_lb = []
    ineq_constraints_ub = []
    ineq_names = []
    eq_constraints = []
    eq_constraints_lb = []
    eq_constraints_ub = []
    eq_names = []

    # Create constraints containing the dynamics.
    # The initial coupling between x(0), u(0) and x(1) is part of the initial condition, provided by a different method.
    for i in range(0, horizon-1):
        eq_constraints.append(agent.dynamics(x[i*n : (i+1)*n, 0], u[(i+1)*q : (i+2)*q, 0]) - x[(i+1)*n : (i+2)*n, 0])
        # These constraints are equality constraints.
        eq_constraints_lb.append(np.zeros((n,1)))
        eq_constraints_ub.append(np.zeros((n,1)))
        eq_names.extend([f'A{agent.id}_dynamics_{i}']*n)

    # Set the state constraints if there are any.
    # Do not add constraints on the fixed initial state.
    if "A" in agent.state_constraints and "b" in agent.state_constraints:
        A = agent.state_constraints["A"]
        b = agent.state_constraints["b"]
        for t in range(horizon):
            ineq_constraints.append(A@x[t*n : (t+1)*n, 0] - b)
            ineq_constraints_lb.append(-np.inf*np.ones((b.shape[0],1)))
            ineq_constraints_ub.append(np.zeros((b.shape[0],1)))
            ineq_names.extend([f'A{agent.id}_state_{t}']*b.shape[0])

    # Set the input constraints if there are any.
    if "A" in agent.input_constraints and "b" in agent.input_constraints:
        A = agent.input_constraints["A"]
        b = agent.input_constraints["b"]
        for t in range(horizon):
            ineq_constraints.append(A@u[t*q : (t+1)*q, 0] - b)
            ineq_constraints_lb.append(-np.inf*np.ones((b.shape[0],1)))
            ineq_constraints_ub.append(np.zeros((b.shape[0],1)))
            ineq_names.extend([f'A{agent.id}_input_{t}']*b.shape[0])

    # Add nonlinear constraints if there are any.
    if agent.nonlinear_constraints is not None:
        for cstr in agent.nonlinear_constraints:
            if len(cstr.name_out()) != 1 and cstr.name_out()[0] != 'g':
                raise ValueError(f"Nonlinear constraints must have exactly one output named 'g'. Check Agent {agent.id}.")
            if len(cstr.name_in()) == 1 and cstr.name_in()[0] == 'x':
                for k in range(horizon):
                    ineq_constraints.append(cstr(x=x[k*n:(k+1)*n, 0])['g'])
                    ineq_constraints_lb.append(-np.inf*np.ones(cstr.size_out('g')))
                    ineq_constraints_ub.append(np.zeros(cstr.size_out('g')))
                    ineq_names.extend([f'{cstr.name_out()[0]}_{agent.id}_x_{k}']*cstr.size_out('g')[0])
            else:
                raise NotImplementedError(f"Nonlinear constraints depending on {cstr.name_in()} are not implemented.")

    return ineq_constraints, ineq_constraints_lb, ineq_constraints_ub, eq_constraints, eq_constraints_lb, eq_constraints_ub, ineq_names, eq_names