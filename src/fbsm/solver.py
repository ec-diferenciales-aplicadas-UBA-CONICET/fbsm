import collections
from scipy.integrate import solve_ivp
import numpy as np

class Control:
    eqs = None # (t, x, u)
    t_span = []
    x0 = []
    t_eval = []
    def __init__(self, eqs, t_span, x0, t_eval):
        self.eqs = eqs
        self.t_span = t_span
        self.x0 = x0
        self.t_eval = t_eval


class Adj:
    eqs = None # (t, x, W)
    t_span = []
    pT = []
    t_eval = []
    def __init__(self, eqs, t_span, p_t, t_eval):
        self.eqs = eqs
        self.t_span = t_span
        self.pT = p_t
        self.t_eval = t_eval


class TerminatingCondition:
    def terminate(self, sol_con, w, sol_adj, p, u, next_u) -> bool:
        return False


class ControlDistanceTerminatingCondition(TerminatingCondition):
    r = 0.0
    z = 0.0
    t_eval = []

    def __init__(self, r, z, t_eval):
        if r <= 0.0:
            raise Exception("terminating control ratio 'r' should be positive")
        if r < 0.0:
            raise Exception("terminating control zero value 'z' should be non-negative")
        if len(t_eval) == 0:
            raise Exception("t values to evaluate on control functions cannot be empty")
        self.r = r
        self.z = z
        self.t_eval = t_eval

    def terminate(self, sol_con, w, sol_adj, p, u, next_u) -> bool:
        u2 = next_u
        u1 = u
        for t in self.t_eval:
            v1 = u1(t)
            v2 = u2(t)
            if v1 == 0.0 and abs(v2) > self.z:
                return False
            if v2 == 0.0 and abs(v1) > self.z:
                return False
            diff = abs(v1 - v2)
            if diff > self.r * abs(v2):
                return False
        return True


class ManyTerminatingCondition(TerminatingCondition):
    conditions = []
    def __init__(self, conditions):
        if len(conditions) == 0:
            raise Exception("Must have at least one condition")
        for c in conditions:
            if not isinstance(c, TerminatingCondition):
                raise Exception("conditions must extend the TerminatingCondition class")
            conditions.append(c)

    def terminate(self, sol_con, w, sol_adj, p, u, next_u) -> bool:
        for c in self.conditions:
            if c.terminate(sol_con, w, sol_adj, p, u, next_u):
                return True
        return False


def strict_incr(arr) -> bool:
    if len(arr) == 0:
        return True
    for i in range(len(arr) - 1):
        if arr[i] >= arr[i+1]:
            return False
    return True

def strict_decr(arr) -> bool:
    if len(arr) == 0:
        return True
    for i in range(len(arr) - 1):
        if arr[i] <= arr[i+1]:
            return False
    return True


def flip_all(arrs):
    return [np.flip(arr) for arr in arrs]


def sort_by_t(t_arr, y_arrs):
    incr = strict_incr(t_arr)
    decr = strict_decr(t_arr)
    if incr:
        return t_arr, y_arrs
    if decr:
        return np.flip(t_arr), flip_all(y_arrs)
    # should never happen
    raise Exception("t values are not sorted (should never happen)")


# allows evaluation of fn by (numpy)arrays of values
def with_array_evaluation(fn):
    def _with_arr_ev(t):
        if isinstance(t, np.ndarray):
            return np.array([fn(ti) for ti in t])
        if isinstance(t, collections.abc.Sequence):
            return [fn(ti) for ti in t]
        return fn(t)
    return _with_arr_ev

class Solver:
    control_sys: Control = None
    adj_sys: Adj = None
    terminating: TerminatingCondition = None
    max_iter: int = 0
    min_iter: int = 0
    all_p_arr = []

    def set_max_iterations(self, n):
        if n < 0:
            raise Exception("max iterations should be non-negative (0 => infinite)")
        if n < self.min_iter:
            raise Exception("max iterations should be greater or equal than min iterations")
        self.max_iter = n
        return self

    def set_min_iterations(self, n):
        if n < 0:
            raise Exception("min iterations should be non-negative")
        if n > self.max_iter:
            raise Exception("min iterations should be lower or equal than max iterations")
        self.min_iter = n
        return self

    def set_terminating_condition(self, t: TerminatingCondition):
        self.terminating = t
        return self

    def default_interp(self, x_arr, y_arr):
        def interp_f(t):
            return np.interp(t, x_arr, y_arr)
        return interp_f

    linear_interp = default_interp

    def sol_interp(self, sol):
        result = []
        t_arr, y_arrs = sort_by_t(sol.t, sol.y)
        for y_arr in y_arrs:
            result.append(self.linear_interp(t_arr, y_arr))
        return result


    def set_control_system(self, control: Control):
        self.control_sys = control
        return self

    def set_adj_system(self, adj: Adj):
        self.adj_sys = adj
        return self

    u0 = None
    def set_initial_control_function(self, u0):
        self.u0 = u0
        return self

    u_f = None

    def set_control_function(self, u_f):
        self.u_f = u_f
        return self

    def _wrapped_control(self):
        all_p_arr = self.all_p_arr
        def u(t):
            previous_u_t = self.u0(t)
            for p_arr in all_p_arr:
                previous_u_t = self.u_f(t, previous_u_t, p_arr)
            return previous_u_t
        return u


    def check_state(self):
        if self.control_sys is None:
            raise Exception("Control system not set")
        if self.adj_sys is None:
            raise Exception("Adj system not set")
        if self.u0 is None:
            raise Exception("Initial control function (u0) not set")
        if self.u_f is None:
            raise Exception("Control function not set")

    def reset(self):
        self.all_p_arr = []


    def solve(self):
        self.check_state()

        sol_con = None
        sol_adj = None
        w = None
        p = None
        iteration = 1
        control_function = self.u0
        new_control_function = None
        while (self.max_iter == 0) or (iteration <= self.max_iter):
            control = self.control_sys
            adj = self.adj_sys
            # Solución del sistema con control
            sol_con = solve_ivp(control.eqs, control.t_span, control.x0, t_eval=control.t_eval, args=(control_function,))
            w = self.sol_interp(sol_con)

            # Solución del sistema adjunto
            sol_adj = solve_ivp(adj.eqs, adj.t_span, adj.pT, t_eval=adj.t_eval, args=(w,))
            p = self.sol_interp(sol_adj)
            self.all_p_arr.append(p)

            # Nuevo control
            new_control_function = self._wrapped_control()

            if self.min_iter <= iteration:
                if self.terminating is not None and self.terminating.terminate(sol_con, w, sol_adj, p, control_function, new_control_function):
                    break
            control_function = new_control_function
            iteration += 1

        return sol_con, w, sol_adj, p, with_array_evaluation(new_control_function), iteration