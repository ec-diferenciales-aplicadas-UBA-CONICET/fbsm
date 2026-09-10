import collections
from scipy.integrate import solve_ivp
import numpy as np


class DirectSystem:
    eqs = None # (t, y, u)
    t_span = []
    x0 = []
    t_eval = []
    def __init__(self, eqs, t_span, x0, t_eval):
        self.eqs = eqs
        self.t_span = t_span
        self.x0 = x0
        self.t_eval = t_eval


class AdjSystem:
    eqs = None # (t, y, W)
    t_span = []
    pT = []
    t_eval = []
    def __init__(self, eqs, t_span, p_t, t_eval):
        self.eqs = eqs
        self.t_span = t_span
        self.pT = p_t
        self.t_eval = t_eval


class ControlParams:
    w_arr = None
    p_arr = None
    y = None
    def __init__(self, w_arr, p_arr):
        self.w_arr = w_arr
        self.p_arr = p_arr

    def with_y(self, y):
        new_params = ControlParams(self.w_arr, self.p_arr)
        new_params.y = y
        return new_params


class TerminatingCondition:
    def terminate(self, sol_con, w, sol_adj, p, u, next_u) -> bool:
        return False


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


# Not available until defining what to do on u(t, x) instead of just u(t)
#
# allows evaluation of fn by (numpy)arrays of values
# def with_array_evaluation(fn):
#     def _with_arr_ev(t):
#         if isinstance(t, np.ndarray):
#             return np.array([fn(ti) for ti in t])
#         if isinstance(t, collections.abc.Sequence):
#             return [fn(ti) for ti in t]
#         return fn(t)
#     return _with_arr_ev


class Solver:
    direct_sys: DirectSystem = None
    adj_sys: AdjSystem = None
    terminating: TerminatingCondition = None
    max_iter: int = 0
    min_iter: int = 0
    all_control_params = []

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


    def set_direct_system(self, direct: DirectSystem):
        self.direct_sys = direct
        return self

    def set_adj_system(self, adj: AdjSystem):
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


    def _make_control_fn(self):
        def u(t, y = None):
            prev_u_value = self.u0(t)
            for control_params in self.all_control_params:
                prev_u_value = self.u_f(t, prev_u_value, control_params.with_y(y))
            return prev_u_value
        return u


    def check_state(self):
        if self.direct_sys is None:
            raise Exception("Direct system not set")
        if self.adj_sys is None:
            raise Exception("Adj system not set")
        if self.u0 is None:
            raise Exception("Initial control function (u0) not set")
        if self.u_f is None:
            raise Exception("Control function not set")

    def reset(self):
        self.all_control_params = []


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
            direct = self.direct_sys
            adj = self.adj_sys
            # Solución del sistema directo
            sol_con = solve_ivp(direct.eqs, direct.t_span, direct.x0, t_eval=direct.t_eval, args=(control_function,))
            w = self.sol_interp(sol_con)

            # Solución del sistema adjunto
            sol_adj = solve_ivp(adj.eqs, adj.t_span, adj.pT, t_eval=adj.t_eval, args=(w,))
            p = self.sol_interp(sol_adj)

            # Nuevo control
            self.all_control_params.append(ControlParams(w, p))
            new_control_function = self._make_control_fn()

            if self.min_iter <= iteration:
                if self.terminating is not None and self.terminating.terminate(sol_con, w, sol_adj, p, control_function, new_control_function):
                    break
            control_function = new_control_function
            iteration += 1

        return sol_con, w, sol_adj, p, new_control_function, iteration