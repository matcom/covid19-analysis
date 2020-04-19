import pandas as pd
import collections
import graphviz
import numpy as np


class Simulation:
    def __init__(self, after_run=None, **parameters):
        self._states = []
        self._transitions = collections.defaultdict(list)
        self._parameters = parameters
        self._after_run = after_run

    @property
    def states(self):
        return list(self._states)

    @property
    def transitions(self):
        return dict(self._transitions)

    @property
    def parameters(self):
        return dict(self._parameters)

    def __getitem__(self, name):
        return self._parameters[name]

    def __setitem__(self, name, value):
        self._parameters[name] = value

    def add_state(self, state: str):
        self._states.append(state)

    def add_transition(self, from_state: str, to_state: str, transition):
        self._transitions[(from_state, to_state)].append(transition)

    def run(self, n: int, **starting_values: dict):
        history = []
        current_values = dict(
            {state: starting_values.get(state, 0) for state in self._states}
        )

        for i in range(n):
            history.append(current_values)
            new_values = dict(current_values)

            for from_state in self.states:
                for to_state in self.states:
                    if from_state == to_state:
                        continue

                    for tr in self._transitions[(from_state, to_state)]:
                        if callable(tr):
                            move = tr(current_values)
                        elif isinstance(tr, str):
                            move = eval(tr, self.parameters, current_values)
                        else:
                            move = tr * current_values[from_state]

                        move = max(0, min(move, new_values[from_state]))

                        new_values[from_state] -= move
                        new_values[to_state] += move

            current_values = new_values

        history = pd.DataFrame(history)

        if self._after_run:
            history = self._after_run(history)

        return history

    def graph(self):
        graph = graphviz.Digraph()

        for state in self.states:
            graph.node(state)

        for (u, v), l in self.transitions.items():
            for k in l:
                if isinstance(k, float):
                    graph.edge(u, v, label="%.3f" % k)
                elif isinstance(k, str):
                    graph.edge(u, v, label=k)
                else:
                    graph.edge(u, v, label="f")

        return graph


def compute_similarity(
    simulation: Simulation, real: pd.DataFrame, columns: list, **starting_values
):
    simulated = simulation.run(len(real), **starting_values)
    errors = 0

    for c in columns:
        errors += np.linalg.norm(simulated[c].values - real[c].values)

    return errors / len(columns), simulated


def optimize_similarity(
    simulation, real, columns: list, parameter_ranges: dict, **starting_values
):
    pass
    """
    Minimiza el valor de similaridad(error de similaridad) probando con todos los posibles valores(parameter_ranges)
    """
    from scipy.optimize import differential_evolution

    # poner en formato para pasarle los valores de los parámetros al optimizados de scipy
    rranges = []
    names_parameters = []
    for i, v in parameter_ranges.items():
        names_parameters.append(i)
        rranges.append(v)
    rranges = tuple(rranges)

    # función a optimizar
    def f(z):
        for name, x in zip(names_parameters, z):
            simulation[name] = x

        sim, series = compute_similarity(simulation, real, columns, **starting_values)
        print(f"Computing similarity for {z}={sim}")

    # llamando al optimizador
    resbrute = differential_evolution(f, rranges)

    gm = resbrute.x  # global minimum
    f_gm = resbrute.fun  # # function value at global minimum

    # devolviendo parametros con el mejor valor encontrado
    result = {}
    for i, name in enumerate(names_parameters):
        result[name] = gm[i]

    return result
