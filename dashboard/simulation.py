import pandas as pd
import collections
import graphviz


class Simulation:
    def __init__(self, **kwargs):
        self.states = []
        self.transitions = collections.defaultdict(list)
        self.kwargs = kwargs

    def add_state(self, state: str):
        self.states.append(state)

    def add_transition(self, from_state: str, to_state: str, transition):
        self.transitions[(from_state, to_state)].append(transition)

    def run(self, n: int, **starting_values: dict):
        history = []
        current_values = dict({state: starting_values.get(state, 0) for state in self.states}, **self.kwargs)

        for i in range(n):
            history.append(current_values)
            new_values = dict(current_values)

            for from_state in self.states:
                for to_state in self.states:
                    if from_state == to_state:
                        continue

                    for tr in self.transitions[(from_state, to_state)]:
                        if callable(tr):
                            move = tr(current_values)
                        elif isinstance(tr, str):
                            move = eval(tr, globals(), current_values)
                        else:
                            move = tr * current_values[from_state]

                        move = min(move, current_values[from_state])

                        new_values[from_state] -= move
                        new_values[to_state] += move

            current_values = new_values

        return pd.DataFrame(history)

    def graph(self):
        graph = graphviz.Digraph()

        for state in self.states:
            graph.node(state)

        for (u,v), l in self.transitions.items():
            for k in l:
                if isinstance(k, float):
                    graph.edge(u, v, label="%.3f" % k)
                elif isinstance(k, str):
                    graph.edge(u, v, label=k)
                else:
                    graph.edge(u, v, label="f")

        return graph


def simulate_trayectories(history: pd.DataFrame, start:str, n_simulations:int=1000):
    raise NotImplementedError()

    steps = len(history)
    trayectories = []

    for _ in range(n_simulations):
        trayectory = []
        current_state = start

        for i in range(steps):
            mass_per_state = None
