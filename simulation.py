import streamlit as st
import pandas as pd
import random
import time
import altair as alt
import numpy as np
import collections

from typing import List, Dict
from enum import Enum


@st.cache
def load_disease_transition() -> pd.DataFrame:
    return pd.read_csv("./data/disease_transitions.csv")


@st.cache
def load_individual_cases_data() -> pd.DataFrame:
    data = []

    with open("./data/datos varios paises covid.txt") as fp:
        for line in fp:
            datum = parse_data(line)    
            if datum is not None:
                data.append(datum)

    return pd.DataFrame(data)


def parse_data(line: str) -> dict:
    pass


class TransitionEstimator:
    def __init__(self):
        self.data = load_disease_transition()
        self.states = list(set(self.data['StateFrom']))
        self._state_data = {}

        for s in self.states:
            self._state_data[s] = self.data[self.data["StateFrom"] == s]

    def transition(self, from_state, age, sex):
        """Computa las probabilidades de transición del estado `from_state` para una 
        persona con edad `age` y sexo `sex.
        """

        # De todos los datos que tenemos, vamos a ordernarlos por diferencia absoluta
        # con la edad deseada, y vamos cogiendo filas hasta acumular al menos 50
        # mediciones. 
        evidence = self.data[(self.data['StateFrom'] == from_state) & (self.data['Sex'] == sex)]
        evidence['AgeDiff'] = (evidence['Age'] - age).abs()
        evidence = evidence.sort_values(['AgeDiff', 'Count']).copy()
        evidence['CountCumul'] = evidence['Count'].cumsum()
        min_required_evidence = evidence[evidence['CountCumul'] >= 50]['CountCumul'].min()
        evidence = evidence[evidence['CountCumul'] <= min_required_evidence]

        # Ahora volvemos a agrupar por estado y recalculamos la media y varianza
        state_to = collections.defaultdict(lambda: dict(count=0, mean=0, std=0))

        for i, row in evidence.iterrows():
            d = state_to[row['StateTo']]
            d['count'] += row['Count']
            d['mean'] += row['Count'] * row['MeanDays']
            d['std'] += row['Count'] * row['StdDays']
            d['state'] = row['StateTo']

        for v in state_to.values():
            v['mean'] /= v['count']
            v['std'] /= v['count']

        # Finalmente tenemos un dataframe con forma
        # `count | mean | std | state`
        # con una entrada por cada estado
        return pd.DataFrame(state_to.values()).sort_values('count')


TRANSITIONS = TransitionEstimator()


@st.cache
def load_interaction_estimates():
    df: pd.DataFrame = pd.read_csv(
        "./data/contact_matrices_152_countries/nicaragua.csv",
        header=None,
        names=[i for i in range(5, 85, 5)]
    )
    df['age'] = [i for i in range(5, 85, 5)]

    return df.set_index("age").to_dict()


class StatePerson:
    """Estados en los que puede estar una persona.
    """
    S = "S"
    Ls = "Ls"
    Lp = "Lp"
    Ip = "Ip"
    Is = "Is"
    Iv = "Iv"
    A = "A"
    R = "R"
    H = "H"
    D = "D"


# método de tranmisión espacial, teniendo en cuenta la localidad
def spatial_transmision(regions, social, status, distance, parameters):
    """
    Método de transmisión espacial, teniendo en cuenta la localidad.

    Args:
        - regions: regiones consideradas en la simulación.
        - social: es el conjunto de grafos que describen las redes en cada región.
        - status: contiene característucas y el estado de salud en cada región, osea las medidas.
        - distance: almacena la distancia entre cada par de regiones.
        - parameters: parameters del modelo epidemiológico para cada individuo.

    Returns:
        - output: es el estado actualizado de cada persona.
    """
    # cantidad de días(steps) que dura la simulación
    simulation_time = 30

    # estadísticas de la simulación
    progress = st.progress(0)
    day = st.empty()
    sick_count = st.empty()

    # por cada paso de la simulación
    for i in range(simulation_time):

        total_individuals = 0

        # por cada región
        for region in regions:
            # por cada persona
            for ind in region:
                # actualizar estado de la persona
                ind.next_step()
                if ind.is_infectious:
                    compute_spread(ind, social, status)
                
                total_individuals += 1
                
            interventions(status)
            # movimientos
            for n_region in regions:
                if n_region != region:
                    # calcular personas que se mueven de una region a otras
                    transportations(n_region, region, distance)

        progress.progress((i + 1) / simulation_time)
        day.markdown(f"#### Día: {i+1}")
        sick_count.markdown(f"#### Individuos simulados: {total_individuals}")
        time.sleep(0.1)


def interventions(status):
    """Modifica el estado de las medidas y como influyen estas en la población.

       Los cambios se almacenan en status
    """
    pass


def transportations(n_region, region, distance):
    """Las personas que se mueven de n_region a region.
    """
    pass


def compute_spread(ind, social, status):
    """Calcula que personas serán infectadas por 'ind' en el paso actual de la simulación.
    """
    connections = eval_connections(social, ind)

    for other in connections:
        if other.state != StatePerson.S:
            continue

        if eval_infections(ind):
            other.set_state(StatePerson.Lp)


def eval_connections(social: Dict[int, Dict[int, float]], person: "Person") -> List["Person"]:
    """Devuelve las conexiones que tuvo una persona en un step de la simulación.
    """
    other_ages = social[person.age]

    for age, lam in other_ages.items():
        people = np.random.poisson(lam)

        for i in range(people):
            yield person.region.spawn(age)


def eval_infections(person) -> bool:
    """Determina si una persona cualquiera se infesta o no, dado que se cruza con "person". 

       En general depende del estado en el que se encuentra person y las probabilidades de ese estado
    """
    return True


class Person:
    total = 0

    def __init__(self, region, age, sex):
        """Crea una nueva persona que por defecto está en el estado de suseptible al virus.
        """
        Person.total += 1
        self.state = StatePerson.S
        self.next_state = None
        self.steps_remaining = None
        self.is_infectious = None
        # llamar método de estado inicial
        self.set_state(StatePerson.S)

        # la persona conoce la region a la que pertenece
        self.region = region
        self.age = age
        self.sex = sex
        self.health_conditions = None

    def next_step(self):
        """Ejecuta un step de tiempo para una persona.
        """
        if self.steps_remaining == 0:
            # actualizar state
            self.state = self.next_state
            # llamar al método del nuevo estado para definir tiempo y next_step
            # if self.state == StatePerson.S:
            #     state, time = self.p_suseptible()
            #     self.next_state = state
            #     self.steps_remaining = time
            if self.state == StatePerson.Ls:
                self.p_latent_sintoms()
            elif self.state == StatePerson.Lp:
                self.p_latent()
            elif self.state == StatePerson.Ip:
                self.p_infect()
            elif self.state == StatePerson.Is:
                self.p_infect_sitoms()
            elif self.state == StatePerson.Iv:
                self.p_infect_sintom_antiviral()
            elif self.state == StatePerson.A:
                self.p_asintomatic()
            else:
                return False
            # en los estados restantes no hay transiciones
        else:
            # decrementar los steps que faltan para cambiar de estado
            self.steps_remaining = self.steps_remaining - 1
        
        return True

    def __repr__(self):
        return f"Person(age={self.age}, sex={self.sex}, state={self.state}, steps_remaining={self.steps_remaining})"

    # Funciones de que pasa en cada estado para cada persona
    # debe devolver el tiempo que la persona va estar en ese estado y 
    # a que estado le toca pasar en el futuro.

    def set_state(self, state):
        self.next_state = state
        self.steps_remaining = 0
        self.next_step()

    def _evaluate_transition(self):
        """Computa a qué estado pasar dado el estado actual y los valores de la tabla.
        """
        df = TRANSITIONS.transition(self.state, self.age, self.sex)
        st.write(df)
        # calcular el estado de transición y el tiempo
        to_state = random.choices(df['state'].values, weights=df['count'].values, k=1)[0]
        state_data = df.set_index('state').to_dict('index')[to_state]
        time = random.normalvariate(state_data['mean'], state_data['std'])

        return to_state, int(time)

    def p_suseptible(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_latent_sintoms(self):
        self.is_infectious = True
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_latent(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_infect(self):
        self.is_infectious = True
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_infect_sitoms(self):
        self.is_infectious = True
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_infect_sintom_antiviral(self):
        self.is_infectious = True
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_asintomatic(self):
        self.is_infectious = True
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_recovered(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_hospitalized(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_death(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()


class Region:
    def __init__(self, population, initial_infected=1):
        self._recovered = 0
        self._population = population
        self._death = 0
        self._simulations = 0
        self._individuals = []

        for i in range(initial_infected):
            p = Person(self, 30, random.choice(['MALE', 'FEMALE']))
            p.set_state(StatePerson.S)
            self._individuals.append(p)

    @property
    def population(self):
        return self._population

    @property
    def recovered(self):
        return self._recovered

    def __iter__(self):
        for i in list(self._individuals):
            if i.state != StatePerson.S:
                yield i

    def spawn(self, age) -> Person:
        p = Person(self, age, random.choice(['MALE', 'FEMALE']))
        self._individuals.append(p)
        return p

    def increase_susceptibles(self, count):
        """Incrementa la cantidad de personas qeu pasan a formar parte de la simulación
        """
        self._simulations += count

    def increase_death(self, count):
        """Incrementa la cantidad de personas qeu pasan a formar parte de los fallecidos
        """
        self._death += count

    def increase_recovered(self, count):
        """Incrementa la cantidad de personas qeu pasan a formar parte de los recuperados
        """
        self._recovered += count
    

def main():
    st.title("Simulación de la epidemia")

    data = load_individual_cases_data()

    st.write(data)

    if st.button("Simular"):
        region = Region(1000, 1)
        spatial_transmision([region], load_interaction_estimates(), None, None, None)


if __name__ == "__main__":
    main()
