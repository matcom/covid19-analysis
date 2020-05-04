import streamlit as st
import pandas as pd
import random
import time
import altair as alt

from typing import List
from enum import Enum


@st.cache
def load_disease_transition() -> pd.DataFrame:
    return pd.read_csv("./data/disease_transitions.csv")


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
            for ind in region.individuals:
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
        if other.is_infectious:
            continue

        if eval_infections(ind):
            other.state = StatePerson.Lp


def eval_connections(social, person) -> List["Person"]:
    """Devuelve las conexiones que tuvo una persona en un step de la simulación.
    """
    pass


def eval_infections(person) -> bool:
    """Determina si una persona cualquiera se infesta o no, dado que se cruza con "person". 

       En general depende del estado en el que se encuentra person y las probabilidades de ese estado
    """
    pass


class Person:
    def __init__(self, region, age):
        """Crea una nueva persona que por defecto está en el estado de suseptible al virus.
        """
        self.state = StatePerson.S
        self.next_state = None
        self.steps_remaining = None
        self.is_infectious = None
        # TODO: llamar método de estado inicial
        self.initialize_person()

        # la persona conoce la region a la que pertenece
        self.region = region
        self.age = age
        self.health_conditions = None

    def initialize_person(self):
        self.next_state = StatePerson.S
        self.steps_remaining = 0

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
                state, time = self.p_latent_sintoms()
                self.next_state = state
                self.steps_remaining = time
            elif self.state == StatePerson.Lp:
                state, time = self.p_latent()
                self.next_state = state
                self.steps_remaining = time
            elif self.state == StatePerson.Ip:
                state, time = self.p_infect()
                self.next_state = state
                self.steps_remaining = time
            elif self.state == StatePerson.Is:
                state, time = self.p_infect_sitoms()
                self.next_state = state
                self.steps_remaining = time
            elif self.state == StatePerson.Iv:
                state, time = self.p_infect_sintom_antiviral()
                self.next_state = state
                self.steps_remaining = time
            elif self.state == StatePerson.A:
                state, time = self.p_asintomatic()
                self.next_state = state
                self.steps_remaining = time
            else:
                return False
            # en los estados restantes no hay transiciones
        else:
            # decrementar los steps que faltan para cambiar de estado
            self.steps_remaining = self.steps_remaining - 1
        
        return True

    # Funciones de que pasa en cada estado para cada persona
    # debe devolver el tiempo que la persona va estar en ese estado y 
    # a que estado le toca pasar en el futuro.

    def _evaluate_transition(self):
        """Computa a qué estado pasar dado el estado actual y los valores de la tabla.
        """
        df = load_disease_transition()
        # calcular el grupo de edad al que pertenece
        age_group = min(df['Age'], key=lambda age: age >= self.age)
        # quedarse con los valores del estado y edad del individuo actual
        df = df[(df['Age'] == age_group) & (df["StateFrom"] == str(self.state))]
        # calcular el estado de transición y el tiempo
        to_state = random.choices(df['StateTo'].values, weights=df['Chance'].values, k=1)[0]
        state_data = df.set_index('StateTo').to_dict('index')[to_state]
        time = random.normalvariate(state_data['MeanDays'], state_data['StdDays'])

        return to_state, int(time)

    def p_suseptible(self):
        self.is_infectious = False
        return self._evaluate_transition()

    def p_latent_sintoms(self):
        self.is_infectious = True
        return self._evaluate_transition()

    def p_latent(self):
        self.is_infectious = True
        return self._evaluate_transition()

    def p_infect(self):
        self.is_infectious = True
        return self._evaluate_transition()

    def p_infect_sitoms(self):
        self.is_infectious = True
        return self._evaluate_transition()

    def p_infect_sintom_antiviral(self):
        self.is_infectious = True
        return self._evaluate_transition()

    def p_asintomatic(self):
        self.is_infectious = True
        return self._evaluate_transition()

    def p_recovered(self):
        self.is_infectious = False
        return self._evaluate_transition()

    def p_hospitalized(self):
        self.is_infectious = True
        return self._evaluate_transition()

    def p_death(self):
        self.is_infectious = False
        return self._evaluate_transition()


class Region:
    def __init__(self, population, initial_infected=1):
        self._recovered = 0
        self._population = population
        self._death = 0
        self._simulations = 0
        self.individuals = []

        for i in range(initial_infected):
            p = Person(self, 30)
            p.state = StatePerson.Lp
            self.individuals.append(p)

    @property
    def population(self):
        return self._population

    @property
    def recovered(self):
        return self._recovered

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
    

@st.cache
def load_interaction_estimates():
    # no funciona, error "Could not find a version that satisfies the requirement xldr"
    df = pd.read_excel(
        "/src/data/contact_matrices_152_countries/MUestimates_all_locations_1.xlsx",
        sheet_name=None,
    )

    return df


def main():
    st.title("Simulación de la epidemia")

    region = Region(1000, 1)

    spatial_transmision([region], None, None, None, None)

    st.write("---")
    st.button("Ejecutar de nuevo")


if __name__ == "__main__":
    main()
