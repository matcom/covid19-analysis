import collections
import random
from typing import Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


PARAMETERS = dict()


class InterventionsManager:
    def __init__(self):
        self._closed_borders = []
        self._testing = []
        self._school_open = []
        self._workforce = []
        self._social_distance = []
        self._use_masks = []
        self._reduce_aglomeration = []
        self._restriction_internal_movement = []
        self.day = 0

    def close_borders(self, start, end):
        """ Activa la medida de cerrar los aeropuertos
        """
        self._closed_borders.append((start, end))

    def is_airport_open(self):
        """ Informa si los aeropuertos están cerrados
        """
        for start, end in self._closed_borders:
            if self.day >= start and self.day <= end:
                return False

        return True

    def activate_testing(self, start, end, percent):
        """ Activa la medida de testiar un % de la población
        """
        self._testing.append((start, end, percent))

    def is_testing_active(self):
        """ Informa si la medida de testeo de personas está activa
        """
       
        for start, end, percent in self._testing:
            if self.day >= start and self.day <= end:
                return percent

        return 0.0

    def school_close(self, start, end):
        """ Activa la medida de cerrar las escuelas
        """
        self._school_open.append((start, end))

    def is_school_open(self):
        """ Informa si la medida de cerrar escuelasestá activa
        """
       
        for start, end in self._school_open:
            if self.day >= start and self.day <= end:
                return False

        return True

    def activate_workforce(self, start, end, percent):
        """ Activa la medida de activar el teletrabajo en un % de la población
        """
        self._workforce.append((start, end, percent))

    def is_workforce(self):
        """ Informa si la medida de activar el teletrabajo en un % de la población
        """
       
        for start, end, percent in self._workforce:
            if self.day >= start and self.day <= end:
                return percent

        return 1.0 

    def activate_social_distance(self, start, end, percent):
        """ Activa la medida de activar el distanciamiento social
        """
        self._social_distance.append((start, end, percent))

    def is_social_distance(self):
        """ Informa si la medida de activar el distanciamiento social
        """
       
        for start, end, percent in self._social_distance:
            if self.day >= start and self.day <= end:
                return percent

        return 1.0 

    def activate_use_masks(self, start, end):
        """ Activa la medida de utilizar mascarillas
        """
        self._use_masks.append((start, end))

    def is_use_masks(self):
        """ Informa si la medida de utilizar mascarillas
        """
       
        for start, end in self._use_masks:
            if self.day >= start and self.day <= end:
                return True

        return False

    def activate_reduce_aglomeration(self, start, end, percent):
        """ Activa la medida de no participar en aglomeraciones públicas
        """
        self._reduce_aglomeration.append((start, end, percent))

    def is_reduce_aglomeration(self):
        """ Informa si la medida de no participar en aglomeraciones públicas
        """
       
        for start, end, percent in self._reduce_aglomeration:
            if self.day >= start and self.day <= end:
                return percent

        return 1.0

    def activate_restriction_internal_movement(self, start, end, percent):
        """ Activa la medida de restricciones en el transporte interno
        """
        self._restriction_internal_movement.append((start, end, percent))

    def is_restriction_internal_movement(self):
        """ Informa si la medida de restricciones en el transporte interno
        """
       
        for start, end, percent in self._restriction_internal_movement:
            if self.day >= start and self.day <= end:
                return percent

        return 1.0

Interventions = InterventionsManager()


@st.cache
def load_disease_transition():
    df = pd.read_csv("./data/disease_transitions_cuba.csv")
    data = collections.defaultdict(lambda: [])

    for i, row in df.iterrows():
        age = row['Age']
        sex = row['Sex']
        state_from = row['StateFrom']
        state_to = row['StateTo']

        data[(age, sex, state_from)].append(dict(
            state=state_to,
            count=row['Count'],
            mean=row['MeanDays'],
            std=row['StdDays'],
        ))

    return dict(data)


class TransitionEstimator:
    def __init__(self):
        self.data = load_disease_transition()

    def transition(self, from_state, age, sex):
        age = (age // 5)  * 5

        if (age, sex, from_state) not in self.data:
            raise ValueError(f"No transitions for {from_state}, age={age}, sex={sex}.")

        df = self.data[(age, sex, from_state)]
        return pd.DataFrame(df).sort_values("count")


    def _transition(self, from_state, age, sex):
        raise Exception("Old method. Do not use.")

        """Computa las probabilidades de transición del estado `from_state` para una 
        persona con edad `age` y sexo `sex.
        """

        # De todos los datos que tenemos, vamos a ordenarlos por diferencia absoluta
        # con la edad deseada, y vamos cogiendo filas hasta acumular al menos 50
        # mediciones.
        evidence = self.data[
            (self.data["StateFrom"] == from_state) & (self.data["Sex"] == sex)
        ].copy()
        evidence["AgeDiff"] = (evidence["Age"] - age).abs()
        evidence = evidence.sort_values(["AgeDiff", "Count"]).copy()
        evidence["CountCumul"] = evidence["Count"].cumsum()

        if evidence["CountCumul"].max() > 50:
            min_required_evidence = evidence[evidence["CountCumul"] >= 50][
                "CountCumul"
            ].min()
            evidence = evidence[evidence["CountCumul"] <= min_required_evidence]

        # Ahora volvemos a agrupar por estado y recalculamos la media y varianza
        state_to = collections.defaultdict(lambda: dict(count=0, mean=0, std=0))

        if len(evidence) == 0:
            raise ValueError(f"No transitions for {from_state}, age={age}, sex={sex}.")

        for i, row in evidence.iterrows():
            d = state_to[row["StateTo"]]
            d["count"] += row["Count"]
            d["mean"] += row["Count"] * row["MeanDays"]
            d["std"] += row["Count"] * row["StdDays"]
            d["state"] = row["StateTo"]

        for v in state_to.values():
            v["mean"] /= v["count"]
            v["std"] /= v["count"]

        # Finalmente tenemos un dataframe con forma
        # `count | mean | std | state`
        # con una entrada por cada estado
        # if not state_to:
        #     raise ValueError(f"No transitions for {from_state}, age={age}, sex={sex}.")

        return pd.DataFrame(state_to.values()).sort_values("count")


TRANSITIONS = TransitionEstimator()


@st.cache
def load_interaction_estimates(location):
    df: pd.DataFrame = pd.read_csv(
        "./data/contact_matrices_152_countries/nicaragua_%s.csv" % location,
        header=None,
        names=[i for i in range(5, 85, 5)],
    )
    df["age"] = [i for i in range(5, 85, 5)]

    return df.set_index("age").to_dict()


class StatePerson:
    """Estados en los que puede estar una persona.
    """

    S = "S"
    L = "L"
    I = "I"
    A = "A"
    U = "U"
    R = "R"
    H = "H"
    D = "D"
    F = "F"


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

    simulation_time = PARAMETERS['DAYS_TO_SIMULATE']

    # estadísticas de la simulación
    progress = st.progress(0)
    day = st.empty()
    sick_count = st.empty()
    all_count = st.empty()

    chart = st.altair_chart(
        alt.Chart(pd.DataFrame(columns=["value", "day", "variable"]))
        .mark_line()
        .encode(y="value:Q", x="day:Q", color="variable:N",),
        use_container_width=True,
    )

    # por cada paso de la simulación
    for i in range(simulation_time):

        total_individuals = 0
        by_state = collections.defaultdict(lambda: 0)
        Interventions.day = i+1

        # por cada región
        for region in regions:
            # llegadas del estranjero
            arrivals(region) 
            # por cada persona
            for ind in region:
                # actualizar estado de la persona
                ind.next_step()
                if ind.is_infectious:
                    compute_spread(ind, social, status)

                total_individuals += 1
                by_state[ind.state] += 1

            interventions(region, status)
            # movimientos
            for n_region in regions:
                if n_region != region:
                    # calcular personas que se mueven de una region a otras
                    transportations(n_region, region, distance)

        progress.progress((i + 1) / simulation_time)
        sick_count.markdown(f"#### Individuos simulados: {total_individuals}")
        all_count.code(dict(by_state))
        day.markdown(f"#### Día: {i+1}")

        chart.add_rows(
            [dict(day=i + 1, value=v, variable=k) for k, v in by_state.items()]
        )


def arrivals(region):
    if Interventions.is_airport_open():
        people = np.random.poisson(PARAMETERS['FOREIGNER_ARRIVALS'])

        for i in range(people):
            p = region.spawn(random.randint(20, 80))
            p.set_state(StatePerson.F)


def interventions(region, status):
    """Modifica el estado de las medidas y como influyen estas en la población.

       Los cambios se almacenan en status
    """
    # si se está testeando activamente
    p = Interventions.is_testing_active()

    if p > 0:
        for ind in region:
            if ind.state in [StatePerson.L, StatePerson.A] and random.uniform(0, 1) < p:
                ind.set_state(StatePerson.H)


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
            other.set_state(StatePerson.L)


def eval_connections(
    social: Dict[str, Dict[int, Dict[int, float]]], person: "Person"
) -> List["Person"]:
    """Devuelve las conexiones que tuvo una persona en un step de la simulación.
    """
 
    the_age = person.age

    if the_age % 5 != 0:
        the_age = (the_age // 5 * 5)

    # contactos en la calle
    other_ages = social['other'][the_age]

    for age, lam in other_ages.items():
        people = np.random.poisson(lam)

        for i in range(people):
            yield person.region.spawn(age)

    # contactos en la escuela
    if Interventions.is_school_open():
        other_ages = social['schools'][the_age]

        for age, lam in other_ages.items():
            people = np.random.poisson(lam)

            for i in range(people):
                yield person.region.spawn(age)

    # contactos en el trabajo
    if age < 18:
        return

    p_work = Interventions.is_workforce()

    if random.uniform(0, 1) < p_work:
        other_ages = social['work'][the_age]

        for age, lam in other_ages.items():
            people = np.random.poisson(lam * p_work)

            for i in range(people):
                yield person.region.spawn(age)



def eval_infections(person) -> bool:
    """Determina si una persona cualquiera se infesta o no, dado que se cruza con "person". 

       En general depende del estado en el que se encuentra person y las probabilidades de ese estado
    """
    return random.uniform(0, 1) < PARAMETERS['CHANCE_OF_INFECTION']


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

            if self.state == StatePerson.L:
                self.p_latent()
            elif self.state == StatePerson.I:
                self.p_infect()
            elif self.state == StatePerson.A:
                self.p_asintomatic()
            elif self.state == StatePerson.H:
                self.p_hospitalized()
            elif self.state == StatePerson.U:
                self.p_uci()
            elif self.state == StatePerson.F:
                self.p_foreigner()
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
        # calcular el estado de transición y el tiempo
        to_state = random.choices(df["state"].values, weights=df["count"].values, k=1)[
            0
        ]
        state_data = df.set_index("state").to_dict("index")[to_state]
        time = random.normalvariate(state_data["mean"], state_data["std"])

        return to_state, int(time)

    def p_latent(self):
        self.is_infectious = False
        change_to_symptoms = 0.5

        if random.uniform(0, 1) < change_to_symptoms:
            self.next_state = StatePerson.A
            self.steps_remaining = 0
            return

        # convertirse en sintomático en (2,14) dias
        self.next_state = StatePerson.I
        self.steps_remaining = random.randint(2, 14)

    def p_foreigner(self):
        self.is_infectious = True
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_infect(self):
        self.is_infectious = True
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_asintomatic(self):
        self.is_infectious = True
        self.next_state = StatePerson.R
        # tiempo en que un asintomático se cura
        self.steps_remaining = random.randint(2, 14)

    def p_recovered(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_hospitalized(self):
        self.is_infectious = False
        self.next_state, self.steps_remaining = self._evaluate_transition()

    def p_uci(self):
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
            p = Person(self, 30, random.choice(["MALE", "FEMALE"]))
            p.set_state(StatePerson.I)
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
        p = Person(self, age, random.choice(["MALE", "FEMALE"]))
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


def run():
    st.title("Simulación de la epidemia")

    PARAMETERS['DAYS_TO_SIMULATE'] = st.sidebar.number_input("Dias a simular", 1, 365, 60)
    PARAMETERS['START_INFECTED'] = st.sidebar.number_input("Cantidad inicial de infectados", 0, 100, 0)
    PARAMETERS['CHANCE_OF_INFECTION'] = st.sidebar.number_input(
        "Posibilidad de infectar", 0.0, 1.0, 0.1, step=0.001
    )
    PARAMETERS['FOREIGNER_ARRIVALS'] = st.sidebar.number_input(
        "Llegada diaria de extranjeros", 0.0, 100.0, 10.0, step=0.01
    )

    st.write("### Medidas")

    if st.checkbox("Cerrar fronteras"):
        start, end = st.slider("Rango de fechas de cierre de fronteras", 0, PARAMETERS['DAYS_TO_SIMULATE'], (10, PARAMETERS['DAYS_TO_SIMULATE']))
        Interventions.close_borders(start, end)

    if st.checkbox("Cerrar escuelas"):
        start, end = st.slider("Rango de fechas de cierre de escuelas", 0, PARAMETERS['DAYS_TO_SIMULATE'], (10, PARAMETERS['DAYS_TO_SIMULATE']))
        Interventions.school_close(start, end)

    if st.checkbox("Testing activo de contactos"):
        start, end = st.slider("Rango de fechas de testing activo", 0, PARAMETERS['DAYS_TO_SIMULATE'], (10, PARAMETERS['DAYS_TO_SIMULATE']))
        percent = st.slider("Porciento de contactos a testear", 0.0, 1.0, 0.5)
        Interventions.activate_testing(start, end, percent)

    if st.checkbox("Disminuir trabajo / trabajar desde la casa"):
        start, end = st.slider("Rango de fechas activo", 0, PARAMETERS['DAYS_TO_SIMULATE'], (10, PARAMETERS['DAYS_TO_SIMULATE']))
        percent = st.slider("Porciento de personas que dejan de trabajar", 0.0, 1.0, 0.5)
        Interventions.activate_workforce(start, end, percent)

    connections = dict(
        work=load_interaction_estimates("work"),
        schools=load_interaction_estimates("schools"),
        other=load_interaction_estimates("other_locations"),
    )

    if st.button("Simular"):
        region = Region(1000, PARAMETERS['START_INFECTED'])
        spatial_transmision([region], connections, None, None, None)
