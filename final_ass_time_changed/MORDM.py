# Import general python packages
import pandas as pd
import numpy as np
import seaborn as sns
import copy

# Import functions
from dike_model_function import DikeNetwork  # @UnresolvedImport
from problem_formulation import get_model_for_problem_formulation
from problem_formulation import sum_over, sum_over_time

# Loading in the necessary modules for EMA workbench and functions
from ema_workbench import (Model, MultiprocessingEvaluator, Scenario,
                           Constraint, ScalarOutcome)
from ema_workbench.util import ema_logging
from ema_workbench import save_results, load_results
from ema_workbench.em_framework.optimization import (EpsilonProgress)

def initialize_model():
    ema_logging.log_to_stderr(ema_logging.INFO)
    print("Initializing model...")
    dike_model, planning_steps = get_model_for_problem_formulation(6)
    print("Model initialized.")
    return dike_model, planning_steps

# Writing a function to create actor specific problem formulations
def problem_formulation_actor(problem_formulation_actor, uncertainties, levers):
    # Load the model:
    function = DikeNetwork()
    # workbench model:
    model = Model('dikesnet', function=function)
    # Outcomes are all costs, thus they have to minimized:
    direction = ScalarOutcome.MINIMIZE

    model.uncertainties = uncertainties
    model.levers = levers

    cost_variables = []
    cost_variables.extend(
        [
            f"{dike}_{e}"
            for e in ["Expected Annual Damage", "Dike Investment Costs"]
            for dike in function.dikelist
        ])
    cost_variables.extend([f"RfR Total Costs"])
    cost_variables.extend([f"Expected Evacuation Costs"])

    if problem_formulation_actor == 4:  # RWS
        model.outcomes.clear()
        model.outcomes = [
            ScalarOutcome('Expected Annual Damage',
                          variable_name=['{}_Expected Annual Damage'.format(dike)
                                         for dike in function.dikelist],
                          function=sum_over, kind=direction),

            ScalarOutcome('Total Investment Costs',
                          variable_name=['{}_Dike Investment Costs'.format(dike)
                                         for dike in function.dikelist] + ['RfR Total Costs'
                                                                           ] + ['Expected Evacuation Costs'],
                          function=sum_over, kind=direction),

            ScalarOutcome('Expected Number of Deaths',
                          variable_name=['{}_Expected Number of Deaths'.format(dike)
                                         for dike in function.dikelist],
                          function=sum_over, kind=direction)]

    elif problem_formulation_actor == 5:  # GELDERLAND
        model.outcomes.clear()
        model.outcomes = [
            ScalarOutcome('Expected Annual Damage A1', variable_name='A.1_Expected Annual Damage', function=sum_over,
                          kind=direction),
            ScalarOutcome('Expected Annual Damage A2', variable_name='A.2_Expected Annual Damage', function=sum_over,
                          kind=direction),
            ScalarOutcome('Expected Annual Damage A3', variable_name='A.3_Expected Annual Damage', function=sum_over,
                          kind=direction),
            ScalarOutcome('Expected Number of Deaths in A1', variable_name='A.1_Expected Number of Deaths',
                          function=sum_over, kind=direction),
            ScalarOutcome('Expected Number of Deaths in A2', variable_name='A.2_Expected Number of Deaths',
                          function=sum_over, kind=direction),
            ScalarOutcome('Expected Number of Deaths in A3', variable_name='A.3_Expected Number of Deaths',
                          function=sum_over, kind=direction),
            ScalarOutcome('Total Costs', variable_name=cost_variables, function=sum_over, kind=direction),
            ScalarOutcome('Aggregated Expected Number of Deaths A4-A5', variable_name=
            ['A.4_Expected Number of Deaths', 'A.5_Expected Number of Deaths'], function=sum_over, kind=direction)]

    elif problem_formulation_actor == 6:  # OVERIJSSEL
        model.outcomes.clear()
        model.outcomes = [
            ScalarOutcome('Expected Annual Damage A4', variable_name='A.4_Expected Annual Damage', function=sum_over,
                          kind=direction),
            ScalarOutcome('Expected Annual Damage A5', variable_name='A.5_Expected Annual Damage', function=sum_over,
                          kind=direction),
            ScalarOutcome('Expected Number of Deaths in A4', variable_name='A.4_Expected Number of Deaths',
                          function=sum_over, kind=direction),
            ScalarOutcome('Expected Number of Deaths in A5', variable_name='A.5_Expected Number of Deaths',
                          function=sum_over, kind=direction),
            ScalarOutcome('Total Costs', variable_name=cost_variables, function=sum_over, kind=direction),
            ScalarOutcome('Aggregated Expected Number of Deaths A1-A3', variable_name=
            ['A.1_Expected Number of Deaths', 'A.2_Expected Number of Deaths',
             'A.3_Expected Number of Deaths'], function=sum_over, kind=direction)]

    else:
        raise TypeError('unknown identifier')
    return model

### Overijssel
if __name__ == '__main__':
    dike_model, planning_steps = initialize_model()

    uncertainties = dike_model.uncertainties
    levers = dike_model.levers

    # Setting the reference scenario
    reference_values = {
        "Bmax": 175,
        "Brate": 1.5,
        "pfail": 0.5,
        "ID flood wave shape": 4,
        "planning steps": 2,
    }
    reference_values.update({f"discount rate {n}": 3.5 for n in planning_steps})
    refcase_scen = {}

    for key in dike_model.uncertainties:
        name_split = key.name.split('_')
        if len(name_split) == 1:
            refcase_scen.update({key.name: reference_values[key.name]})
        else:
            refcase_scen.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario('reference', **refcase_scen)

    ######### Overijssel
    model = problem_formulation_actor(6, uncertainties, levers)

    # Deepcopying the uncertainties and levers
    uncertainties = copy.deepcopy(dike_model.uncertainties)
    levers = copy.deepcopy(dike_model.levers)

    # Running the optimization for Overijssel
    convergence_metrics = {EpsilonProgress()}
    constraint = [Constraint("Total Costs", outcome_names="Total Costs", function=lambda x: max(0, x - 700000000))]

    results_epsilon = pd.DataFrame()  # Initialize an empty DataFrame
    results_outcomes = pd.DataFrame()
    with MultiprocessingEvaluator(model) as evaluator:
        for _ in range(2):
            (y, t) = evaluator.optimize(nfe=10, searchover='levers',
                                        convergence=convergence_metrics,
                                        epsilons=[0.1] * len(model.outcomes), reference=ref_scenario,
                                        constraints=constraint)

            results_epsilon = pd.concat([results_epsilon, t])
            results_outcomes = pd.concat([results_outcomes, y])
            print(results_outcomes)


    # Save the concatenated DataFrame to a CSV file
    results_epsilon.to_csv('Week24_MORDM_epsilon_overijssel_PD6.csv', index=False)
    results_outcomes.to_csv('Week24_MORDM_outcomes_overijssel_PD6.csv', index=False)
    print("Optimization results saved.")

# ######### Gelderland
if __name__ == '__main__':
    dike_model, planning_steps = initialize_model()

    uncertainties = dike_model.uncertainties
    levers = dike_model.levers

    # Setting the reference scenario
    reference_values = {
        "Bmax": 175,
        "Brate": 1.5,
        "pfail": 0.5,
        "ID flood wave shape": 4,
        "planning steps": 2,
    }
    reference_values.update({f"discount rate {n}": 3.5 for n in planning_steps})
    refcase_scen = {}

    for key in dike_model.uncertainties:
        name_split = key.name.split('_')
        if len(name_split) == 1:
            refcase_scen.update({key.name: reference_values[key.name]})
        else:
            refcase_scen.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario('reference', **refcase_scen)

    ######### Overijssel
    model2 = problem_formulation_actor(5, uncertainties, levers)

    convergence_metrics = {EpsilonProgress()}
    constraint = [Constraint("Total Costs", outcome_names="Total Costs", function=lambda x: max(0, x - 700000000))]

    results_epsilon2 = pd.DataFrame()  # Initialize an empty DataFrame
    results_outcomes2 = pd.DataFrame()
    with MultiprocessingEvaluator(model2) as evaluator:
        for _ in range(2):
            (y, t) = evaluator.optimize(nfe=10, searchover='levers',
                                        convergence=convergence_metrics,
                                        epsilons=[0.1] * len(model2.outcomes), reference=ref_scenario,
                                        constraints=constraint)

            results_epsilon2 = pd.concat([results_epsilon2, t])
            results_outcomes2 = pd.concat([results_outcomes2, y])
            print(results_outcomes2)

    # Save the concatenated DataFrame to a CSV file
    results_epsilon2.to_csv('Week23_MORDM_Gelderland_PD5.csv', index=False)
    results_outcomes2.to_csv('Week23_MORDM_Gelderland_PD5.csv', index=False)

