# Import general python packages
import pandas as pd
import copy


# Import functions
from dike_model_function import DikeNetwork  # @UnresolvedImport
from problem_formulation import get_model_for_problem_formulation
from problem_formulation import sum_over,time_step_0,time_step_1, time_step_2, time_step_3, time_step_4


# Loading in the necessary modules for EMA workbench and functions
from ema_workbench import (Model, MultiprocessingEvaluator, Scenario,
                           Constraint, ScalarOutcome, TimeSeriesOutcome, ArrayOutcome)
from ema_workbench.util import ema_logging
from ema_workbench import save_results, Policy
from ema_workbench.em_framework.optimization import (EpsilonProgress)
from ema_workbench.analysis import parcoords


def initialize_model():
    ema_logging.log_to_stderr(ema_logging.INFO)
    print("Initializing model...")
    dike_model, planning_steps = get_model_for_problem_formulation(7)
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

    if problem_formulation_actor == 6:  # GELDERLAND
        model.outcomes.clear()
        model.outcomes = [
            ScalarOutcome(f'Total_period_Costs_0',
                          variable_name=dike_model.outcomes['Total_period_Costs'].variable_name,
                          function=time_step_0, kind=direction),
            ScalarOutcome(f'Total_period_Costs_1',
                          variable_name=dike_model.outcomes['Total_period_Costs'].variable_name,
                          function=time_step_1, kind=direction),
            # ScalarOutcome(f'Total_period_Costs_2',
            #               variable_name=dike_model.outcomes['Total_period_Costs'].variable_name,
            #               function=time_step_2, kind=direction),
            # ScalarOutcome(f'Total_period_Costs_3',
            #               variable_name=dike_model.outcomes['Total_period_Costs'].variable_name,
            #               function=time_step_3, kind=direction),
            # ScalarOutcome(f'Total_period_Costs_4',
            #               variable_name=dike_model.outcomes['Total_period_Costs'].variable_name,
            #               function=time_step_4, kind=direction),
            ScalarOutcome('Expected Annual Damage A1_', variable_name='A.1_Expected Annual Damage', function=sum_over,
                          kind=direction),
            ScalarOutcome('Expected Annual Damage A2_', variable_name='A.2_Expected Annual Damage', function=sum_over,
                          kind=direction),
            ScalarOutcome('Expected Annual Damage A3_', variable_name='A.3_Expected Annual Damage', function=sum_over,
                          kind=direction),
            ScalarOutcome('Total Costs', variable_name=cost_variables, function=sum_over, kind=direction),
            ScalarOutcome("Expected Number of Deaths_", variable_name=
            [f"{dike}_Expected Number of Deaths" for dike in function.dikelist], function=sum_over, kind=direction)]


    elif problem_formulation_actor == 7:  # OVERIJSSEL
        model.outcomes.clear()
        model.outcomes = [
            ScalarOutcome(f'Total_period_Costs_0',
                          variable_name=dike_model.outcomes['Total_period_Costs'].variable_name,
                          function=time_step_0, kind=direction),
            ScalarOutcome(f'Total_period_Costs_1',
                          variable_name=dike_model.outcomes['Total_period_Costs'].variable_name,
                          function=time_step_1, kind=direction),
            # ScalarOutcome(f'Total_period_Costs_2',
            #               variable_name=dike_model.outcomes['Total_period_Costs'].variable_name,
            #               function=time_step_2, kind=direction),
            # ScalarOutcome(f'Total_period_Costs_3',
            #               variable_name=dike_model.outcomes['Total_period_Costs'].variable_name,
            #               function=time_step_3, kind=direction),
            # # ScalarOutcome(f'Total_period_Costs_4',
            #               variable_name=dike_model.outcomes['Total_period_Costs'].variable_name,
            #               function=time_step_4, kind=direction),
            ScalarOutcome('Expected Annual Damage A4_', variable_name='A.4_Expected Annual Damage', function=sum_over,
                          kind=direction),
            ScalarOutcome('Expected Annual Damage A5_', variable_name='A.5_Expected Annual Damage', function=sum_over,
                          kind=direction),
            ScalarOutcome('Total Costs', variable_name=cost_variables, function=sum_over, kind=direction),
            ScalarOutcome("Expected Number of Deaths_", variable_name=
            [f"{dike}_Expected Number of Deaths" for dike in function.dikelist], function=sum_over, kind=direction)]

    else:
        raise TypeError('unknown identifier')
    return model


def divide_into_segments(df, objective_column, n_segments=3):
    """Divides the DataFrame into equal segments based on the objective column."""
    df_sorted = df.sort_values(by=objective_column)
    segment_length = len(df_sorted) // n_segments
    selected_indices = [segment_length * i + segment_length // 2 for i in range(n_segments)]
    return df_sorted.iloc[selected_indices]

if __name__ == '__main__':
    # Initialize model and planning steps
    dike_model, planning_steps = initialize_model()

    # Deepcopying the uncertainties and levers to avoid side-effects
    uncertainties = copy.deepcopy(dike_model.uncertainties)
    levers = copy.deepcopy(dike_model.levers)

    # Defining the model
    model = problem_formulation_actor(7, uncertainties, levers)

    # Initialize the function for dike network
    function = DikeNetwork()

    # Load policy set data
    policy_set = pd.read_csv("./Outcomes/Overijssel Multi MORDM_Policies.csv")

    # Select policies based on extreme values in specific columns
    # Select policies based on extreme values in specific columns
    policy_snip = [
                      policy_set.iloc[:, -i].idxmin() for i in range(2, 8)
                  ] + [
                      policy_set.iloc[:, -i].idxmax() for i in range(2, 8)
                  ]

    # Selecting columns with objectives
    objective_columns = policy_set.columns[-7:-1]

    # Select one solution from each segment for each objective
    selected_policies = pd.DataFrame()
    for objective in objective_columns:
        segment_policies = divide_into_segments(policy_set, objective)
        selected_policies = pd.concat([selected_policies, segment_policies])

    # Combine and ensure unique policies
    policy_snip2 = selected_policies.index.tolist()
    total_snip = policy_snip + policy_snip2
    unique_snip = list(set(total_snip))

    # Retrieve policies based on unique indices
    policies = policy_set.loc[unique_snip]
    policies = policies.iloc[:, 1:51]

    # Convert policies to Policy instances
    rcase_policies = []
    for i, policy in policies.iterrows():
        rcase_policies.append(Policy(str(i), **policy.to_dict()))

    # Run the experiment with the specified number of scenarios
    n_scenarios = 1000
    with MultiprocessingEvaluator(model) as evaluator:
        reference_policies_results = evaluator.perform_experiments(n_scenarios, rcase_policies)

    # Save the results
    save_results(reference_policies_results, 'Visualisations/data/MultiMORDM_Overijssel_big.tar.gz')
