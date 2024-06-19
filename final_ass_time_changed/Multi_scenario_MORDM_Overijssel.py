# Import general python packages
import pandas as pd
import copy

# Import functions
from dike_model_function import DikeNetwork  # @UnresolvedImport
from problem_formulation import get_model_for_problem_formulation
from problem_formulation import (sum_over, time_step_0,time_step_1,
                                 time_step_2, time_step_3, time_step_4)
# Loading in the necessary modules for EMA workbench and functions
from ema_workbench import (Model, MultiprocessingEvaluator, Scenario,
                           Constraint, ScalarOutcome)
from ema_workbench.util import ema_logging
from ema_workbench import Policy
from ema_workbench import save_results, load_results
from ema_workbench.em_framework.optimization import ArchiveLogger, EpsilonProgress
from ema_workbench.em_framework.optimization import epsilon_nondominated, to_problem

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


    def create_scen(values, id):
        scen = {}
        for key in dike_model.uncertainties:
            name_split = key.name.split('_')
            if len(name_split) == 1:
                scen.update({key.name: values[key.name]})
            else:
                scen.update({key.name: values[name_split[1]]})
        return Scenario(f"scen_{id}", **scen)

    ref_scenario = Scenario('reference', **refcase_scen)
    print(ref_scenario)
    values = reference_values
    ref_scenario_test = create_scen(values, 'test')
    print(ref_scenario_test)


    ######### Overijssel
    model = problem_formulation_actor(7, uncertainties, levers)

    # Deepcopying the uncertainties and levers
    uncertainties = copy.deepcopy(dike_model.uncertainties)
    levers = copy.deepcopy(dike_model.levers)

    # Running the optimization for Overijssel
    function = DikeNetwork()
    convergence_metrics = {EpsilonProgress()}

    ###### HERE entering the scenario selection

    #GOOD1 scenario
    s1_values = { "Bmax": 100,
        "Brate": 1.0,
        "pfail": 0.1,
        "ID flood wave shape": 114,
        "planning steps": 5,
        "discount rate 0": 3.5,
        "discount rate 1": 3.5,
        "discount rate 2": 4.5,
        "discount rate 3": 4.5,
        "discount rate 4": 4.5}

    # BAD2
    s3_values = {"Bmax": 200,
                 "Brate": 2,
                 "pfail": 0.9,
                 "ID flood wave shape": 123,
                 "planning steps": 5,
                 "discount rate 0": 2.5,
                 "discount rate 1": 1.5,
                 "discount rate 2": 2.5,
                 "discount rate 3": 1.5,
                 "discount rate 4": 2.5}

    scenarios = [ref_scenario, create_scen(s3_values, 'Bad1'), create_scen(s1_values, 'Good1')]

    constraint = [Constraint("Total Costs", outcome_names='Total Costs', function=lambda x: max(0, x - 500000000))]


    def optimize(scenario, nfe, model, epsilons, constraint2):
        results = []
        convergences = pd.DataFrame()
        problem = to_problem(model, searchover="levers")

        with MultiprocessingEvaluator(model) as evaluator:
            for i in range(3):
                convergence_metrics = [
                    ArchiveLogger(
                        "./archives",
                        [l.name for l in model.levers],
                        [o.name for o in model.outcomes],
                        base_filename=f"Mutli_MORDM_{scenario.name}_seed_{i}.tar.gz",
                    ),
                    EpsilonProgress(),
                ]


                (result, convergence) = evaluator.optimize(nfe= nfe, searchover='levers',
                                                         convergence=convergence_metrics,
                                                         epsilons=  [1,1,1,1,1,0.1],
                                                         reference=scenario, constraints=constraint2)

                results.append(result)
                convergences = pd.concat([convergences, convergence])

        # merge the results using a non-dominated sort
        refer_set = epsilon_nondominated(results, epsilons, problem)
        reference_set = refer_set.loc[~refer_set.iloc[:, 1:51].duplicated()]

        return reference_set, convergences


    results_epsilon = pd.DataFrame()  # Initialize an empty DataFrame
    results_outcomes = pd.DataFrame()

    for scenario in scenarios:
        epsilons =  [1,1,1,1,1,0.1]

        # note that 100000 nfe is again rather low to ensure proper convergence
        resul = optimize(scenario, 25000, model, epsilons, constraint)

        y, t = resul

        # epsilon df
        results_epsilon = pd.concat([results_epsilon, t])

        #outcomes df
        results_outcomes = pd.concat([results_outcomes, y])

    robust_results = results_outcomes[results_outcomes.iloc[:, 1:51].duplicated(keep=False)]

    results_epsilon.to_csv('Overijssel_Multi_MORDM_epsilon.csv', index=False)
    results_outcomes.to_csv("Overijssel_Multi_MORDM_outcomes.csv", index=False)
    robust_results.to_csv("Overijssel_Multi_MORDM_outcomes_robust.csv", index=False)

    ### Overijssel Exploration
    policies = robust_results.iloc[:,1:51]

    rcase_policies = []

    for i, policy in policies.iterrows():
        rcase_policies.append(Policy(str(i), **policy.to_dict()))

    n_scenarios = 2000
    with MultiprocessingEvaluator(model) as evaluator:
        reference_policies_results = evaluator.perform_experiments(n_scenarios,
                                                rcase_policies)
    save_results(reference_policies_results, 'Overijssel_Multi_MORDM_outcomes_explr.tar.gz')