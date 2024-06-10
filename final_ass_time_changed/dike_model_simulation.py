from ema_workbench import Model, MultiprocessingEvaluator, Policy, Scenario

from ema_workbench.em_framework.evaluators import perform_experiments
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging
from ema_workbench import Samplers
import time
from ema_workbench import save_results
from problem_formulation import get_model_for_problem_formulation


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    dike_model, planning_steps = get_model_for_problem_formulation(6)

    # Build a user-defined scenario and policy:
    reference_values = {
        "Bmax": 175,
        "Brate": 1.5,
        # set pfail to 1 for model validation (getting out stochasticity)
        "pfail": 0.5,
        "ID flood wave shape": 4,
        "planning steps": 4,
    }
    reference_values.update({f"discount rate {n}": 3.5 for n in planning_steps})
    scen1 = {}

    for key in dike_model.uncertainties:
        name_split = key.name.split("_")

        if len(name_split) == 1:
            scen1.update({key.name: reference_values[key.name]})

        else:
            scen1.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario("reference", **scen1)

    # no dike increase, no warning, none of the rfr
    zero_policy = {"DaysToThreat": 0}
    zero_policy.update({f"DikeIncrease {n}": 0 for n in planning_steps})
    zero_policy.update({f"RfR {n}": 0 for n in planning_steps})

    # to validate model changes, check no difference but with one policy
    #zero_policy.update({f"DikeIncrease {2}": 2})

    pol0 = {}

    for key in dike_model.levers:
        s1, s2 = key.name.split("_")
        pol0.update({key.name: zero_policy[s2]})

    policy0 = Policy("Policy 0", **pol0)
    # Call random scenarios or policies:
    #    n_scenarios = 5
    #    scenarios = sample_uncertainties(dike_model, 50)
    #    n_policies = 10

    #VALIDATION RESULT T5
    # #single run
    # start = time.time()
    # dike_model.run_model(ref_scenario, policy0)
    # end = time.time()
    # print(end - start)
    # results = dike_model.outcomes_output
    # print(results)
    # print(ref_scenario)
    # print(policy0)
    #
    # {'A.1_Expected Annual Damage': [0.0, 0.0, 0.0, 0.0, 0.0],
    #  'A.1_Expected Number of Deaths': [0.0, 0.0, 0.0, 0.0, 0.0],
    #  'A.2_Expected Annual Damage': [0.0, 0.0, 0.0, 0.0, 0.0],
    #  'A.2_Expected Number of Deaths': [0.0, 0.0, 0.0, 0.0, 0.0],
    #  'A.3_Expected Annual Damage': [0.0, 0.0, 0.0, 0.0, 0.0],
    #  'A.3_Expected Number of Deaths': [0.0, 0.0, 0.0, 0.0, 0.0],
    #  'A.4_Expected Annual Damage': [0.0, 0.0, 0.0, 0.0, 0.0],
    #  'A.4_Expected Number of Deaths': [0.0, 0.0, 0.0, 0.0, 0.0],
    #  'A.5_Expected Annual Damage': [0.0, 0.0, 0.0, 0.0, 0.0],
    #  'A.5_Expected Number of Deaths': [0.0, 0.0, 0.0, 0.0, 0.0], 'All Costs': 137151747.29458064}
    # Scenario({'discount rate 0': 3.5, 'discount rate 1': 3.5, 'discount rate 2': 3.5, 'discount rate 3': 3.5,
    #           'discount rate 4': 3.5, 'A.0_ID flood wave shape': 4, 'A.1_Bmax': 175, 'A.1_pfail': 1, 'A.1_Brate': 1.5,
    #           'A.2_Bmax': 175, 'A.2_pfail': 1, 'A.2_Brate': 1.5, 'A.3_Bmax': 175, 'A.3_pfail': 1, 'A.3_Brate': 1.5,
    #           'A.4_Bmax': 175, 'A.4_pfail': 1, 'A.4_Brate': 1.5, 'A.5_Bmax': 175, 'A.5_pfail': 1, 'A.5_Brate': 1.5})
    # Policy(
    #     {'0_RfR 0': 0, '0_RfR 1': 0, '0_RfR 2': 0, '0_RfR 3': 0, '0_RfR 4': 0, '1_RfR 0': 0, '1_RfR 1': 0, '1_RfR 2': 0,
    #      '1_RfR 3': 0, '1_RfR 4': 0, '2_RfR 0': 0, '2_RfR 1': 0, '2_RfR 2': 0, '2_RfR 3': 0, '2_RfR 4': 0, '3_RfR 0': 0,
    #      '3_RfR 1': 0, '3_RfR 2': 0, '3_RfR 3': 0, '3_RfR 4': 0, '4_RfR 0': 0, '4_RfR 1': 0, '4_RfR 2': 0, '4_RfR 3': 0,
    #      '4_RfR 4': 0, 'EWS_DaysToThreat': 0, 'A.1_DikeIncrease 0': 0, 'A.1_DikeIncrease 1': 0, 'A.1_DikeIncrease 2': 2,
    #      'A.1_DikeIncrease 3': 0, 'A.1_DikeIncrease 4': 0, 'A.2_DikeIncrease 0': 0, 'A.2_DikeIncrease 1': 0,
    #      'A.2_DikeIncrease 2': 2, 'A.2_DikeIncrease 3': 0, 'A.2_DikeIncrease 4': 0, 'A.3_DikeIncrease 0': 0,
    #      'A.3_DikeIncrease 1': 0, 'A.3_DikeIncrease 2': 2, 'A.3_DikeIncrease 3': 0, 'A.3_DikeIncrease 4': 0,
    #      'A.4_DikeIncrease 0': 0, 'A.4_DikeIncrease 1': 0, 'A.4_DikeIncrease 2': 2, 'A.4_DikeIncrease 3': 0,
    #      'A.4_DikeIncrease 4': 0, 'A.5_DikeIncrease 0': 0, 'A.5_DikeIncrease 1': 0, 'A.5_DikeIncrease 2': 2,
    #      'A.5_DikeIncrease 3': 0, 'A.5_DikeIncrease 4': 0})

    # series run
    #
    # #Perform experiments without policies
    # results = perform_experiments(dike_model, scenarios=100, policies = 10)
    #
    # experiments, outcomes = results


    # from ema_workbench import MultiprocessingEvaluator, ema_logging
    # ema_logging.log_to_stderr(ema_logging.INFO)
    #
    # with MultiprocessingEvaluator(dike_model) as evaluator:
    #      results = evaluator.perform_experiments(scenarios= 1000, policies=20)

    #experiments_sobol, outcomes_sobol = results_sobol

    save_results(results, 'Experiments/W24_validate5T_5t.gz')
    #save_results(results, 'Experiments/W24_Open_Exploration_5t_10_3_PD6.tar.gz')
    #save_results(results_sobol, 'Experiments/Week22_Open_exploration_Sobol_1000_noP.tar.gz')


