


policy_set = pd.read_csv("Gelderland_Multi_MORDM_outcomes.csv")

    # add column indicating under which scenario which policy was found
    policy_snip = []
    for i in range(1, 8):
        policy_snip.append(policy_set.iloc[:, -i].idxmin())
        policy_snip.append(policy_set.iloc[:, -i].idxmax())

    pareto_df = policy_set

    # Selecteer de laatste 7 kolommen (objectieven)
    objective_columns = pareto_df.columns[-7:]

    # Verdeel elke doelstelling in 3 segmenten en selecteer één oplossing uit elk segment
    selected_policies = pd.DataFrame()

    for objective in objective_columns:
        # Sorteer de Pareto-set op de huidige doelstelling
        pareto_df_sorted = pareto_df.sort_values(by=objective)

        # Verdeel de Pareto-set in 3 gelijke segmenten
        indices = np.linspace(0, len(pareto_df_sorted) - 1, 4, dtype=int)

        # Omzetten naar een enkele lijst van indices
        selected_indices = (indices[:-1] + np.diff(indices) // 2).tolist()

        # Selecteer de rijen met de geselecteerde indices en voeg toe aan selected_policies
        selected_policies = pd.concat([selected_policies, pareto_df_sorted.iloc[selected_indices]])

    policy_snip2 = selected_policies.index.tolist()

    total_snip = policy_snip + policy_snip2
    #len(total_snip)  # 35 = 14+21 dus klopt
    unique_snip = list(set(total_snip))

    policies = policy_set.loc[unique_snip]