import copy
import itertools
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def extract_grid_info(agent, env_copy, city, day, t):
    """Extract useful behavioural measures when yoking to human choices."""
    paths = env_copy.path_states[t].copy()
    obs_list = [tuple(obs[:2]) for obs in env_copy.obs.tolist()]
    obs_list = list(set(obs_list)) # no repeated obs!
    for i, path in enumerate(paths):
        
        try:

            ## get the number of states that overlap with the paths
            overlap = set(path).intersection(set(obs_list))
            path_past_overlap = len(overlap)

            ## get the number of costs and no-costs that comprise these overlapping states
            try:
                path_past_observed_high_costs = sum(env_copy.costs[obs[0], obs[1]] == env_copy.high_cost for obs in overlap)
                path_past_observed_low_costs = sum(env_copy.costs[obs[0], obs[1]] == env_copy.low_cost for obs in overlap)
            except:
                path_past_observed_high_costs = sum(env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost for obs in overlap)
                path_past_observed_low_costs = sum(env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost for obs in overlap)
            agent.path_future_overlaps[city, day, t, i] = env_copy.path_future_overlaps[t][i]
            agent.path_past_overlaps[city, day, t, i] = path_past_overlap
            agent.path_past_observed_high_costs[city, day, t, i] = path_past_observed_high_costs
            agent.path_past_observed_low_costs[city, day, t, i] = path_past_observed_low_costs



        ## sometimes need to convert each np array to list of tuples...
        except:
            # paths = [set(map(tuple, path)) for path in paths]
            path = set(map(tuple, path))
            overlap = set(path).intersection(set(obs_list))
            path_past_overlap = len(overlap)
            try:
                path_past_observed_high_costs = sum(env_copy.costs[obs[0], obs[1]] == env_copy.high_cost for obs in overlap)
                path_past_observed_low_costs = sum(env_copy.costs[obs[0], obs[1]] == env_copy.low_cost for obs in overlap)
            except:
                path_past_observed_high_costs = sum(env_copy.costss[t][int(obs[0]), int(obs[1])] == env_copy.high_cost for obs in overlap)
                path_past_observed_low_costs = sum(env_copy.costss[t][int(obs[0]), int(obs[1])] == env_copy.low_cost for obs in overlap)
            agent.path_future_overlaps[city, day, t, i] = env_copy.path_future_overlaps[t][i]
            agent.path_past_overlaps[city, day, t, i] = path_past_overlap
            agent.path_past_observed_high_costs[city, day, t, i] = path_past_observed_high_costs
            agent.path_past_observed_low_costs[city, day, t, i] = path_past_observed_low_costs

        assert agent.path_past_overlaps[city, day, t, i] == agent.path_past_observed_high_costs[city, day, t, i] + agent.path_past_observed_low_costs[city, day, t, i], 'path {} past overlap does not match observed costs and no-costs\n path past overlap: {}, path observed costs: {}, path observed no-costs: {}'.format(i+1, agent.path_past_overlaps[city, day, t, i], agent.path_past_observed_high_costs[city, day, t, i], agent.path_past_observed_low_costs[city, day, t, i])


        ## get aligned vs orthogonal states
        path_states = env_copy.path_states[t][i]
        aligned_states, orthogonal_states = env_copy.path_aligned_states[t][i], env_copy.path_orthogonal_states[t][i]
        agent.aligned_arm_len[city, day, t, i] = len(aligned_states)
        agent.orthogonal_arm_len[city, day, t, i] = len(orthogonal_states)

        ## get info on costs on rows and columns
        observed_high_cost_cols = {obs[1] for obs in obs_list if env_copy.costs[int(obs[0]), int(obs[1])] == env_copy.high_cost}
        observed_low_cost_cols = {obs[1] for obs in obs_list if  env_copy.costs[int(obs[0]), int(obs[1])] == env_copy.low_cost}
        observed_high_cost_rows = {obs[0] for obs in obs_list if env_copy.costs[int(obs[0]), int(obs[1])] == env_copy.high_cost}
        observed_low_cost_rows = {obs[0] for obs in obs_list if  env_copy.costs[int(obs[0]), int(obs[1])] == env_copy.low_cost}
        observed_high_cost_states = {tuple(obs) for obs in obs_list if env_copy.costs[int(obs[0]), int(obs[1])] == env_copy.high_cost}
        observed_low_cost_states = {tuple(obs) for obs in obs_list if env_copy.costs[int(obs[0]), int(obs[1])] == env_copy.low_cost}
        # try:
        #     observed_high_cost_cols = {obs[1] for obs in obs_list if env_copy.costs[obs[0], obs[1]] == env_copy.high_cost}
        #     observed_low_cost_cols = {obs[1] for obs in obs_list if env_copy.costs[obs[0], obs[1]] == env_copy.low_cost}
        #     observed_high_cost_rows = {obs[0] for obs in obs_list if env_copy.costs[obs[0], obs[1]] == env_copy.high_cost}
        #     observed_low_cost_rows = {obs[0] for obs in obs_list if env_copy.costs[obs[0], obs[1]] == env_copy.low_cost}
        #     observed_high_cost_states = {tuple(obs) for obs in obs_list if env_copy.costs[obs[0], obs[1]] == env_copy.high_cost}
        #     observed_low_cost_states = {tuple(obs) for obs in obs_list if env_copy.costs[obs[0], obs[1]] == env_copy.low_cost}
        # except:
        #     observed_high_cost_cols = {obs[1] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost}
        #     observed_low_cost_cols = {obs[1] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost}
        #     observed_high_cost_rows = {obs[0] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost}
        #     observed_low_cost_rows = {obs[0] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost}
        #     observed_high_cost_states = {tuple(obs) for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost}
        #     observed_low_cost_states = {tuple(obs) for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost}

        if env_copy.context == 'column':

            ### gen costs

            ## count how many of these aligned states have observations on the main column
            agent.aligned_arm_gen_high_costs[city, day, t, i] = sum(
                1 for state in aligned_states
                if (state[1] in observed_high_cost_cols)
            )
            agent.aligned_arm_gen_low_costs[city, day, t, i] = sum(
                1 for state in aligned_states
                if (state[1] in observed_low_cost_cols)
            )

            ## count how many of the orthogonal states have observations on their respective columns
            agent.orthogonal_arm_gen_high_costs[city, day, t, i] = sum(
                1 for state in orthogonal_states
                if (state[1] in observed_high_cost_cols)
            )
            agent.orthogonal_arm_gen_low_costs[city, day, t, i] = sum(
                1 for state in orthogonal_states
                if (state[1] in observed_low_cost_cols)
            )


        elif env_copy.context == 'row':

            ### gen costs

            ## count how many of these aligned states have observations on the main row
            agent.aligned_arm_gen_high_costs[city, day, t, i] = sum(
                1 for state in aligned_states
                if (state[0] in observed_high_cost_rows)
                and (tuple(state) not in obs_list)
            )
            agent.aligned_arm_gen_low_costs[city, day, t, i] = sum(
                1 for state in aligned_states
                if (state[0] in observed_low_cost_rows)
                and (tuple(state) not in obs_list)
            )

            ## count how many of the orthogonal states have observations on their respective rows
            agent.orthogonal_arm_gen_high_costs[city, day, t, i] = sum(
                1 for state in orthogonal_states
                if (state[0] in observed_high_cost_rows)
                and (tuple(state) not in obs_list)
            )
            agent.orthogonal_arm_gen_low_costs[city, day, t, i] = sum(
                1 for state in orthogonal_states
                if (state[0] in observed_low_cost_rows)
                and (tuple(state) not in obs_list)
            )

        ## axis overlaps with future states
        agent.path_future_rel_overlaps[city, day, t, i] = env_copy.path_future_rel_overlaps[t][i]
        agent.path_future_irrel_overlaps[city, day, t, i] = env_copy.path_future_irrel_overlaps[t][i]
    
    ## misc
    agent.day_costs[city, day, t] = np.nansum(agent.total_costs[city, day, :t+1])
    agent.path_len[city, day, t] = len(env_copy.path_states[t][0])


def run_grid(agent, hyperparams, agent_name='CE', df_trials=None, envs=None, fit=True, yoked=False, progress=False):
    """Run an agent on the grid (AFC) task, looping over cities/days/trials."""

    ## init expt info
    try:
        n_trials = int(df_trials['trial'].max())
        n_days = int(df_trials['day'].max() )
        n_cities = int(df_trials['city'].max())
        N = envs['city_1_grid_1_env_object'][0].N
        n_afc = df_trials['path_chosen'].nunique()
    except:
        n_trials = hyperparams['n_trials']
        n_days = hyperparams['n_days']
        n_cities = hyperparams['n_cities']
        N = hyperparams['N']
        n_afc = hyperparams['n_afc']

    ## determine policy - i.e. greedy vs softmax
    if hyperparams is None:
        hyperparams = {}
    if 'greedy' in hyperparams:
        agent.greedy = hyperparams['greedy']
    else:
        agent.greedy = True

    ## initialise model's internal variables
    agent.n_afc = n_afc
    agent.p_choice = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.p_correct = np.zeros((n_cities, n_days, n_trials))
    agent.p_chose_orthogonal = np.zeros((n_cities, n_days, n_trials))
    agent.p_chose_more_future_rel_overlap = np.zeros((n_cities, n_days, n_trials))
    agent.Q_vals = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.actions = np.zeros((n_cities, n_days, n_trials))
    agent.objective = np.zeros((n_cities, n_days, n_trials))
    agent.CE_actions = np.zeros((n_cities, n_days, n_trials)) + np.nan
    agent.CE_p_choice = np.zeros((n_cities, n_days, n_trials, agent.n_afc)) + np.nan
    agent.CE_p_correct = np.zeros((n_cities, n_days, n_trials)) + np.nan
    agent.CE_Q_vals = np.zeros((n_cities, n_days, n_trials, agent.n_afc)) + np.nan
    agent.context_priors = np.zeros((n_cities, n_days, n_trials))
    agent.context_posteriors = np.zeros((n_cities, n_days, n_trials))
    agent.leaf_visits = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.total_costs = np.zeros((n_cities, n_days, n_trials))
    agent.path_quality = np.zeros((n_cities, n_days, n_trials))
    agent.true_context = []
    if fit:
        agent.n_total_trials = len(df_trials)
        agent.trial_loss = np.zeros(agent.n_total_trials)

    ## for extracting some useful trial data...
    agent.path_future_overlaps = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.path_future_rel_overlaps = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.path_future_irrel_overlaps = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.path_past_overlaps = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.path_past_observed_high_costs = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.path_past_observed_low_costs = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.path_len = np.zeros((n_cities, n_days, n_trials))
    agent.day_costs = np.zeros((n_cities, n_days, n_trials))
    agent.distr_diff = np.zeros((n_cities, n_days, n_trials))

    ## observations on the context-aligned and orthogonal arm of each path
    agent.aligned_arm_actual_high_costs = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.aligned_arm_actual_low_costs = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.orthogonal_arm_actual_high_costs = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.orthogonal_arm_actual_low_costs = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.aligned_arm_gen_high_costs = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.aligned_arm_gen_low_costs = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.orthogonal_arm_gen_high_costs = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.orthogonal_arm_gen_low_costs = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.aligned_arm_len = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.orthogonal_arm_len = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.aligned_path_aligned_arm_len = np.zeros((n_cities, n_days, n_trials))
    agent.aligned_path_orthogonal_arm_len = np.zeros((n_cities, n_days, n_trials))
    agent.orthogonal_path_aligned_arm_len = np.zeros((n_cities, n_days, n_trials))
    agent.orthogonal_path_orthogonal_arm_len = np.zeros((n_cities, n_days, n_trials))

    ## define whether or not we're extracting expt info based on yoked
    # if yoked:
    #     extract_fn = extract_grid_info
    # else:
    #     extract_fn = lambda agent, env_copy, city, day, t: None
    extract_fn = extract_grid_info
    

    if progress:
        pbar = tqdm(total=n_cities*n_days*n_trials, desc='Running {} agent'.format(agent_name), leave=False)

    ## loop through cities
    for city in range(n_cities):

        ## expts 1-2 - unknown context
        start_of_day_context_prior = 0.5

        ## loop through days
        for day in range(n_days):

            ## get the environment for this day
            if envs:
                env = envs['city_{}_grid_{}_env_object'.format(city+1, day+1)][0]

                ## need to do some fixes for old envs
                if env.expt == '2AFC':
                    env.expt = 'AFC'

                ## get context alignment of states
                env.path_aligned_states, env.path_orthogonal_states, env.path_weights = env.get_alignment(env.path_states)

                ## some envs are outdated and need to be given sim_weight_map
                if not hasattr(env, 'sim_weight_map'):
                    env.set_sim_weights(agent.aligned_weight, agent.orthogonal_weight)

            env_copy = copy.deepcopy(env)
            assert not hasattr(env_copy, 'obs'), 'env_copy.obs should not exist before the first trial: {}'.format(len(env_copy.obs),', city:', city+1, 'day:', day+1)

            ## expt 3 - context is known
            if env_copy.context == 'column':
                start_of_day_context_prior = 1.0
            elif env_copy.context == 'row':
                start_of_day_context_prior = 0.0

            ## context prior resets
            agent.context_prior = start_of_day_context_prior


            ## FIX FOR OLD ENVS: rename some attributes (episode --> trial, etc.)
            if hasattr(env_copy, 'n_episodes'):
                env_copy.n_trials = env_copy.n_episodes

            ## initialise planner
            agent.mcts = None
            tree_reset = True


            ## loop through trials within day
            for t in range(n_trials):

                ## reset env/trial
                env_copy.reset()
                env_copy.set_sim(True)

                ### extract useful behavioural measures (no-op if not yoked)
                extract_fn(agent, env_copy, city, day, t)


                ## agent-specific path selection
                Q_vals = agent.compute_Q(env_copy, tree_reset)
                action_probs = agent.softmax(Q_vals)

                ## action selection
                assert not np.isnan(np.nansum(Q_vals)), 'no Q estimates": {}'.format(Q_vals)
                if agent.greedy:
                    max_Q = np.nanmax(Q_vals)
                    action = np.argmax(Q_vals)
                else:
                    action = np.random.choice(len(Q_vals), p=action_probs)

                agent.actions[city, day, t] = action
                agent.Q_vals[city, day, t] = Q_vals
                agent.p_choice[city, day, t] = action_probs
                correct_path = np.argmax(env_copy.path_actual_costs[t])
                agent.p_correct[city, day, t] = agent.p_choice[city, day, t][correct_path]
                # agent.objective[city, day, t] = env_copy.objective
                if env_copy.objective == 'costs':
                    agent.objective[city, day, t] = -1
                elif env_copy.objective == 'rewards':
                    agent.objective[city, day, t] = 1

                ## let's also calculate the CE choice under the current agent's knowledge
                CE_Q_vals = agent.compute_CE_Q(env_copy)
                CE_action = np.argmax(CE_Q_vals)
                CE_action_probs = agent.softmax(CE_Q_vals)
                agent.CE_Q_vals[city, day, t] = CE_Q_vals
                agent.CE_actions[city, day, t] = CE_action
                agent.CE_p_choice[city, day, t] = CE_action_probs
                agent.CE_p_correct[city, day, t] = agent.CE_p_choice[city, day, t][correct_path]

                ## get info on orthogonal/overlap of paths
                more_future_rel_overlap = np.argmax(env_copy.path_future_rel_overlaps[t])
                agent.p_chose_more_future_rel_overlap[city, day, t] = agent.p_choice[city, day, t][more_future_rel_overlap]
                if env_copy.context == 'row':
                    if env_copy.dominant_axis_A[t] == 'horizontal':
                        aligned_path = 0
                        orthogonal_path = 1
                    elif env_copy.dominant_axis_A[t] == 'vertical':
                        aligned_path = 1
                        orthogonal_path = 0
                    elif env_copy.dominant_axis_A[t] == 'L-shaped':
                        aligned_path = np.nan
                        orthogonal_path = np.nan
                elif env_copy.context == 'column':
                    if env_copy.dominant_axis_A[t] == 'vertical':
                        aligned_path = 0
                        orthogonal_path = 1
                    elif env_copy.dominant_axis_A[t] == 'horizontal':
                        aligned_path = 1
                        orthogonal_path = 0
                    elif env_copy.dominant_axis_A[t] == 'L-shaped':
                        aligned_path = np.nan
                        orthogonal_path = np.nan
                if not np.isnan(orthogonal_path):
                    agent.p_chose_orthogonal[city, day, t] = agent.p_choice[city, day, t][orthogonal_path]
                    
                    ## also save arm lengths etc.
                    agent.aligned_path_aligned_arm_len[city,day,t] = agent.aligned_arm_len[city, day, t, aligned_path]
                    agent.aligned_path_orthogonal_arm_len[city,day,t] = agent.orthogonal_arm_len[city, day, t, aligned_path]
                    agent.orthogonal_path_aligned_arm_len[city,day,t] = agent.aligned_arm_len[city, day, t, orthogonal_path]
                    agent.orthogonal_path_orthogonal_arm_len[city,day,t] = agent.orthogonal_arm_len[city, day, t, orthogonal_path]

                else:
                    agent.p_chose_orthogonal[city, day, t] = np.nan
                    agent.aligned_path_aligned_arm_len[city,day,t] = np.nan
                    agent.aligned_path_orthogonal_arm_len[city,day,t] = np.nan
                    agent.orthogonal_path_aligned_arm_len[city,day,t] = np.nan
                    agent.orthogonal_path_orthogonal_arm_len[city,day,t] = np.nan


                ### take ppt's action if a) we are fitting, or b) we are extracting behavioural measures by yoking to ppt's choices
                missed=False
                if (fit) or (yoked):

                    ## first check if the participant has made a choice
                    try:
                        missed = pd.isna(df_trials.loc[(df_trials['city'] == city+1) & (df_trials['day'] == day+1) & (df_trials['trial'] == t+1), 'path_chosen'].values[0])
                    except:
                        missed = True

                    ## if the participant has made a choice, then we use their action (rather than the model's)
                    if not missed:
                        action = df_trials.loc[(df_trials['city'] == city+1) & (df_trials['day'] == day+1) & (df_trials['trial'] == t+1), 'path_chosen'].values[0]=='b'
                        assert np.isclose(df_trials.loc[(df_trials['city'] == city+1) & (df_trials['day'] == day+1) & (df_trials['trial'] == t+1), 'path_A_expected_cost'].values[0], env_copy.path_expected_costs[t][0], rtol=1e-5), 'expected cost does not match ppt data\n env: {}, ppt: {}'.format(env_copy.path_expected_costs[t][0], df_trials.loc[(df_trials['city'] == city+1) & (df_trials['day'] == day+1) & (df_trials['trial'] == t+1), 'path_A_expected_cost'].values[0])

                    else:
                        agent.p_choice[city, day, t] = np.nan
                        agent.p_correct[city, day, t] = np.nan
                        # agent.objective[city, day, t] = env_copy.objective
                        if env_copy.objective == 'costs':
                            agent.objective[city, day, t] = -1
                        elif env_copy.objective == 'rewards':
                            agent.objective[city, day, t] = 1
                        agent.p_chose_more_future_rel_overlap[city, day, t] = np.nan
                        agent.p_chose_orthogonal[city, day, t] = np.nan
                        agent.Q_vals[city, day, t] = np.nan
                        agent.actions[city, day, t] = np.nan
                        agent.context_priors[city, day, t] = np.nan
                        agent.context_posteriors[city, day, t] = np.nan
                        agent.leaf_visits[city, day, t] = np.nan
                        agent.total_costs[city, day, t] = np.nan

                ## only interact with the environment if participant made a choice
                if not missed:
                    env_copy.set_sim(False)
                    action_sequence = env_copy.path_actions[t][action]
                    _, cost, _, _, _ = env_copy.step(action)
                    agent.total_costs[city, day, t] = cost
                    agent.path_quality[city, day, t] = agent.total_costs[city, day, t]/agent.path_len[city, day, t]
                    day_terminated = t == (n_trials-1)


                ## else, skip to next trial??
                else:
                    # env_copy._trial += 1
                    env_copy.increment_trial()
                    print('skipping city {}, day {}, trial {} because participant missed their choice'.format(city+1, day+1, t+1))


                ## update MCTS tree
                if (not missed) and (not day_terminated):
                    tree_reset = agent.update_tree(env_copy, action)
                else:
                    tree_reset = True

                ## update the sampler with the new observations
                agent.sampler.set_obs(env_copy.obs)

                ## get the context prior - i.e. the probability with which samples were drawn
                context_prior = agent.context_prior

                ## update the context prior for the next trial
                context_posterior = agent.sampler.update_context_posterior(start_of_day_context_prior)
                agent.context_prior = context_posterior
                agent.context_priors[city, day, t] = context_prior
                agent.context_posteriors[city, day, t] = context_posterior

                ## carry over the context prob to the next run, if on the final trial of the day
                if t == (n_trials-1):
                    start_of_day_context_prior = context_posterior

                    ## also need to clear sampler
                    agent.sampler = None

                ## update progress bar
                if progress:
                    pbar.update(1)
        agent.true_context.append(env_copy.context)
    if progress:
        pbar.close()

    ## if we are fitting, calculate the loss
    if fit:
        agent.loss_func(df_trials)
        return agent.loss

    ## or, if we are running our own simulations, give the simulation output
    elif (not fit) & (df_trials is None):
        sim_out ={
            'participant':[],
            'agent':[],
            'city':[],
            'day':[],
            'trial':[],
            'context':[],
            'objective':[],
            'actions':[],
            'CE_actions':[],
            'total_costs':[],
            'distr_diff':[],
            'p_choice_A':[],
            'p_choice_B':[],
            'p_choice_C':[],
            'p_correct':[],
            'p_chose_more_future_rel_overlap':[],
            'p_chose_orthogonal':[],
            'Q_a':[],
            'Q_b':[],
            'Q_c':[],
            'CE_p_choice_A':[],
            'CE_p_choice_B':[],
            'CE_p_choice_C':[],
            'CE_p_correct':[],
            'CE_Q_a':[],
            'CE_Q_b':[],
            'CE_Q_c':[],
            'aligned_path_aligned_arm_len':[],
            'aligned_path_orthogonal_arm_len':[],
            'orthogonal_path_aligned_arm_len':[],
            'orthogonal_path_orthogonal_arm_len':[],
            'leaf_visits_a':[],
            'leaf_visits_b':[],
            'leaf_visits_c':[],
            'temp': [],
            'aligned_weight': [],
            'orthogonal_weight': [],
            'horizon': [],
        }
        for c in range(n_cities):
            for d in range(n_days):
                for t in range(n_trials):
                    sim_out['participant'].append(envs['participant'])
                    sim_out['agent'].append(agent_name)
                    sim_out['city'].append(c+1)
                    sim_out['day'].append(d+1)
                    sim_out['trial'].append(t+1)
                    sim_out['objective'].append(agent.objective[c][d][t])
                    sim_out['actions'].append(agent.actions[c][d][t])
                    sim_out['CE_actions'].append(agent.CE_actions[c][d][t])
                    sim_out['total_costs'].append(agent.total_costs[c][d][t])
                    sim_out['distr_diff'].append(agent.distr_diff[c][d][t])
                    sim_out['context'].append(agent.true_context[c])
                    sim_out['p_correct'].append(agent.p_correct[c][d][t])
                    sim_out['p_chose_more_future_rel_overlap'].append(agent.p_chose_more_future_rel_overlap[c][d][t])
                    sim_out['p_chose_orthogonal'].append(agent.p_chose_orthogonal[c][d][t])
                    sim_out['p_choice_A'].append(agent.p_choice[c][d][t][0])
                    sim_out['p_choice_B'].append(agent.p_choice[c][d][t][1])
                    sim_out['Q_a'].append(agent.Q_vals[c][d][t][0])
                    sim_out['Q_b'].append(agent.Q_vals[c][d][t][1])
                    sim_out['leaf_visits_a'].append(agent.leaf_visits[c][d][t][0])
                    sim_out['leaf_visits_b'].append(agent.leaf_visits[c][d][t][1])
                    sim_out['CE_p_correct'].append(agent.CE_p_correct[c][d][t])
                    sim_out['CE_p_choice_A'].append(agent.CE_p_choice[c][d][t][0])
                    sim_out['CE_p_choice_B'].append(agent.CE_p_choice[c][d][t][1])
                    sim_out['CE_Q_a'].append(agent.CE_Q_vals[c][d][t][0])
                    sim_out['CE_Q_b'].append(agent.CE_Q_vals[c][d][t][1])
                    sim_out['aligned_path_aligned_arm_len'].append(agent.aligned_path_aligned_arm_len[c][d][t])
                    sim_out['aligned_path_orthogonal_arm_len'].append(agent.aligned_path_orthogonal_arm_len[c][d][t])
                    sim_out['orthogonal_path_aligned_arm_len'].append(agent.orthogonal_path_aligned_arm_len[c][d][t])
                    sim_out['orthogonal_path_orthogonal_arm_len'].append(agent.orthogonal_path_orthogonal_arm_len[c][d][t])
                    sim_out['temp'].append(agent.temp)
                    sim_out['aligned_weight'].append(agent.aligned_weight)
                    sim_out['orthogonal_weight'].append(agent.orthogonal_weight)
                    sim_out['horizon'].append(agent.horizon)

                    if agent.n_afc==3:
                        sim_out['p_choice_C'].append(agent.p_choice[c][d][t][2])
                        sim_out['leaf_visits_c'].append(agent.leaf_visits[c][d][t][2])
                        sim_out['Q_c'].append(agent.Q_vals[c][d][t][2])
                        sim_out['CE_p_choice_C'].append(agent.CE_p_choice[c][d][t][2])
                        sim_out['CE_Q_c'].append(agent.CE_Q_vals[c][d][t][2])
                    else:
                        sim_out['p_choice_C'].append(np.nan)
                        sim_out['leaf_visits_c'].append(np.nan)
                        sim_out['Q_c'].append(np.nan)
                        sim_out['CE_p_choice_C'].append(np.nan)
                        sim_out['CE_Q_c'].append(np.nan)
        return sim_out


def run_bandit(agent, env, greedy=False, verbose=True):
    """Run an agent on the bandit task for n_trials, returning trial-by-trial data."""
    n_trials = env.n_trials
    n_arms = env.n_afc

    Q_history = np.zeros((n_trials, n_arms))
    p_choice_history = np.zeros((n_trials, n_arms))
    actions = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    chose_optimal = np.zeros(n_trials, dtype=bool)
    bayes_regret = np.zeros(n_trials)
    post_probs = np.zeros((n_trials, n_arms))

    ## CE tracking — what would the CE agent have chosen with the same info?
    CE_Q_history = np.zeros((n_trials, n_arms))
    CE_actions = np.zeros(n_trials, dtype=int)
    CE_chose_optimal = np.zeros(n_trials, dtype=bool)
    CE_bayes_regret = np.zeros(n_trials)

    optimal_arm = np.argmax(env.p_dist)

    env.reset()

    for t in range(n_trials):
        env_copy = copy.deepcopy(env)
        env_copy.set_sim(True)

        Q = agent.compute_Q(env_copy)
        probs = agent.softmax(Q)
        mean_probs = agent.sampler.mean_probs()

        ## CE Q values under the same posterior
        CE_Q = agent.compute_CE_Q(env_copy)
        CE_action = int(np.argmax(CE_Q))

        Q_history[t] = Q
        p_choice_history[t] = probs
        post_probs[t] = mean_probs
        CE_Q_history[t] = CE_Q
        CE_actions[t] = CE_action
        CE_chose_optimal[t] = (CE_action == optimal_arm)
        CE_bayes_regret[t] = env.p_dist[optimal_arm] - env.p_dist[CE_action]

        if greedy:
            max_Q = np.nanmax(Q)
            best_arms = np.where(Q == max_Q)[0]
            action = int(np.random.choice(best_arms))

            # debugging
            if action != CE_action:
                if verbose:
                    print(f"Trial {t+1}: Greedy action {action} differs from CE action {CE_action} with Q values {Q} and CE Q values {CE_Q}")
        else:
            max_Q = np.nanmax(Q)
            best_arms = np.where(Q == max_Q)[0]
            if len(best_arms) > 1:
                action = int(np.random.choice(best_arms))
            else:
                action = int(best_arms[0])

        env.set_sim(False)
        _, reward, terminated, truncated, _ = env.step(action)

        actions[t] = action
        rewards[t] = reward
        chose_optimal[t] = (action == optimal_arm)
        bayes_regret[t] = env.p_dist[optimal_arm] - env.p_dist[action]

        agent.init_sampler(env)

        if verbose:
            print(f"  trial {t+1:>3}/{n_trials}  \n Q={np.round(Q, 3)}  CE_Q={np.round(CE_Q, 3)}"
                  f"p={np.round(probs, 3)} \n pulled arm {action} with reward {reward:.0f}")

        if terminated:
            break

    return {
        'Q': Q_history,
        'p_choice': p_choice_history,
        'actions': actions,
        'rewards': rewards,
        'cumulative_reward': np.cumsum(rewards),
        'chose_optimal': chose_optimal,
        'true_probs': env.p_dist.copy(),
        'optimal_arm': optimal_arm,
        'bayes_regret': bayes_regret,
        'post_probs': post_probs,
        'CE_Q': CE_Q_history,
        'CE_actions': CE_actions,
        'CE_chose_optimal': CE_chose_optimal,
        'CE_bayes_regret': CE_bayes_regret,
    }

def run_emp(agent, env, verbose=True):
    """Run an agent on the empowerment bandit task for n_trials."""
    n_trials = env.n_trials
    n_arms = env.n_afc
    n_outcomes = env.n_outcomes

    p_choice_history = np.zeros((n_trials, n_arms))
    actions = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)

    env.reset()

    for t in range(n_trials):
        env_copy = copy.deepcopy(env)
        env_copy.set_sim(True)
        

        Q = agent.compute_Q(env_copy)
        probs = agent.softmax(Q)
        mean_probs = agent.sampler.mean_probs()
        max_Q = np.nanmax(Q)
        best_arms = np.where(Q == max_Q)[0]
        if len(best_arms) > 1:
            action = int(np.random.choice(best_arms))
        else:            
            action = int(best_arms[0])
        
        p_choice_history[t] = probs

        env.set_sim(False)
        (_, outcome), reward, terminated, truncated, _ = env.step(action)

        actions[t] = action
        outcomes[t] = outcome
        rewards[t] = reward

        if verbose:
            print(f"  trial {t+1:>3}/{n_trials}  Q-values {np.round(Q, 3)}  pulled arm {action}, outcome {outcome}, "
                  f"empowerment reward {reward:.4f}")

        if terminated or truncated:
            break

    return {
        'p_choice': p_choice_history,
        'actions': actions,
        'outcomes': outcomes,
        'rewards': rewards,
        'cumulative_reward': np.cumsum(rewards),
        'true_p_matrix': env.p_matrix.copy(),
        'posterior_p_matrix': env.posterior_p_matrix.copy(),
        'ell': env.ell,
    }


def _bellman_emp_V(alphas, n_arms, n_outcomes, depth, termination_arm, ell):
    """Bayes-adaptive optimal value with `depth` future pulls remaining.

    V(h, 0)   = emp_l(h)
    V(h, d>0) = max_a sum_o p(o|a, h) * V(h u (a,o), d-1)

    `alphas` is the running Dirichlet posterior; mutated in place and restored.
    """
    posterior_p = alphas / alphas.sum(axis=1, keepdims=True)
    if depth == 0:
        return float(np.sum(np.max(posterior_p, axis=0) ** ell))
    if termination_arm:
        best = float(np.sum(np.max(posterior_p, axis=0) ** ell)) ## value of terminating immediately, i.e. current empowerment, without any more samples
    else:
        best = -np.inf
    for a in range(n_arms):
        denom = alphas[a].sum()
        ev = 0.0
        for o in range(n_outcomes):
            p_o = alphas[a, o] / denom
            alphas[a, o] += 1
            ev += p_o * _bellman_emp_V(alphas, n_arms, n_outcomes, depth - 1, termination_arm, ell)
            alphas[a, o] -= 1
        if ev > best:
            best = ev
    return best


def _bellman_emp_Q(current_alphas, n_arms, n_outcomes, h, termination_arm, ell, verbose=False):
    """Per-first-action Bayes-adaptive optimal Q with horizon h.

    Q[a_1] = sum_o p(o|a_1, h) * V(h u (a_1, o), depth = h-1).
    Subsequent actions are taken to maximise expected end-state empowerment
    given the resulting belief, i.e. argmax_a inside _bellman_emp_V.
    """
    Q = np.zeros(n_arms + termination_arm)
    work = current_alphas.astype(float).copy()
    for a in range(n_arms):
        denom = work[a].sum()
        for o in range(n_outcomes):
            p_o = work[a, o] / denom
            work[a, o] += 1
            V = _bellman_emp_V(work, n_arms, n_outcomes, h - 1, termination_arm, ell)
            Q[a] += p_o * V
            if verbose:
                print(f"action {a}, outcome {o}, p(o|a,h)={p_o:.4f}, V(h u (a,o), h-1)={V:.4f}")
            work[a, o] -= 1

    ## Q(terminate) is just the immediate empowerment under the current belief, no future pulls
    if termination_arm:
        posterior_p = current_alphas / current_alphas.sum(axis=1, keepdims=True)
        Q[-1] = float(np.sum(np.max(posterior_p, axis=0) ** ell))
    return Q


def _uniform_tail_emp_Q(current_alphas, n_arms, n_outcomes, h, ell):
    """Q per first action by exhaustive (a, o) enumeration with PPD weights.

    Equivalent to assuming a uniform random policy over subsequent actions
    a_2, ..., a_h and weighting outcome sequences by their posterior-predictive
    probability under env.alphas. A lower bound on Bayes-adaptive optimal Q;
    kept alongside _bellman_emp_Q for empirical comparison.
    """
    Q = np.zeros(n_arms)
    Z = np.zeros(n_arms)
    pairs = list(itertools.product(range(n_arms), range(n_outcomes)))

    for seq in itertools.product(pairs, repeat=h):
        a_1 = seq[0][0]
        seq_counts = np.zeros((n_arms, n_outcomes), dtype=int)
        log_w = 0.0
        for (a, o) in seq:
            num = current_alphas[a, o] + seq_counts[a, o]
            den = current_alphas[a].sum() + seq_counts[a].sum()
            log_w += np.log(num) - np.log(den)
            seq_counts[a, o] += 1

        final_alphas = current_alphas + seq_counts
        posterior_p = final_alphas / final_alphas.sum(axis=1, keepdims=True)
        emp = float(np.sum(np.max(posterior_p, axis=0) ** ell))

        w = np.exp(log_w)
        Q[a_1] += w * emp
        Z[a_1] += w

    return Q / Z


def run_emp_enum(agent, env, horizon=None, policy='bellman', termination_arm=None, verbose=True):
    """Run an empowerment-bandit agent with exact Q estimates.

    `policy='bellman'` (default): Bayes-adaptive optimal Q via the recursion
        V(h, 0)   = emp_l(h)
        V(h, d>0) = max( emp_l(h),                                       # terminate
                         max_a sum_o p(o|a,h) V(h u (a,o), d-1) )        # pull arm a
        Q(h, a_1) = sum_o p(o|a_1, h) V(h u (a_1, o), H-1)
        Q(h, terminate) = emp_l(h)
    Subsequent actions are assumed Bayes-optimal -- this is the value BAMCP
    approximates.

    `policy='uniform_tail'`: Q under a uniform random follow-up policy --
    exhaustive enumeration of all (a, o) sequences with posterior-predictive
    weights, averaged uniformly over the action tail. Lower bound on the
    Bellman Q; useful as a comparison baseline.

    `termination_arm`: if True (or auto-detected from `env.termination_arm`),
    the agent has an extra action that immediately collects the current
    empowerment and ends the episode. `uniform_tail` does not currently
    support termination.

    H = min(horizon, n_trials - t) is the remaining horizon at each trial,
    p(o|a, h) is the posterior predictive of the env's Dirichlet posterior.
    """
    if termination_arm is None:
        termination_arm = bool(getattr(env, 'termination_arm', False))

    if policy == 'bellman':
        Q_fn = lambda alphas, n_a, n_o, h_, e: _bellman_emp_Q(
            alphas, n_a, n_o, h_, termination_arm, e, verbose=verbose)
    elif policy == 'uniform_tail':
        if termination_arm:
            raise NotImplementedError("uniform_tail policy does not support termination_arm")
        Q_fn = _uniform_tail_emp_Q
    else:
        raise ValueError(f"unknown policy {policy!r}; expected 'bellman' or 'uniform_tail'")

    n_trials = env.n_trials
    n_arms = getattr(env, 'n_arms', env.n_afc - int(termination_arm))
    n_outcomes = env.n_outcomes
    ell = env.ell
    n_actions = n_arms + int(termination_arm)
    terminate_idx = n_arms if termination_arm else None

    Q_history = np.zeros((n_trials, n_actions))
    p_choice_history = np.zeros((n_trials, n_actions))
    p_repeat_choice = np.zeros(n_trials)
    emp_improvement = np.zeros((n_trials, n_actions))
    actions = np.zeros(n_trials, dtype=int)
    outcomes = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)

    env.reset()

    ## calculate initial empowerment under flat prior
    flat_prior_p = np.ones((n_arms, n_outcomes)) / n_outcomes
    prev_emp = env.empowerment(flat_prior_p, ell)
    print('initial emp:', prev_emp)

    last_t = n_trials - 1
    for t in range(n_trials):
        h = (n_trials - t) if horizon is None else min(horizon, n_trials - t)

        Q = Q_fn(env.alphas.copy(), n_arms, n_outcomes, h, ell)
        probs = agent.softmax(Q)

        max_Q = np.nanmax(Q)
        best_arms = np.where(Q == max_Q)[0]
        if len(best_arms) > 1:
            action = int(np.random.choice(best_arms))
        else:
            action = int(best_arms[0])

        Q_history[t] = Q
        p_choice_history[t] = probs

        env.set_sim(False)
        (_, outcome), reward, terminated, truncated, _ = env.step(action)

        actions[t] = action
        outcomes[t] = outcome
        rewards[t] = reward
        last_t = t

        emp_improvement[t] = Q / prev_emp
        prev_emp = reward

        if t == 0:
            p_repeat_choice[t] = np.nan
        else:
            last_action = actions[t-1]
            p_repeat_choice[t] = probs[last_action]

        if verbose:
            action_str = 'terminate' if action == terminate_idx else f'arm {action}'
            print(f"  trial {t+1:>3}/{n_trials}  Q={np.round(Q, 4)}  "
                  f"chose {action_str}, outcome {outcome}, "
                  f"empowerment reward {reward:.4f}")

        if terminated or truncated:
            break

    ## trim trailing zeros if the agent terminated early
    keep = last_t + 1
    return {
        'Q': Q_history[:keep],
        'p_choice': p_choice_history[:keep],
        'p_repeat_choice': p_repeat_choice[:keep],
        'actions': actions[:keep],
        'outcomes': outcomes[:keep],
        'emp_improvement': emp_improvement[:keep],
        'rewards': rewards[:keep],
        'cumulative_reward': np.cumsum(rewards[:keep]),
        'true_p_matrix': env.p_matrix.copy(),
        'posterior_p_matrix': env.posterior_p_matrix.copy(),
        'ell': ell,
        'termination_arm': termination_arm,
        'terminated_early': terminated and (action == terminate_idx) if termination_arm else False,
    }


def enumerate_emp_histories(n_arms=2, n_outcomes=2, n_trials=3, alpha=1.0, termination_arm = True,
                            ells=(0.33, 1.0, 3.0), temp=1.0):
    """Enumerate every reachable (a, o) history of length 0..n_trials-1.

    For each history h and each ell, computes:
      - current_emp: empowerment of the posterior implied by h
      - Q[a]: Bayes-adaptive optimal value of taking arm a from h with the
        remaining horizon n_trials - len(h)
      - probs[a]: softmax(Q / temp) — the agent's choice probabilities
      - p_repeat: probs[h[-1][0]] if h non-empty else NaN
      - delta_emp[a]: 1-step expected empowerment gain
            E_o~p(o|a,h)[Emp(h u (a,o))] - Emp(h)

    Returns a long-format DataFrame with one row per (ell, history).
    """
    import importlib.util as _ilu
    from scipy.special import softmax as _softmax

    _spec = _ilu.spec_from_file_location("bandit", "gym_bandits/bandit.py")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    EmpBandit = _mod.EmpBandit

    rows = []
    pairs = list(itertools.product(range(n_arms), range(n_outcomes)))

    for ell in ells:
        for t in range(n_trials):
            for history in itertools.product(pairs, repeat=t):
                alphas = np.full((n_arms, n_outcomes), float(alpha))
                for (a, o) in history:
                    alphas[a, o] += 1

                current_p = alphas / alphas.sum(axis=1, keepdims=True)
                current_emp = EmpBandit.empowerment(current_p, ell)

                h_remaining = n_trials - t
                Q = _bellman_emp_Q(alphas.copy(), n_arms, n_outcomes, 
                                   h_remaining, termination_arm, ell, verbose=False)
                best_a = np.argmax(Q)

                probs = _softmax(Q / temp)

                delta_emp = np.zeros(n_arms)
                for a in range(n_arms):
                    denom = alphas[a].sum()
                    expected = 0.0
                    for o in range(n_outcomes):
                        p_o = alphas[a, o] / denom
                        next_alphas = alphas.copy()
                        next_alphas[a, o] += 1
                        next_p = next_alphas / next_alphas.sum(axis=1, keepdims=True)
                        expected += p_o * EmpBandit.empowerment(next_p, ell)
                    delta_emp[a] = expected - current_emp
                    # delta_emp[a] = Q[a] - current_emp

                prev_action = history[-1][0] if t > 0 else None
                p_repeat = probs[prev_action] if prev_action is not None else np.nan

                row = {
                    'ell': ell,
                    't': t,
                    'history': history,
                    'history_str': '-'.join(f'a{a}o{o}' for (a, o) in history) or 'init',
                    'prev_action': prev_action if prev_action is not None else np.nan,
                    'current_emp': current_emp,
                    'p_repeat': p_repeat,
                    'best_a': best_a,
                }
                for a in range(n_arms):
                    row[f'Q_{a}'] = Q[a]
                    row[f'p_{a}'] = probs[a]
                    row[f'delta_emp_{a}'] = delta_emp[a]
                if termination_arm:
                    row['Q_terminate'] = Q[-1]
                    row['p_terminate'] = probs[-1]
                rows.append(row)

    df = pd.DataFrame(rows)
    df['history_counts'], df['history_counts_str'] = zip(*df['history'].apply(get_history_counts))

    return df


## define unordered 'history_counts', i.e. sufficient statistic for belief state
def get_history_counts(history):
    if history == 'init':
        return 'init'
    pairs = history
    counts = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1
    sorted_counts = tuple(sorted(counts.items()))
    str_counts = '-'.join(f'a{a}o{o}:{count}' for ((a, o), count) in sorted_counts)
    return sorted_counts, str_counts




