import copy
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
            path_past_observed_high_costs = sum(env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost for obs in overlap)
            path_past_observed_low_costs = sum(env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost for obs in overlap)
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
        observed_high_cost_cols = {obs[1] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost}
        observed_low_cost_cols = {obs[1] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost}
        observed_high_cost_rows = {obs[0] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost}
        observed_low_cost_rows = {obs[0] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost}
        observed_high_cost_states = {tuple(obs) for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost}
        observed_low_cost_states = {tuple(obs) for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost}

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
    agent.aligned_path_aligned_arm_len = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.aligned_path_orthogonal_arm_len = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.orthogonal_path_aligned_arm_len = np.zeros((n_cities, n_days, n_trials, agent.n_afc))
    agent.orthogonal_path_orthogonal_arm_len = np.zeros((n_cities, n_days, n_trials, agent.n_afc))

    ## define whether or not we're extracting expt info based on yoked
    if yoked:
        extract_fn = extract_grid_info
    else:
        extract_fn = lambda agent, env_copy, city, day, t: None
    

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
                    env_copy._trial += 1
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
            'lapse': [],
            'arm_weight': [],
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
                    sim_out['lapse'].append(agent.lapse)
                    sim_out['arm_weight'].append(agent.arm_weight)
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
