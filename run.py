from stockbot.main.dbupdate import update
from stockbot.util.environment import envSetup
from stockbot.main.models import Model
from stockbot.logs.logs import Logger
from stockbot import session
from tensorflow import keras
import tensorflow as tf
import pickle as pkl
import pandas as pd
import numpy as np
import json
import os

# TODO: Add logging to classes!
# TODO: Add Readme!

def getLogDictInfo(pkgLocation, className, methodName):
    return {
        'class': className,
        'method': methodName,
        'pkgLocation': pkgLocation
    }

TOTAL_ITERATIONS = 10
REFRESH          = True
ACTIONS          = ['buy', 'sell', 'hold']
WINDOW           = 45
COMPNUM          = 100

predictions = pd.read_csv('predictions.csv', index_col=0).reset_index(drop=True)
predDict    = {0:'Buy', 1: 'Sell', 2:'Hold'}
allSymbols  = ['SPY', 'GOOG', 'MSFT', 'TSLA','QQQ','IWM','GLD','SMH','TLT','IYT','XLF',
              'JNK','FREL','XLV','XLI','VXX','WMT','OIH','MCHI','IEV','FDIS','CAT','NVDA',
              'LEN','PFE','MRK','LULU','NKE','ULTA','HAS','AAPL','META','INTC','COIN','GS',
              'V','AXP','DAL','HD','AMZN','UPS','CMG','VZ','BA','CRM','JPM','DIS','SBUX','RTX','XOM']

logger = Logger.loggerSetup()
max_reward_models = json.load(open('max_reward_models.json','r'))
Logger.info(getLogDictInfo('run', __name__, __name__),
            'Max Reward Models: ' + json.dumps(max_reward_models),
            logger)

for i, target in enumerate(['TSLA', 'SPY', 'MSFT', 'QQQ', 'GOOG']):
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    main_action_history = []
    shares_history = []
    balance_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0
    iterations = -1

    Logger.info(getLogDictInfo('run', __name__, __name__),
                'Running Model for: ' + target, logger)

    symbols, rewardArr, anchor_date, factorRes = Model.getDecomposition(i, logger, update,
                                                                        allSymbols, session,
                                                                        COMPNUM, target)
    if i == 0:
        Logger.info(getLogDictInfo('run', __name__, __name__),
                    'Anchor Date: ' + str(anchor_date[-1]), logger)

    env = envSetup(window=WINDOW,
                   compData=factorRes[0],
                   rewardData=rewardArr,
                   actionDict=dict(zip(range(len(ACTIONS)),ACTIONS)),
                   balance=10000)

    Logger.info(getLogDictInfo('run', __name__, __name__),
                'Environment Set Up.', logger)

    observation_dimensions = WINDOW*COMPNUM
    num_actions            = len(ACTIONS)

    # Configuration parameters for the whole setup
    seed        = 42
    gamma       = 1                                # Discount factor for past rewards
    eps         = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
    num_inputs  = observation_dimensions
    num_actions = num_actions
    num_hidden  = 256 # 128

    try:
        model = keras.models.load_model(target.lower() + '_larger_penalty.keras')
        Logger.info(getLogDictInfo('run', __name__, __name__),
                    'Loaded Existing Model for ' + target.lower(), logger)
    except:
        Logger.info(getLogDictInfo('run', __name__, __name__),
                    'No Existing Model for ' + target.lower(), logger)
        model = Model.basicACModel(num_inputs, num_hidden, num_actions)

    if target in max_reward_models:
        if REFRESH:
            max_reward = max_reward_models[target]/1.25
        else:
            max_reward = max_reward_models[target]
        optimizer = keras.optimizers.legacy.Adam(learning_rate=1e-5)
    else:
        max_reward = 0
        optimizer = keras.optimizers.legacy.Adam(learning_rate=1e-2)

    while True:  # Run until solved
        iterations+=1
        state = env.reset()
        episode_reward = 0
        env.penalty = -5

        # Borrowed and adapted from https://keras.io/examples/rl/actor_critic_cartpole/.
        with tf.GradientTape() as tape:
            for timestep in range(env.maxiter):
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                # Predict action probabilities and estimated future rewards
                # from environment state

                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])
                # Sample action from action probability distribution
                action = np.random.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, action]))
                # Apply the sampled action in our environment
                state, reward, done, _ = env.step(it=timestep, action=action)
                main_action_history.append(action)

                rewards_history.append(reward)
                episode_reward += reward
                if done:
                    break

            rewards_history_copy = rewards_history.copy()
            # Quick check on Total Buys and Sells
            penaltyTradesBool = np.array(rewards_history) == env.penalty
            totalBuys         = (np.array(main_action_history)[~penaltyTradesBool]==env.reverseActionDict['buy']).sum()
            totalSells        = (np.array(main_action_history)[~penaltyTradesBool]==env.reverseActionDict['sell']).sum()

            if totalBuys > totalSells:
                correction = env.sharePrice*env.shares
                rewards_history[-1] += correction

            # Update running reward to check condition for solving
            # running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            running_reward = episode_reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic

            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss
                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )
            
            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Log details
        episode_count += 1
        template = target + " running reward: {:.2f} at episode {}"
        Logger.info(getLogDictInfo('run', __name__, __name__),
                    template.format(running_reward, episode_count), logger)

        if running_reward > max_reward:
            max_reward = running_reward
            buysellaction = [env.actionDict[m] for m in main_action_history]
            results = pd.DataFrame([rewards_history_copy, buysellaction]).T
            results['date']= anchor_date[WINDOW:]

            max_reward_models[target] = max_reward
            json.dump(max_reward_models, open('max_reward_models.json', 'w'))
            results.to_csv(target.lower() + '_larger_penalty.results', index=None)
            model.save(target.lower() + "_larger_penalty.keras")

            Logger.info(getLogDictInfo('run', __name__, __name__),
                        'Saved Model for ' + target + ' with reward: ' + str(max_reward), logger)

        if iterations > TOTAL_ITERATIONS:
            break

        action_probs_history.clear()
        critic_value_history.clear()
        main_action_history.clear()
        rewards_history.clear()

    try:
        model = keras.models.load_model(target.lower() + '_larger_penalty.keras')
    except Exception as e:
        Logger.error(getLogDictInfo('run', __name__, __name__),
                    str(e), logger)
        continue

    state = tf.convert_to_tensor(env.predData.flatten())
    state = tf.expand_dims(state, 0)
    Logger.info(getLogDictInfo('run', __name__, __name__),
                target + str(model(state)), logger)

    target_pred = model(state)[0].numpy()
    pred_date   = str(anchor_date[-1].date().strftime('%Y-%m-%d'))
    closing     = round(rewardArr[-1][-1],2)
    targetIdx   = np.argmax(target_pred[0]) if target_pred[0][np.argmax(target_pred[0])] > 0.45 else 2

    predItem = [pred_date, closing, predDict[targetIdx],
                target.lower(), target_pred[0][0], target_pred[0][1],
                target_pred[0][2]]

    predictions.loc[len(predictions.index)] = predItem

    del globals()['model']
    del globals()['grads']
    del globals()['optimizer']
    del globals()['factorRes']
    tf.keras.backend.clear_session()

predictions.to_csv('predictions.csv')
os.remove('./factorRes.pkl')
Logger.info(getLogDictInfo('run', __name__, __name__),
        'Saved Predictions & Deleted Factor Components.', logger)