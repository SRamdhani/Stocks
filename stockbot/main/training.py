from dataclasses import dataclass
from ..logs.logs import Logger
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import json

@dataclass(frozen=True, unsafe_hash=True)
class Train:

    @staticmethod
    def train(env, model, num_actions, gamma,
              target, logger, anchor_date, window, refresh,
              max_reward_models, total_iterations, eps):

        huber_loss = keras.losses.Huber()
        action_probs_history = []
        critic_value_history = []
        main_action_history = []
        rewards_history = []
        episode_count = 0
        iterations = -1

        if target in max_reward_models:
            if refresh:
                max_reward = max_reward_models[target] / 1.25
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

            # Adapted from https://keras.io/examples/rl/actor_critic_cartpole/.
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
                totalBuys         = (np.array(main_action_history)[~penaltyTradesBool ]==env.reverseActionDict['buy']).sum()
                totalSells        = (np.array(main_action_history)[~penaltyTradesBool ]==env.reverseActionDict['sell']).sum()

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
            Logger.info(Logger().getLogDictInfo('run', __name__, __name__),
                        template.format(running_reward, episode_count), logger)

            if running_reward > max_reward:
                max_reward = running_reward
                buysellaction = [env.actionDict[m] for m in main_action_history]
                results = pd.DataFrame([rewards_history_copy, buysellaction]).T
                results['date' ]= anchor_date[window:]

                max_reward_models[target] = max_reward
                json.dump(max_reward_models, open('max_reward_models.json', 'w'))
                results.to_csv(target.lower() + '_larger_penalty.results', index=None)
                model.save(target.lower() + "_larger_penalty.keras")

                Logger.info(Logger().getLogDictInfo(__class__.__name__, __name__, 'train'),
                            'Saved Model for ' + target + ' with reward: ' + str(max_reward), logger)

            if iterations > total_iterations:
                break

            action_probs_history.clear()
            critic_value_history.clear()
            main_action_history.clear()
            rewards_history.clear()
