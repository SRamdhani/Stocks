from stockbot.main.dbupdate import update
from stockbot.util.environment import envSetup
from stockbot.main.models import Model
from stockbot.logs.logs import Logger
from stockbot import session
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os

# TODO: Add logging to classes!
# TODO: Add Readme!

# Configuration parameters for the whole setup
TOTAL_ITERATIONS = 10
REFRESH          = True
ACTIONS          = ['buy', 'sell', 'hold']
WINDOW           = 45
COMPNUM          = 100

observation_dimensions = WINDOW * COMPNUM
num_actions = len(ACTIONS)
seed        = 42
gamma       = 1  # Discount factor for past rewards
eps         = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
num_inputs  = observation_dimensions
num_hidden  = 256  # 128

logger      = Logger.loggerSetup()
predictions = pd.read_csv('predictions.csv', index_col=0).reset_index(drop=True)
predDict    = {0:'Buy', 1: 'Sell', 2:'Hold'}

allSymbols  = ['SPY', 'GOOG', 'MSFT', 'TSLA','QQQ','IWM','GLD','SMH','TLT','IYT','XLF',
              'JNK','FREL','XLV','XLI','VXX','WMT','OIH','MCHI','IEV','FDIS','CAT','NVDA',
              'LEN','PFE','MRK','LULU','NKE','ULTA','HAS','AAPL','META','INTC','COIN','GS',
              'V','AXP','DAL','HD','AMZN','UPS','CMG','VZ','BA','CRM','JPM','DIS','SBUX','RTX','XOM']

max_reward_models = json.load(open('max_reward_models.json','r'))
Logger.info(Model().getLogDictInfo('run', __name__, __name__),
            'Max Reward Models: ' + json.dumps(max_reward_models),
            logger)

for i, target in enumerate(['TSLA', 'SPY', 'MSFT', 'QQQ', 'GOOG']):

    Logger.info(Model().getLogDictInfo('run', __name__, __name__),
                'Running Model for: ' + target, logger)

    symbols, rewardArr, anchor_date, factorRes = Model.getDecomposition(i, logger, update,
                                                                        allSymbols, session,
                                                                        COMPNUM, target)
    if i == 0:
        Logger.info(Model().getLogDictInfo('run', __name__, __name__),
                    'Anchor Date: ' + str(anchor_date[-1]), logger)

    env = envSetup(window=WINDOW,
                   compData=factorRes[0],
                   rewardData=rewardArr,
                   actionDict=dict(zip(range(len(ACTIONS)),ACTIONS)),
                   balance=10000)

    Logger.info(Model().getLogDictInfo('run', __name__, __name__),
                'Environment Set Up.', logger)

    try:
        model = keras.models.load_model(target.lower() + '_larger_penalty.keras')
        Logger.info(Model().getLogDictInfo('run', __name__, __name__),
                    'Loaded Existing Model for ' + target.lower(), logger)
    except Exception as e:
        Logger.info(Model().getLogDictInfo('run', __name__, __name__), str(e), logger)
        Logger.info(Model().getLogDictInfo('run', __name__, __name__),
                    'No Existing Model for ' + target.lower(), logger)
        model = Model.basicACModel(num_inputs, num_hidden, num_actions)

    Model().train(env, model, num_actions, gamma,
          target, logger, anchor_date, WINDOW, REFRESH,
          max_reward_models, TOTAL_ITERATIONS, eps)

    try:
        model = keras.models.load_model(target.lower() + '_larger_penalty.keras')
    except Exception as e:
        Logger.error(Model().getLogDictInfo('run', __name__, __name__),
                     str(e), logger)
        continue

    state = tf.convert_to_tensor(env.predData.flatten())
    state = tf.expand_dims(state, 0)
    Logger.info(Model().getLogDictInfo('run', __name__, __name__),
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
os.remove('factorRes.pkl')
Logger.info(Model().getLogDictInfo('run', __name__, __name__),
        'Saved Predictions & Deleted Factor Components.', logger)