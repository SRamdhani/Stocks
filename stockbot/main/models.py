from tensorly.decomposition import parafac
from tensorflow.keras import layers
from dataclasses import dataclass
from ..util.utils import Utility
from ..logs.logs import Logger
from tensorflow import keras
import tensorly as tl
import pickle as pkl
import random
import os

@dataclass(frozen=True, unsafe_hash=True)
class Model:
    @staticmethod
    def getLogDictInfo(pkgLocation, className, methodName):
        return {
            'class': className,
            'method': methodName,
            'pkgLocation': pkgLocation
        }

    ''' Static method to run the decomposition to get component level features.'''
    @staticmethod
    def decomposition(allSymbols, session, components = 10, rewardName = 'GOOG', anchor_symbol = 'SPY', getFactorRes = False):
        TensorArray, TensorSymbol, RewardArray, anchor_date = Utility.createTensor(allSymbols, session, rewardName, anchor_symbol)
        TensorArray = TensorArray.transpose(1,2,0)

        if getFactorRes:
            random.seed(10)
            ttensor = tl.tensor(TensorArray)
            factors = parafac(ttensor, rank=components, init='random', tol=10e-6)

            factorRes = []

            for f in factors:
                factorRes.append(f)

            return factorRes[1], TensorSymbol, RewardArray, anchor_date
        else:
            return TensorSymbol, RewardArray, anchor_date

    '''
        Basic Reinforcement Learning MLP model for action and critic.
    '''
    @staticmethod
    def basicACModel(num_inputs, num_hidden, num_actions):
        inputs = layers.Input(shape=(num_inputs,))
        weight = layers.Dense(num_hidden)(inputs)
        action = layers.Dense(num_actions, activation="softmax")(weight)
        critic = layers.Dense(1)(weight)
        return keras.Model(inputs=inputs, outputs=[action, critic])


    @staticmethod
    def getDecomposition(i, logger, update, allSymbols, session, compnum, target):
        if i == 0:
            Logger.info(Model.getLogDictInfo(__class__.__name__, __name__, 'getDecomposition'),
                        'Database update.' , logger)

            update.database(allSymbols, session=session)

            if not os.path.exists('~/stocks/factorRes.pkl'):
                Logger.info(Model.getLogDictInfo(__class__.__name__, __name__, 'getDecomposition'),
                            'Factor Components Does Not Exist. Creating Pickle Backup.', logger)

                factorRes, symbols, rewardArr, anchor_date = Model.decomposition(allSymbols, session, components = compnum,
                                                                        rewardName = target, anchor_symbol = 'SPY',
                                                                        getFactorRes = True)


                with open('~/stocks/factorRes.pkl','wb') as f:
                    pkl.dump(factorRes,f)
            else:
                Logger.info(Model.getLogDictInfo(__class__.__name__, __name__, 'getDecomposition'),
                            'Factor Components Exists. Loading Pickle Backup.', logger)

                symbols, rewardArr, anchor_date = Model.decomposition(allSymbols, session, components = compnum,
                                                             rewardName = target, anchor_symbol = 'SPY',
                                                             getFactorRes = False)

                with open('~/stocks/factorRes.pkl','rb') as f:
                    factorRes = pkl.load(f)

        else:
            Logger.info(Model.getLogDictInfo(__class__.__name__, __name__, 'getDecomposition'),
                        'Factor Components Exists. Loading Pickle Backup.', logger)

            symbols, rewardArr, anchor_date = Model.decomposition(allSymbols, session, components = compnum,
                                                         rewardName = target, anchor_symbol = 'SPY',
                                                         getFactorRes = False)
            with open('~/stocks/factorRes.pkl','rb') as f:
                factorRes = pkl.load(f)

        return symbols, rewardArr, anchor_date, factorRes