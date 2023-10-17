from tensorly.decomposition import parafac
from tensorflow.keras import layers
from dataclasses import dataclass
from ..util.utils import Utility
from tensorflow import keras
import tensorly as tl
import random

@dataclass(frozen=True, unsafe_hash=True)
class Model:
    '''
        Static method to run the decomposition to get component level features.
    '''
    @staticmethod
    def Decomposition(allSymbols, session, components = 10, rewardName = 'GOOG', anchor_symbol = 'SPY', getFactorRes = False):
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