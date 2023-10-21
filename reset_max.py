import json

max_reward_models = json.load(open('max_reward_models.json','r'))

for key in max_reward_models:
	max_reward_models[key] = 0.

json.dump(max_reward_models, open('max_reward_models.json','w'))
