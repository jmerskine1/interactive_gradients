import optax
import yaml

with open("config.yaml", "r") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            learning_rate = config[0]['hyperparams']['learning_rate']
            momentum = config[0]['hyperparams']['momentum']
            yamlfile.close()

sgd_opt = optax.sgd(learning_rate ,momentum)
adam_opt = optax.adam(learning_rate)