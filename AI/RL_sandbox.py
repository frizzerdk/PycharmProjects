M_CONFIG = {
    "learning_rate": [0.001, 0.002, 0.003],
    "batch_size": [32, 64, 128],
    "gamma": [0.95, 0.99, 0.999],
    "tau": [0.001, 0.01, 0.1],
    "sigma":1,
    "text":"test"

}

import itertools

def generate_combinations(M_CONFIG):
    # Get the default values
    default_values = [v[0] if isinstance(v, list) else v for v in M_CONFIG.values()]

    combinations = []
    # Iterate over each option and generate combinations where it doesn't use the default value
    for i, key in enumerate(M_CONFIG.keys()):
        if isinstance(M_CONFIG[key], list):
            for val in M_CONFIG[key][1:]:
                combo = list(default_values)
                combo[i] = val
                combinations.append(dict(zip(M_CONFIG.keys(), combo)))


    # Add the default combination
    combinations.append(dict(zip(M_CONFIG.keys(), default_values)))

    return combinations
for c in generate_combinations(M_CONFIG):
    print(c)

