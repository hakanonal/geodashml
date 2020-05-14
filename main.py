from environment import environment

config = {
    'discount': 0.95,
    'exploration_rate': 0.9,
    'decay_factor': 0.9999,
    'learning_rate': 0.1,
    'episode': 10,
    'hide_browser' : 1
}

e = environment(config=config)
try:
    e.start()
finally:
    e.end()