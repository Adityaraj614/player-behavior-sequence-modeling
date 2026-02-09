ACTIONS = [
    "move",
    "shoot",
    "hide",
    "reload",
    "explore",
    "loot",
    "fight",
    "die",
    "win",
    "quit"
]

ACTION_TO_ID = {action: idx for idx, action in enumerate(ACTIONS)}
ID_TO_ACTION = {idx: action for action, idx in ACTION_TO_ID.items()}

# Add initial action space definition
