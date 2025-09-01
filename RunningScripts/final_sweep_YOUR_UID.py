import wandb
YOUR_WANDB_USERNAME = "diana-morgan"
project = "nlp_d_and_n"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "seed 7 for 17",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "ENV_Test_proba_accuracy_per_mean_user_and_bot"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["LSTM"]},
        "seed": {"values": list(range(7, 8))},
        "online_simulation_factor": {"values": [4]},
        "features": {"values": ["EFs"]},
        "basic_nature": {"values": [17]}
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
