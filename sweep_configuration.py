sweep_config = {
            "method": "grid",  # Use Bayesian optimization for hyperparameter tuning
            "metric": {"name": "avg_reward_100ep", "goal": "maximize"},
            "parameters": {
                "batch_size":{"values": [128, 256]},
                "embedding_loss_coeff":{"values":[0, 0.2, 0.4, 0.8]},
                "embedding":{"values":["energy", "vanilla"]},
                "env":{"values":["LunarLander-v2", "CartPole-v0", "CartPole-v1", "Acrobot-v1"]},
                "use_log":{"values":[True, False]}
            }
        }