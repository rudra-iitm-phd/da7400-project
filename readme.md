# Basic Instructions

Run the `requirements.sh` to install gymnasium, wandb, swig and tqdm 

Please download and install pytorch as per your system specifications

## Run a wandb sweep

If you're initating a new sweep, then run

```bash
python main.py --wandb_sweep 
```

If you're trying to dump the results into a specific sweep with a `sweep_id`, then run

```bash
python main.py --wandb_sweep --sweep_id da24d008-iit-madras/da7400-test/<sweep-id>
```

## Configuring the sweep

If you're trying to log some specific metrics/parameter values please look into the following files

- `argument_parser.py` --> Add your custom arguments

- `sweep_configuration.py` --> make sure to give wandb the access to tune the particular parameters

- `main.py` --> look for `args` and make sure the arguments are properly captured

- `base_agent.py` --> If you desired to log a particular metric, make sure you're logging it. 
