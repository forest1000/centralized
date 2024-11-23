import subprocess
for i in range(3,4):
    run_name = f"Fixmatch_with_adamw_unseen{i}_0.99_eps_1e-7_decay_0.01"
    config = "/media/morikawa/DataHDD/home/centralized/configs/spinal/run_conf.yaml"
    command = [
        "python3", "main.py",
        "--trainer", "Fixmatch",
        "--run_name", run_name,
        "--unseen_client",  str(i),
        "--config", config,
        "--ema", 'False'
    ]
    subprocess.run(command)