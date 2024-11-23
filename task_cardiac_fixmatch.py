import subprocess
for i in range(1,2):
    run_name = f"Fixmatch_unseen{i}_0.99"
    config = "/media/morikawa/DataHDD/home/centralized/configs/cardiac/run_conf.yaml"
    command = [
        "python3", "main.py",
        "--trainer", "fixmatch",
        "--run_name", run_name,
        "--unseen_client",  str(i),
        "--config", config,
        "--ema", 'False'
    ]
    subprocess.run(command)