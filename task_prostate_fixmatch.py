import subprocess
for i in range(1,7):
    run_name = f"Fixmatch_unseen{i}"
    config = "/media/morikawa/DataHDD/home/centralized/configs/prostate/run_conf.yaml"
    command = [
        "python3", "main.py",
        "--trainer", "fixmatch",
        "--run_name", run_name,
        "--unseen_client",  str(i),
        "--config", config
    ]
    subprocess.run(command)