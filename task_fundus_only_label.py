import subprocess
for i in range(2,5):
    run_name = f"only_label_unseen{i}"
    config = "/media/morikawa/DataHDD/home/centralized/configs/fundus/run_conf.yaml"
    command = [
        "python3", "main.py",
        "--trainer", "only_label",
        "--run_name", run_name,
        "--unseen_client", str(i),
        "--config", config
    ]
    subprocess.run(command)