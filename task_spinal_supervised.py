import subprocess
for i in range(3,4):
    run_name = f"supervised_unseen{i}_adamw"
    config = "/media/morikawa/DataHDD/home/centralized/configs/spinal/run_conf.yaml"
    
    command = [
        "python3", "main.py",
        "--trainer", "supervised",
        "--run_name", run_name,
        "--unseen_client", str(i),
        "--config", config
    ]

    subprocess.run(command)