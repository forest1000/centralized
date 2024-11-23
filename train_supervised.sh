export WANDB_API_KEY="f0ab93d54567cc8cc3fc5fff22fe87a1f12e7268"
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY is not set. Please set it before running the script."
    exit 1
fi
echo $WANDB_API_KEY | wandb login --relogin


python3 main.py --trainer supervised --run_name supervised --unseen_client 3 --config "/media/morikawa/DataHDD/home/centralized/configs/fundus/run_conf.yaml"