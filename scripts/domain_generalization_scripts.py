import os 
import pandas as pd


task_list = ["cardiac", "spinal"]

for task_name in task_list:
    #base directory
    base_dir = f"/media/morikawa/DataHDD/home/data/segmentation/{task_name}/semi"
    clients = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    #store directory
    output_base_dir = os.path.join(base_dir, "domain_generalization")
    os.makedirs(output_base_dir, exist_ok=True)
    
    RANDOM_SEED = 42
    
    for test_client in clients:
        if test_client == "domain_generalization":
            continue
        print(f'Processing test client: {test_client}')
        experiment_dir = os.path.join(output_base_dir, f"experiment_{test_client}")
        os.makedirs(experiment_dir, exist_ok=True)
        if task_name == 'spinal':
            test_csv_file = os.path.join(base_dir, test_client, "labeled_data.csv")
        else:
            test_csv_file = os.path.join(base_dir, test_client, "data.csv")
       
        
        if os.path.exists(test_csv_file): 
            
            test_df = pd.read_csv(test_csv_file)

        else:
            print(f"Warning: {test_csv_file} dose not exist")
            continue
        source_dfs = []
        unlabeled_source_dfs = []
        labeled_source_dfs = []

        #val_dfs = []

        # aggregate the data each client has other than test client
        for source_client in clients:
            if source_client == test_client:
                continue
            if source_client == "domain_generalization":
                continue
            if task_name == 'spinal':
                source_train_csv = os.path.join(base_dir, source_client, "labeled_data.csv")
            else:
                source_train_csv = os.path.join(base_dir, source_client, "data.csv")
            source_unlabeled_csv = os.path.join(base_dir, source_client, "unlabeled.csv")
            source_labeled_csv = os.path.join(base_dir, source_client, "labeled.csv")
            #source_val_csv = os.path.join(base_dir, source_client, "val.csv")


            # read data for train
            if os.path.exists(source_train_csv):
                df = pd.read_csv(source_train_csv)
                source_dfs.append(df)
            else:
                print(f"Warning: {source_train_csv} does not exist. Skipping this client.")

            # read data for unlabeled train
            if os.path.exists(source_unlabeled_csv):
                df = pd.read_csv(source_unlabeled_csv)
                unlabeled_source_dfs.append(df)
            else:
                print(f"Warning: No source unlabeled data found for experiment {test_client}")
            
            # read data labeled train
            if os.path.exists(source_labeled_csv):
                df = pd.read_csv(source_labeled_csv)
                labeled_source_dfs.append(df)
            else:
                print(f"Warning: No source unlabeled data found for experiment {test_client}")
            # for Val
            """
            if os.path.exists(source_val_csv):
                df = pd.read_csv(source_val_csv)
                labeled_source_dfs.append(df)
            else:
                print(f"Warning: No source validation data found for experiment {test_client}")
                """
        # concat and shaffle
        combined_train_df = pd.concat(source_dfs, ignore_index=True)
        combined_train_df = combined_train_df.sample(frac=1, random_state=RANDOM_SEED)
        
        combined_train_unlabeled_df = pd.concat(unlabeled_source_dfs, ignore_index=True)
        combined_train_unlabeled_df = combined_train_unlabeled_df.sample(frac=1, random_state = RANDOM_SEED)
        
        combined_train_labeled_df = pd.concat(labeled_source_dfs, ignore_index=True)
        combined_train_labeled_df = combined_train_labeled_df.sample(frac=1, random_state = RANDOM_SEED)
        """
        combined_val_df = pd.concat(val_dfs, ignore_index=True)
        combined_val_df = combined_val_df.sample(frac=1, random_state=RANDOM_SEED)
        """
        train_output_path = os.path.join(experiment_dir, "train.csv")
        unlabeled_output_path = os.path.join(experiment_dir, "unlabeled.csv")
        labeled_output_path = os.path.join(experiment_dir, "labeled.csv")
        """
        val_output_path = os.path.join(experiment_dir, "val.csv")
        """
        test_output_path = os.path.join(experiment_dir, "test.csv")
        
        combined_train_df.to_csv(train_output_path, index=False)
        combined_train_unlabeled_df.to_csv(unlabeled_output_path, index=False)
        combined_train_labeled_df.to_csv(labeled_output_path, index=False)
        """
        combined_val_df.to_csv(val_output_path, index=False)
        """
        test_df.to_csv(test_output_path, index=False)
        
        print(f"Saved shuffled train, unlabeled, labeled and test csv to {experiment_dir}\n")
        
print("All experiments processed.")