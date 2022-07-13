import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from sklearn.model_selection import KFold


def create_datalist(
    dataset_input_dir: str,
    output_dir: str,
    task_id: str,
    num_folds: int,
    seed: int,
):
    task_name = {
        "500": "Task500_BraTS2021"
        
    }

    dataset_file_path = os.path.join(
        dataset_input_dir, task_name[task_id], "dataset.json"
    )

    with open(dataset_file_path, "r") as f:
        dataset = json.load(f)

    dataset_with_folds = dataset.copy()

    keys = [line["image"].split("/")[-1].split(".")[0] for line in dataset["training"]]
    
    training_list=[{'image':[line["image"][:-7]+f'_000{x}.nii.gz' for x in [0,1,2,3]],'label':line['label']} for line in dataset["training"]] 
    dataset_train_dict = dict(zip(keys, training_list))
    all_keys_sorted = np.sort(keys)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
        val_data = []
        train_data = []
        train_keys = np.array(all_keys_sorted)[train_idx]
        test_keys = np.array(all_keys_sorted)[test_idx]
        for key in test_keys:
            val_data.append(dataset_train_dict[key])
        for key in train_keys:
            train_data.append(dataset_train_dict[key])

        dataset_with_folds["validation_fold{}".format(i)] = val_data
        dataset_with_folds["train_fold{}".format(i)] = train_data
        print(train_data)
    del dataset

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(
        os.path.join(output_dir, "dataset_task{}.json".format(task_id)), "w"
    ) as f:
        json.dump(dataset_with_folds, f)
        print("data list for {} has been created!".format(task_name[task_id]))
        f.close()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-input_dir", "--input_dir", type=str, default="./"
    )
    parser.add_argument("-output_dir", "--output_dir", type=str, default="config/")
    parser.add_argument(
        "-task_id", "--task_id", type=str, default="500", help="task id"
    )
    parser.add_argument(
        "-num_folds", "--num_folds", type=int, default=5, help="number of folds"
    )
    parser.add_argument("-seed", "--seed", type=int, default=12345, help="seed number")

    args = parser.parse_args()

    create_datalist(
        dataset_input_dir=args.input_dir,
        output_dir=args.output_dir,
        task_id=args.task_id,
        num_folds=args.num_folds,
        seed=args.seed,
    )