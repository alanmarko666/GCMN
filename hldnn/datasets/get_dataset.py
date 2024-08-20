from config import config
import datasets.line_dataset as line_dataset
import datasets.synthetic.gen_tree_tasks as gen_tree_tasks
import datasets.molecules as molecules


def get_dataset():
    train_dataset, test_datasets, valid_dataset = None,None,None
    if config.DATASET.startswith("treetask") or config.DATASET.startswith("graphtask"):
        name = 'data/{}'.format(config.DATASET)
        generator = {
            "treetask1": gen_tree_tasks.generate_tree_task_1_data,
            "treetask2": gen_tree_tasks.generate_tree_task_2_data,
            "treetask3": gen_tree_tasks.generate_tree_task_3_data,
            "treetask4": gen_tree_tasks.generate_tree_task_4_data,
            "treetask5": gen_tree_tasks.generate_tree_task_5_data,
            "treetask6": gen_tree_tasks.generate_tree_task_6_data,
            "treetask7": gen_tree_tasks.generate_tree_task_7_data,
            "treetask8": gen_tree_tasks.generate_tree_task_8order_data,
            "graphtask1": gen_tree_tasks.generate_graph_task_1_data,
            "graphtask2": gen_tree_tasks.generate_graph_task_2_data,
            #"graphtask3": gen_tree_tasks.generate_graph_task_3_data,
            }[config.DATASET]

        train_dataset=gen_tree_tasks.get_tree_dataset(name="{}-train".format(name),
            count=config.DATASET_TRAIN_SIZE,
            min_n=config.DATASET_TRAIN_MIN_N,
            max_n=config.DATASET_TRAIN_MAX_N,
            generator=generator,
            force_download=config.DATASET_FORCE_DOWNLOAD)

        valid_dataset=gen_tree_tasks.get_tree_dataset(name="{}-valid".format(name),
            count=config.DATASET_VALID_SIZE,
            min_n=config.DATASET_VALID_MIN_N,
            max_n=config.DATASET_VALID_MAX_N,
            generator=generator,
            force_download=config.DATASET_FORCE_DOWNLOAD)

        test_datasets=[ \
            gen_tree_tasks.get_tree_dataset(name="{}-test".format(name),
                count=size, min_n=min_n, max_n=max_n,
                generator=generator,
                force_download=config.DATASET_FORCE_DOWNLOAD) \
            for (size, min_n, max_n) in config.DATASET_TESTS \
        ]
    elif config.DATASET.startswith("aqsol"):
        train_dataset=molecules.get_aqsol_tree_dataset(split="train")
        valid_dataset=molecules.get_aqsol_tree_dataset(split="val")
        test_datasets=[molecules.get_aqsol_tree_dataset(split="test")]
        
    elif config.DATASET.startswith("zinc"):
        train_dataset=molecules.get_zinc_tree_dataset(split="train")
        valid_dataset=molecules.get_zinc_tree_dataset(split="val")
        test_datasets=[molecules.get_zinc_tree_dataset(split="test")]
    elif config.DATASET == "mutag":
        train_dataset=molecules.get_mutag_tree_dataset(split="train")
        valid_dataset=molecules.get_mutag_tree_dataset(split="val")
        test_datasets=[molecules.get_mutag_tree_dataset(split="test")]
    elif config.DATASET == "qm9":
        train_dataset=molecules.get_qm9_tree_dataset(split="train")
        valid_dataset=molecules.get_qm9_tree_dataset(split="val")
        test_datasets=[molecules.get_qm9_tree_dataset(split="test")]
    elif config.DATASET == "esol":
        train_dataset=molecules.get_esol_tree_dataset(split="train")
        valid_dataset=molecules.get_esol_tree_dataset(split="val")
        test_datasets=[molecules.get_esol_tree_dataset(split="test")]
    elif config.DATASET == "molhiv":
        train_dataset=molecules.get_molhiv_tree_dataset(split="train")
        valid_dataset=molecules.get_molhiv_tree_dataset(split="valid")
        test_datasets=[molecules.get_molhiv_tree_dataset(split="test")]
    elif config.DATASET == "pepstruct":
        train_dataset=molecules.get_peptides_structural_tree_dataset(split="train")
        valid_dataset=molecules.get_peptides_structural_tree_dataset(split="val")
        test_datasets=[molecules.get_peptides_structural_tree_dataset(split="test")]
    else:
        raise NotImplemented
    limit = config["TEST_SMALL_DATASET"]
    if limit is not None:
        return {"train": train_dataset[:limit], "valid": valid_dataset[:limit],"test": [td[:limit]for td in test_datasets]}
    return {"train": train_dataset, "valid": valid_dataset,"test": test_datasets}