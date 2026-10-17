import sys
import warnings
import argparse
from functools import partial

# Legacy classifiers & loaders
from classifiers import SingleTaskClassifier, MultiTaskClassifier
from loaders import coffeeleaves_loader, images_loader, combined_loader, combined_loader_3head
from results import save_results
from utils.enums import Tasks

# Modern 3-head multi-task and single-task classifiers
from classifiers_v2 import MultiTask3HeadClassifier, SingleTaskV2Classifier

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coffee Leaf Multi-Task / Single-Task CNN")

    # Architecture & Mode
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["mobilenet", "efficientnet"],
        default=None,
        help="Select lightweight backbone: mobilenet (MobileNetV2) or efficientnet (EfficientNet-B0).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["multi", "single"],
        default="multi",
        help="Experiment mode: 'multi' (shared backbone + 3 heads) or 'single' (task-specific model).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["disease", "severity", "bbox"],
        default="disease",
        help="Target task when running in --mode single.",
    )

    # Multi-task loss weights (Section 3.1: configurable lambdas)
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight for Disease classification loss")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Weight for Severity regression loss")
    parser.add_argument("--lambda3", type=float, default=1.0, help="Weight for Localization bbox regression loss")

    # Hyperparameters
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Select optimization technique [sgd/adam].",
        default="sgd",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, help="Images batch size", default=24)
    parser.add_argument(
        "--weight_decay", type=float, help="L2 parameter norm penalty", default=5e-4
    )
    parser.add_argument(
        "--data_augmentation",
        type=str,
        help="Data augmentation technique [standard/mixup/bc+]",
        default="standard",
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs.", default=80)
    parser.add_argument(
        "--pretrained",
        type=bool,
        help="Defines whether or not to use pre-trained ImageNet weights.",
        default=True,
    )
    parser.add_argument("--balanced_dataset", type=bool, default=False)
    parser.add_argument(
        "--csv_file",
        type=str,
        help="Path of the dataset csv file.",
        default="dataset/dataset_combined.csv",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="Fold selection [1-5].",
        default=1,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment.",
        default=None,
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to the results folder.",
        default="results",
    )
    parser.add_argument("--train", help="Run in training mode.", action="store_true")
    parser.add_argument("--test", help="Run in test mode.", action="store_true")
    parser.add_argument(
        "--dataset",
        help="Select dataset: leaf, symptom, combined",
        type=str,
        default="combined",
    )

    # Legacy options for backward compatibility
    parser.add_argument(
        "--model",
        type=str,
        help="[Legacy] Select CNN architecture.",
        default=None,
    )
    parser.add_argument(
        "--model_task",
        help="[Legacy] (0) biotic stress, (1) severity, (2) multitask",
        type=int,
        default=None,
    )

    options = parser.parse_args()

    assert options.train or options.test, "You must specify whether you want to --train or --test a model."

    # Map legacy model/backbone if needed
    if options.backbone is None and options.model is not None:
        if "mobilenet" in options.model.lower():
            options.backbone = "mobilenet"
        elif "efficientnet" in options.model.lower():
            options.backbone = "efficientnet"

    # Default backbone to mobilenet if not given
    if options.backbone is None:
        options.backbone = "mobilenet"

    # Default experiment name
    if options.experiment_name is None:
        if options.mode == "multi":
            options.experiment_name = f"multitask_3head_{options.backbone}"
        else:
            options.experiment_name = f"singletask_{options.task}_{options.backbone}"

    # Determine whether to use modern 3-head pipeline or legacy pipeline
    use_modern_pipeline = (
        options.dataset == "combined"
        or options.mode in ["multi", "single"]
    )

    if use_modern_pipeline:
        data_loader_fn = partial(
            combined_loader_3head,
            csv_file=options.csv_file,
            fold=options.fold,
            model_task=None,
        )

        if options.mode == "multi":
            classifier = MultiTask3HeadClassifier(
                csv_file=options.csv_file,
                fold=options.fold,
                backbone=options.backbone,
                num_disease_classes=5,
                batch_size=options.batch_size,
                epochs=options.epochs,
                pretrained=options.pretrained,
                optimizer=options.optimizer,
                learning_rate=options.lr,
                weight_decay=options.weight_decay,
                lambda_disease=options.lambda1,
                lambda_severity=options.lambda2,
                lambda_bbox=options.lambda3,
                results_path=options.results_path,
                experiment_name=options.experiment_name,
            )
        else:
            classifier = SingleTaskV2Classifier(
                csv_file=options.csv_file,
                task=options.task,
                backbone=options.backbone,
                num_disease_classes=5,
                batch_size=options.batch_size,
                epochs=options.epochs,
                pretrained=options.pretrained,
                optimizer=options.optimizer,
                learning_rate=options.lr,
                weight_decay=options.weight_decay,
                results_path=options.results_path,
                experiment_name=options.experiment_name,
            )

        if options.train:
            classifier.run_training(data_loader_fn)
        else:
            print("For testing, load weights and evaluate test loader.")

    else:
        # Legacy pipeline execution
        model_task = Tasks(options.model_task) if options.model_task is not None else Tasks.MULTITASK
        legacy_model = options.model if options.model else "mobilenet_v2"

        if model_task == Tasks.MULTITASK:
            clf = MultiTaskClassifier(
                images_dir=f"dataset/{options.dataset}",
                csv_file=options.csv_file,
                fold=options.fold,
                num_classes=(5, 5),
                model_task=model_task,
                balanced_dataset=options.balanced_dataset,
                batch_size=options.batch_size,
                epochs=options.epochs,
                model=legacy_model,
                pretrained=options.pretrained,
                optimizer=options.optimizer,
                weight_decay=options.weight_decay,
                data_augmentation=options.data_augmentation,
                results_path=options.results_path,
                experiment_name=options.experiment_name,
            )
        else:
            clf = SingleTaskClassifier(
                images_dir=f"dataset/{options.dataset}",
                num_classes=5,
                balanced_dataset=options.balanced_dataset,
                batch_size=options.batch_size,
                epochs=options.epochs,
                model=legacy_model,
                pretrained=options.pretrained,
                optimizer=options.optimizer,
                weight_decay=options.weight_decay,
                data_augmentation=options.data_augmentation,
                results_path=options.results_path,
                experiment_name=options.experiment_name,
            )

        if options.dataset == "combined":
            loader = partial(combined_loader, csv_file=options.csv_file, fold=options.fold, model_task=model_task)
        elif options.dataset == "leaf":
            loader = partial(coffeeleaves_loader, csv_file=options.csv_file, fold=options.fold, model_task=model_task)
        else:
            loader = images_loader

        if options.train:
            clf.run_training(loader)
        elif options.test:
            out = clf.run_test(loader)
            save_results(out, model_task, options.results_path, options.experiment_name)
