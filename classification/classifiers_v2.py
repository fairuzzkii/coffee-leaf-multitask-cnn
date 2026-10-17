import os
import time
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from architectures.multitask_model import build_model
from deep_training import ModelTraining
from utils.utils import create_results_folder


def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) for boxes in [cx, cy, w, h] format."""
    # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
    xA1 = boxA[:, 0] - boxA[:, 2] / 2
    yA1 = boxA[:, 1] - boxA[:, 3] / 2
    xA2 = boxA[:, 0] + boxA[:, 2] / 2
    yA2 = boxA[:, 1] + boxA[:, 3] / 2

    xB1 = boxB[:, 0] - boxB[:, 2] / 2
    yB1 = boxB[:, 1] - boxB[:, 3] / 2
    xB2 = boxB[:, 0] + boxB[:, 2] / 2
    yB2 = boxB[:, 1] + boxB[:, 3] / 2

    # Coordinates of intersection rectangle
    x_inter1 = torch.max(xA1, xB1)
    y_inter1 = torch.max(yA1, yB1)
    x_inter2 = torch.min(xA2, xB2)
    y_inter2 = torch.min(yA2, yB2)

    inter_w = torch.clamp(x_inter2 - x_inter1, min=0)
    inter_h = torch.clamp(y_inter2 - y_inter1, min=0)
    inter_area = inter_w * inter_h

    areaA = torch.clamp(xA2 - xA1, min=0) * torch.clamp(yA2 - yA1, min=0)
    areaB = torch.clamp(xB2 - xB1, min=0) * torch.clamp(yB2 - yB1, min=0)
    union_area = areaA + areaB - inter_area

    iou = inter_area / (union_area + 1e-8)
    return iou


class MultiTask3HeadClassifier(ModelTraining):
    """Classifier for 3-head Multi-Task CNN (Disease, Severity, Localization)."""

    def __init__(
        self,
        csv_file,
        fold=1,
        backbone="mobilenet",
        num_disease_classes=5,
        batch_size=24,
        epochs=80,
        pretrained=True,
        optimizer="sgd",
        learning_rate=0.01,
        weight_decay=5e-4,
        lambda_disease=1.0,
        lambda_severity=1.0,
        lambda_bbox=1.0,
        results_path="results",
        experiment_name="multitask_3head",
    ):
        super(MultiTask3HeadClassifier, self).__init__()
        self.csv_file = csv_file
        self.fold = fold
        self.backbone = backbone.lower()
        self.num_disease_classes = num_disease_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.pretrained = pretrained
        self.optimizer_name = optimizer.lower()
        self.lr = learning_rate
        self.weight_decay = weight_decay

        # Loss weights
        self.lambda_dis = lambda_disease
        self.lambda_sev = lambda_severity
        self.lambda_bbox = lambda_bbox

        self.experiment_name = experiment_name
        self.results_folder = create_results_folder(results_path, experiment_name)

    def train_epoch(self, loader, model, criterion_dis, criterion_sev, criterion_bbox, optimizer):
        model.train()
        total_loss = 0.0
        total_samples = 0

        dis_correct = 0
        sev_abs_err = 0.0
        sev_count = 0
        bbox_ious = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pbar = tqdm(loader, desc="Train", leave=False)
        for images, labels_dis, labels_sev, labels_bbox, mask_sev, mask_bbox in pbar:
            images = images.to(device)
            labels_dis = labels_dis.to(device)
            labels_sev = labels_sev.to(device)
            labels_bbox = labels_bbox.to(device)
            mask_sev = mask_sev.to(device)
            mask_bbox = mask_bbox.to(device)

            batch_len = images.size(0)

            # Forward pass
            out_dis, out_sev, out_bbox = model(images)

            # 1. Disease Loss (Cross Entropy)
            loss_dis = criterion_dis(out_dis, labels_dis)

            # 2. Severity Loss (MSE, masked for unlabelled)
            loss_sev_raw = criterion_sev(out_sev, labels_sev)
            loss_sev = (loss_sev_raw * mask_sev.unsqueeze(1)).sum() / (mask_sev.sum() + 1e-8)

            # 3. Bbox Loss (Smooth L1, masked)
            loss_bbox_raw = criterion_bbox(out_bbox, labels_bbox)
            loss_bbox = (loss_bbox_raw * mask_bbox.unsqueeze(1)).sum() / (mask_bbox.sum() * 4 + 1e-8)

            # Combined Loss
            loss = (
                self.lambda_dis * loss_dis
                + self.lambda_sev * loss_sev
                + self.lambda_bbox * loss_bbox
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # Tracking
            total_loss += loss.item() * batch_len
            total_samples += batch_len

            # Disease Acc
            preds_dis = torch.argmax(out_dis, dim=1)
            dis_correct += (preds_dis == labels_dis).sum().item()

            # Severity MAE
            if mask_sev.sum() > 0:
                valid_idx = mask_sev > 0.5
                sev_abs_err += torch.abs(out_sev[valid_idx] - labels_sev[valid_idx]).sum().item()
                sev_count += valid_idx.sum().item()

            # Bbox IoU
            ious = calculate_iou(out_bbox, labels_bbox)
            bbox_ious.extend(ious.detach().cpu().numpy().tolist())

            cur_dis_acc = 100.0 * dis_correct / total_samples
            cur_sev_mae = (sev_abs_err / sev_count) if sev_count > 0 else 0.0
            pbar.set_description(
                f"Loss: {loss.item():.3f} | DisACC: {cur_dis_acc:.1f}% | SevMAE: {cur_sev_mae:.3f}"
            )

        mean_loss = total_loss / total_samples
        dis_acc = 100.0 * dis_correct / total_samples
        sev_mae = (sev_abs_err / sev_count) if sev_count > 0 else 0.0
        mean_iou = float(np.mean(bbox_ious)) if len(bbox_ious) > 0 else 0.0

        return {
            "loss": mean_loss,
            "disease_acc": dis_acc,
            "severity_mae": sev_mae,
            "bbox_iou": mean_iou,
        }

    def evaluate(self, loader, model, criterion_dis, criterion_sev, criterion_bbox):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total_loss = 0.0
        total_samples = 0

        y_true_dis, y_pred_dis = [], []
        y_true_sev, y_pred_sev = [], []
        bbox_ious = []

        start_time = time.time()

        with torch.no_grad():
            for images, labels_dis, labels_sev, labels_bbox, mask_sev, mask_bbox in tqdm(
                loader, desc="Eval", leave=False
            ):
                images = images.to(device)
                labels_dis = labels_dis.to(device)
                labels_sev = labels_sev.to(device)
                labels_bbox = labels_bbox.to(device)
                mask_sev = mask_sev.to(device)
                mask_bbox = mask_bbox.to(device)

                batch_len = images.size(0)

                out_dis, out_sev, out_bbox = model(images)

                loss_dis = criterion_dis(out_dis, labels_dis)
                loss_sev_raw = criterion_sev(out_sev, labels_sev)
                loss_sev = (loss_sev_raw * mask_sev.unsqueeze(1)).sum() / (mask_sev.sum() + 1e-8)
                loss_bbox_raw = criterion_bbox(out_bbox, labels_bbox)
                loss_bbox = (loss_bbox_raw * mask_bbox.unsqueeze(1)).sum() / (mask_bbox.sum() * 4 + 1e-8)

                loss = (
                    self.lambda_dis * loss_dis
                    + self.lambda_sev * loss_sev
                    + self.lambda_bbox * loss_bbox
                )

                total_loss += loss.item() * batch_len
                total_samples += batch_len

                preds_dis = torch.argmax(out_dis, dim=1).cpu().numpy()
                y_pred_dis.extend(preds_dis.tolist())
                y_true_dis.extend(labels_dis.cpu().numpy().tolist())

                # Severity
                valid_mask = (mask_sev > 0.5).cpu().numpy()
                if valid_mask.sum() > 0:
                    y_pred_sev.extend(out_sev.squeeze(1).cpu().numpy()[valid_mask].tolist())
                    y_true_sev.extend(labels_sev.squeeze(1).cpu().numpy()[valid_mask].tolist())

                ious = calculate_iou(out_bbox, labels_bbox)
                bbox_ious.extend(ious.cpu().numpy().tolist())

        total_eval_time = time.time() - start_time
        latency_ms_per_image = (total_eval_time / total_samples) * 1000.0 if total_samples > 0 else 0.0

        mean_loss = total_loss / total_samples
        dis_acc = accuracy_score(y_true_dis, y_pred_dis) * 100.0
        dis_f1 = f1_score(y_true_dis, y_pred_dis, average="macro") * 100.0

        sev_mae = mean_absolute_error(y_true_sev, y_pred_sev) if len(y_true_sev) > 0 else 0.0
        sev_rmse = np.sqrt(mean_squared_error(y_true_sev, y_pred_sev)) if len(y_true_sev) > 0 else 0.0
        mean_iou = float(np.mean(bbox_ious)) if len(bbox_ious) > 0 else 0.0

        return {
            "loss": mean_loss,
            "disease_acc": dis_acc,
            "disease_f1": dis_f1,
            "severity_mae": sev_mae,
            "severity_rmse": sev_rmse,
            "bbox_iou": mean_iou,
            "latency_ms": latency_ms_per_image,
        }

    def run_training(self, data_loader_fn):
        print(f"\n==========================================")
        print(f"Starting Multi-Task 3-Head Training")
        print(f"Backbone: {self.backbone} | Epochs: {self.epochs} | Batch: {self.batch_size}")
        print(f"Lambdas: Dis={self.lambda_dis}, Sev={self.lambda_sev}, Bbox={self.lambda_bbox}")
        print(f"Results Folder: {self.results_folder}")
        print(f"==========================================\n")

        # 1. Prepare Data
        train_loader, val_loader, test_loader = data_loader_fn(
            images_dir=None,
            batch_size=self.batch_size,
            balanced_dataset=False,
            csv_file=self.csv_file,
            fold=self.fold,
            model_task=None,
        )

        # 2. Build Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(
            backbone=self.backbone,
            mode="multi",
            num_disease_classes=self.num_disease_classes,
            pretrained=self.pretrained,
        ).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,} (Trainable: {trainable_params:,})")

        # 3. Criteria & Optimizer
        criterion_dis = nn.CrossEntropyLoss()
        criterion_sev = nn.MSELoss(reduction="none")
        criterion_bbox = nn.SmoothL1Loss(reduction="none")

        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )

        logs = {
            "config": {
                "backbone": self.backbone,
                "mode": "multi",
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "optimizer": self.optimizer_name,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "lambda_dis": self.lambda_dis,
                "lambda_sev": self.lambda_sev,
                "lambda_bbox": self.lambda_bbox,
                "total_params": total_params,
            },
            "history": [],
        }

        best_score = -float("inf")
        best_epoch = -1

        for epoch in range(self.epochs):
            t0 = time.time()
            train_m = self.train_epoch(
                train_loader, model, criterion_dis, criterion_sev, criterion_bbox, optimizer
            )
            val_m = self.evaluate(
                val_loader, model, criterion_dis, criterion_sev, criterion_bbox
            )
            epoch_time = time.time() - t0

            # Learning rate decay
            optimizer = self.adjust_learning_rate(
                optimizer, self.optimizer_name, epoch, self.epochs
            )

            # Composite validation score (higher is better):
            # 60% Disease Acc + 20% (1 - Sev MAE)*100 + 20% (Bbox IoU * 100)
            score = (
                0.6 * val_m["disease_acc"]
                + 0.2 * max(0.0, (1.0 - val_m["severity_mae"])) * 100.0
                + 0.2 * (val_m["bbox_iou"] * 100.0)
            )

            print(
                f"[Epoch {epoch+1:03d}/{self.epochs:03d}] "
                f"Train Loss: {train_m['loss']:.3f} (DisACC: {train_m['disease_acc']:.1f}%) | "
                f"Val DisACC: {val_m['disease_acc']:.1f}% (F1: {val_m['disease_f1']:.1f}%) | "
                f"Val SevMAE: {val_m['severity_mae']:.3f} | "
                f"Val BboxIoU: {val_m['bbox_iou']:.3f} | Time: {epoch_time:.1f}s"
            )

            epoch_record = {
                "epoch": epoch + 1,
                "train": train_m,
                "val": val_m,
                "score": score,
            }
            logs["history"].append(epoch_record)

            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                torch.save(
                    model.state_dict(), os.path.join(self.results_folder, "best_weights.pth")
                )
                print(f"   --> Best model saved at epoch {epoch+1} (Score: {score:.2f})")

            # Save latest log
            with open(os.path.join(self.results_folder, "train_logs.json"), "w") as fp:
                json.dump(logs, fp, indent=2)

        # 4. Final Evaluation on Test Set with Best Weights
        print("\nLoading best weights for final test evaluation...")
        best_weights_path = os.path.join(self.results_folder, "best_weights.pth")
        if os.path.exists(best_weights_path):
            model.load_state_dict(torch.load(best_weights_path, map_location=device))

        test_m = self.evaluate(test_loader, model, criterion_dis, criterion_sev, criterion_bbox)
        print("\n================ FINAL TEST RESULTS ================")
        print(f"Disease Accuracy : {test_m['disease_acc']:.2f}%")
        print(f"Disease Macro-F1 : {test_m['disease_f1']:.2f}%")
        print(f"Severity MAE     : {test_m['severity_mae']:.4f}")
        print(f"Severity RMSE    : {test_m['severity_rmse']:.4f}")
        print(f"Bbox Mean IoU    : {test_m['bbox_iou']:.4f}")
        print(f"Inference Latency: {test_m['latency_ms']:.2f} ms/image")
        print(f"===================================================\n")

        logs["test_results"] = test_m
        logs["best_epoch"] = best_epoch
        with open(os.path.join(self.results_folder, "train_logs.json"), "w") as fp:
            json.dump(logs, fp, indent=2)

        return test_m

    # Base class abstract methods implementation stubs
    def train(self, *args, **kwargs):
        pass

    def validation(self, *args, **kwargs):
        pass

    def run_test(self, *args, **kwargs):
        pass


class SingleTaskV2Classifier(ModelTraining):
    """Classifier for Single-Task baseline models for comparative study in thesis."""

    def __init__(
        self,
        csv_file,
        task="disease",
        backbone="mobilenet",
        num_disease_classes=5,
        batch_size=24,
        epochs=80,
        pretrained=True,
        optimizer="sgd",
        learning_rate=0.01,
        weight_decay=5e-4,
        results_path="results",
        experiment_name=None,
    ):
        super(SingleTaskV2Classifier, self).__init__()
        self.csv_file = csv_file
        self.task = task.lower()
        self.backbone = backbone.lower()
        self.num_disease_classes = num_disease_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.pretrained = pretrained
        self.optimizer_name = optimizer.lower()
        self.lr = learning_rate
        self.weight_decay = weight_decay

        if experiment_name is None:
            experiment_name = f"singletask_{self.task}_{self.backbone}"
        self.experiment_name = experiment_name
        self.results_folder = create_results_folder(results_path, experiment_name)

    def run_training(self, data_loader_fn):
        print(f"\n==========================================")
        print(f"Starting Single-Task Training: {self.task.upper()}")
        print(f"Backbone: {self.backbone} | Epochs: {self.epochs} | Batch: {self.batch_size}")
        print(f"Results Folder: {self.results_folder}")
        print(f"==========================================\n")

        train_loader, val_loader, test_loader = data_loader_fn(
            images_dir=None,
            batch_size=self.batch_size,
            balanced_dataset=False,
            csv_file=self.csv_file,
            fold=1,
            model_task=None,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(
            backbone=self.backbone,
            mode="single",
            single_task=self.task,
            num_disease_classes=self.num_disease_classes,
            pretrained=self.pretrained,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Single-Task Parameters: {total_params:,}")

        if self.task == "disease":
            criterion = nn.CrossEntropyLoss()
        elif self.task == "severity":
            criterion = nn.MSELoss(reduction="none")
        else:
            criterion = nn.SmoothL1Loss(reduction="none")

        optimizer = (
            torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            if self.optimizer_name == "adam"
            else torch.optim.SGD(
                model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        )

        best_metric = -float("inf") if self.task == "disease" else float("inf")

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            total_samples = 0

            for images, labels_dis, labels_sev, labels_bbox, mask_sev, mask_bbox in tqdm(
                train_loader, desc=f"Train Ep {epoch+1}", leave=False
            ):
                images = images.to(device)
                b_size = images.size(0)

                if self.task == "disease":
                    labels = labels_dis.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                elif self.task == "severity":
                    labels = labels_sev.to(device)
                    mask = mask_sev.to(device)
                    outputs = model(images)
                    raw_loss = criterion(outputs, labels)
                    loss = (raw_loss * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-8)
                else:
                    labels = labels_bbox.to(device)
                    mask = mask_bbox.to(device)
                    outputs = model(images)
                    raw_loss = criterion(outputs, labels)
                    loss = (raw_loss * mask.unsqueeze(1)).sum() / (mask.sum() * 4 + 1e-8)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                train_loss += loss.item() * b_size
                total_samples += b_size

            optimizer = self.adjust_learning_rate(optimizer, self.optimizer_name, epoch, self.epochs)
            train_loss = train_loss / total_samples

            # Validation
            model.eval()
            val_loss = 0.0
            val_total = 0
            correct = 0
            abs_err = 0.0
            count_sev = 0

            with torch.no_grad():
                for images, labels_dis, labels_sev, labels_bbox, mask_sev, mask_bbox in val_loader:
                    images = images.to(device)
                    b_size = images.size(0)

                    if self.task == "disease":
                        labels = labels_dis.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                    elif self.task == "severity":
                        labels = labels_sev.to(device)
                        mask = mask_sev.to(device)
                        outputs = model(images)
                        raw_loss = criterion(outputs, labels)
                        loss = (raw_loss * mask.unsqueeze(1)).sum() / (mask.sum() + 1e-8)
                        valid_idx = mask > 0.5
                        if valid_idx.sum() > 0:
                            abs_err += torch.abs(outputs[valid_idx] - labels[valid_idx]).sum().item()
                            count_sev += valid_idx.sum().item()
                    else:
                        labels = labels_bbox.to(device)
                        mask = mask_bbox.to(device)
                        outputs = model(images)
                        raw_loss = criterion(outputs, labels)
                        loss = (raw_loss * mask.unsqueeze(1)).sum() / (mask.sum() * 4 + 1e-8)

                    val_loss += loss.item() * b_size
                    val_total += b_size

            val_loss = val_loss / val_total
            if self.task == "disease":
                val_metric = 100.0 * correct / val_total
                print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val ACC: {val_metric:.2f}%")
                if val_metric > best_metric:
                    best_metric = val_metric
                    torch.save(model.state_dict(), os.path.join(self.results_folder, "best_weights.pth"))
            elif self.task == "severity":
                val_metric = (abs_err / count_sev) if count_sev > 0 else 0.0
                print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val MAE: {val_metric:.4f}")
                if val_metric < best_metric:
                    best_metric = val_metric
                    torch.save(model.state_dict(), os.path.join(self.results_folder, "best_weights.pth"))
            else:
                print(f"[Epoch {epoch+1:03d}] Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
                if val_loss < best_metric:
                    best_metric = val_loss
                    torch.save(model.state_dict(), os.path.join(self.results_folder, "best_weights.pth"))

        print(f"Training finished. Best weights saved to {self.results_folder}/best_weights.pth")

    def train(self, *args, **kwargs):
        pass

    def validation(self, *args, **kwargs):
        pass

    def run_test(self, *args, **kwargs):
        pass
