import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score)
from tap import Tap
from torch.utils.data import (DataLoader, TensorDataset,
                              WeightedRandomSampler)
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from dataset import get_dataset
from models import BNN,  LlamaToxicClassifier, Guardian, SafetyGuard


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.
    Focuses learning on hard examples by down-weighting easy ones.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        # p_t = probability of the correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # alpha_t = alpha for positive, (1-alpha) for negative
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


class Trainer(object):

    def __init__(self, args) -> None:
        self.args = args

        wandb.init(
            reinit=True,
            config=args.as_dict(),
            project=args.wandb_project,
            name=args.exp_name,
        )

        train_features = torch.load(args.train_features, map_location="cpu")
        train_labels = torch.load(args.train_labels, map_location="cpu").long()

        # Load augmentation rounds (skip if not found)
        for r in range(args.num_rounds):
            aug_feat_path = args.aug_features[r]
            aug_lbl_path = args.aug_labels[r]
            if os.path.exists(aug_feat_path) and os.path.exists(aug_lbl_path):
                train_features = torch.cat([train_features, torch.load(aug_feat_path, map_location="cpu")])
                train_labels = torch.cat([train_labels, torch.load(aug_lbl_path, map_location="cpu")])
                print(f"Loaded augmentation round {r}")
            else:
                print(f"Skipping augmentation round {r} (files not found)")

        val_features = torch.load(args.val_features, map_location="cpu")
        val_labels = torch.load(args.val_labels, map_location="cpu").long()

        # Print data stats
        n_train_pos = (train_labels == 1).sum().item()
        n_val_pos = (val_labels == 1).sum().item()
        print(f"\n[DATA] Train: {len(train_labels)} samples ({n_train_pos} pos, {len(train_labels)-n_train_pos} neg)")
        print(f"[DATA] Val:   {len(val_labels)} samples ({n_val_pos} pos, {len(val_labels)-n_val_pos} neg)")

        # Compute pos_weight for loss function based on training set imbalance
        n_neg = (train_labels == 0).sum().float()
        n_pos = (train_labels == 1).sum().float()
        self.pos_weight = (n_neg / n_pos).item()
        print(f"[DATA] pos_weight for loss: {self.pos_weight:.2f}\n")

        val_ds = TensorDataset(val_features, val_labels)
        self.val_loader = DataLoader(
            val_ds, args.batch_size, shuffle=False)

        train_ds = TensorDataset(train_features, train_labels)

        class_sample_count = torch.tensor(
            [(train_labels == t).sum() for t in torch.unique(train_labels, sorted=True)])
        weight = 1.0 / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in train_labels])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        self.tr_loader = DataLoader(train_ds, args.batch_size, sampler=sampler)

        self.device = torch.cuda.current_device()

        input_dim = train_features.size(1)
        self.model = BNN(input_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=0.01
        )

        # Cosine annealing with warm restarts every 50 epochs
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )

    def train(self):
        global_step = 1
        # Focal Loss to handle class imbalance better
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        
        best_metric_val = -1.0
        best_metrics = None
        patience = 100  # Early stopping patience
        no_improve_count = 0
        
        save_dir = os.path.join(
            f"save/{self.args.version}", "bnn_small")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = os.path.join(save_dir, "model.pt")

        for epoch in tqdm(range(self.args.epochs), dynamic_ncols=True):

            t = tqdm(
                self.tr_loader,
                total=len(self.tr_loader),
                leave=False,
                dynamic_ncols=True,
            )
            for batch in t:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                self.model.train()
                self.model.zero_grad()

                logits = self.model(x)
                loss = criterion(logits, y.float())

                kl_loss = self.model.get_kl()

                kl_weight = 1.0 / len(self.tr_loader.dataset)
                total_loss = loss + kl_weight * kl_loss
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.args.max_norm
                )

                self.optimizer.step()

                wandb.log({"loss/train": loss.item(),
                          "kl/train": kl_loss.item(),
                          "lr": self.optimizer.param_groups[0]['lr']}, step=global_step)
                num_ones = torch.sum(y).item()
                t.set_description(
                    f"epoch: {epoch}, step: {global_step}, loss: {loss.item(): .4f} kl: {kl_loss.item(): .4f},  num ones: {num_ones}"
                )
                global_step += 1

            # Step the cosine scheduler per epoch
            self.scheduler.step()

            if self.val_loader is not None:
                metrics = self.eval(self.val_loader, split="val")
                wandb.log(metrics, step=global_step)
                
                # Check for best model
                if metrics["auprc/val"] > best_metric_val:
                    best_metric_val = metrics["auprc/val"]
                    best_metrics = metrics
                    no_improve_count = 0
                    self.model.train()
                    ckpt = {
                        "state_dict": self.model.state_dict(),
                        "layer_idx": self.args.layer_idx
                    }
                    torch.save(ckpt, save_file)
                    print(f"--> Saved new best model at epoch {epoch} with AUPRC: {best_metric_val:.4f}")
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                        break

        print("\n" + "="*50)
        print("TRAINING COMPLETED. BEST VALIDATION RESULTS:")
        if best_metrics is not None:
            for k, v in best_metrics.items():
                print(f"  {k}: {v:.4f}")
        print("="*50 + "\n")

        if not self.args.skip_test:
            if str(self.args.version) == "dynaguard_1p7b_8b":
                self.test_on_features()
            else:
                self.test()
        wandb.finish()

    def eval(self, dataloader, split="val"):
        all_labels = []
        all_loss = []
        all_scores = []
        criterion = nn.BCEWithLogitsLoss()
        self.model.eval()
        for batch in tqdm(
            dataloader, leave=False, dynamic_ncols=True, desc=f"run {split}"
        ):
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                loss = criterion(logits, y.float())

                scores = torch.sigmoid(logits)

                all_scores.append(scores.cpu())
                all_loss.append(loss.item())

                all_labels.append(y.cpu().long())

        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_scores = torch.cat(all_scores, dim=0).numpy()
        avg_loss = np.mean(all_loss)

        # Search over a wider range of thresholds
        best_f1 = -1.0
        best_metrics = {}
        for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            preds = (all_scores > thresh).astype(int)
            precision = precision_score(all_labels, preds, zero_division=1.0)
            recall = recall_score(all_labels, preds, zero_division=1.0)
            f1 = f1_score(all_labels, preds, zero_division=1.0)
            acc = np.mean(preds == all_labels)
            
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {
                    f"loss/{split}": avg_loss,
                    f"f1/{split}": f1,
                    f"precision/{split}": precision,
                    f"recall/{split}": recall,
                    f"acc/{split}": acc,
                    "best_thresh": thresh
                }

        auc = average_precision_score(all_labels, all_scores)
        best_metrics[f"auprc/{split}"] = auc

        return best_metrics

    @torch.no_grad()
    def test_on_features(self):
        import os
        from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
        
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", str(self.args.version))
        test_feat_path = os.path.join(data_dir, "test_features.pt")
        test_lbl_path = os.path.join(data_dir, "test_labels.pt")
        
        if not os.path.exists(test_feat_path):
            print("Không tìm thấy file test_features.pt. Hãy chạy file split_val.py trước.")
            return

        test_features = torch.load(test_feat_path, map_location="cpu")
        test_labels = torch.load(test_lbl_path, map_location="cpu").long()

        ds = TensorDataset(test_features, test_labels)
        dataloader = DataLoader(ds, batch_size=self.args.eval_batch_size, shuffle=False)

        # Load best model from the checkpoint we just saved
        save_dir = os.path.join(f"save/{self.args.version}", "bnn_small")
        ckpt = torch.load(os.path.join(save_dir, "model.pt"), map_location="cpu")
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval()

        all_labels = []
        all_scores = []

        for x, y in dataloader:
            x = x.to(self.device)
            logits = self.model(x)
            scores = torch.sigmoid(logits)
            
            all_scores.append(scores.cpu())
            all_labels.append(y.long().cpu())

        all_labels = torch.cat(all_labels).numpy()
        all_scores = torch.cat(all_scores).numpy()

        auprc = average_precision_score(all_labels, all_scores)

        print("\n" + "*"*50)
        print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST CÂN BẰNG TẠI CÁC NGƯỠNG (THRESHOLDS):")
        
        best_f1 = -1
        best_metrics = {}
        
        for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            preds = (all_scores > thresh).astype(int)
            acc = np.mean(preds == all_labels)
            precision = precision_score(all_labels, preds, zero_division=1.0)
            recall = recall_score(all_labels, preds, zero_division=1.0)
            f1 = f1_score(all_labels, preds, zero_division=1.0)
            
            print(f"--> Threshold {thresh:.1f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "best_thresh": thresh}

        print(f"\nAUPRC chung: {auprc:.4f}")
        print("*"*50 + "\n")
        
        wandb.log({
            "test/acc": best_metrics["acc"],
            "test/precision": best_metrics["precision"],
            "test/recall": best_metrics["recall"],
            "test/f1": best_metrics["f1"],
            "test/auprc": auprc
        })

    @torch.no_grad()
    def test(self):
        small_model = LlamaToxicClassifier(self.device, version="1b")

        version_key = str(self.args.version)

        if version_key == "guardian":
            print("guardian")
            large_model = Guardian(self.device)
        elif version_key in {"1", "2", "3"}:
            print("llama")
            large_model = LlamaToxicClassifier(
                self.device, version=int(version_key))
        else:
            raise NotImplementedError(
                f"Unsupported version for test(): {self.args.version}. "
                "Use --skip_test when training a custom router dataset."
            )
        save_dir = os.path.join(
            f"save/{version_key}", "bnn_small")
        ckpt_path = os.path.join(save_dir, "model.pt")
        model = SafetyGuard(
            ckpt_path,
            small_model,
            large_model,
        )
        
        model.eval()

        for dataset_name in ["wildguard-test-prompt", 
                             "toxic-chat", "openai", "wildguard-test", "xstest", "harmbench"]:
            dataset = get_dataset(dataset_name)
            prompts = dataset["prompts"]
            responses = dataset["responses"]
            labels = dataset["labels"]

            ds = TensorDataset(torch.arange(len(prompts)))
            dataloader = DataLoader(
                ds, self.args.eval_batch_size, shuffle=False)

            preds = []
            probs = []
            final_labels = []
            num_large = 0
            for batch in tqdm(dataloader, leave=False):
                ids = batch[0].tolist()
                batch_prompts = []
                batch_responses = []
                batch_labels = []

                for idx in ids:
                    batch_prompts.append(prompts[idx])
                    batch_labels.append(labels[idx])
                    if responses is not None:
                        batch_responses.append(responses[idx])
                    else:
                        batch_responses = None

                result = model(
                    batch_prompts,
                    batch_responses,
                    batch_labels,
                )

                num_large += result["num_large"]
                probs.append(result["probs"])
                preds.append(result["preds"])
                final_labels.append(result["final_labels"])

            probs = torch.cat(probs).numpy()
            preds = torch.cat(preds).numpy()
            final_labels = torch.cat(final_labels).numpy()

            acc = np.mean(final_labels == preds)
            f1 = f1_score(final_labels, preds)
            precision = precision_score(final_labels, preds)
            recall = recall_score(final_labels, preds)
            auc = average_precision_score(final_labels, preds)

            output_dir = os.path.join(
                "results", f"{dataset_name}", f"{self.args.version}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            wandb.log({f"f1/{dataset_name}": f1,
                       f"precision/{dataset_name}": precision,
                       f"recall/{dataset_name}": recall,
                       f"accuracy/{dataset_name}": acc,
                       f"auc/{dataset_name}": auc,
                       f"large_ratio/{dataset_name}": num_large / len(prompts)})

class Argument(Tap):
    save_dir: str = "./save"
    layer_idx: int = -1

    # Variables dependent on 'mode' (set to None initially)
    train_features: str = None
    train_labels: str = None
    val_features: str = None
    val_labels: str = None
    num_rounds: int = 7
    
    # Other optional arguments
    lr: float = 5e-4
    max_norm: float = 1.0
    num_warmup_steps: int = 100
    batch_size: int = 256
    eval_batch_size: int = 16
    epochs: int = 500
    version: str = "3"

    num_layers: int = 2
    skip_test: bool = False
    # wandb
    exp_name: str = "debug"
    wandb_project: str = "saferoute"

    def process_args(self):
        """Dynamically set paths based on the mode argument."""
        version_key = str(self.version)
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", version_key)
        self.train_features = os.path.join(data_dir, "train_features.pt")
        self.train_labels = os.path.join(data_dir, "train_labels.pt")
        self.val_features = os.path.join(data_dir, "val_features.pt")
        self.val_labels = os.path.join(data_dir, "val_labels.pt")

        self.aug_features = []
        self.aug_labels = []
        if self.num_rounds > 0:
            for r in range(self.num_rounds):
                self.aug_features.append(os.path.join(data_dir, f"round{r}_features.pt"))
                self.aug_labels.append(os.path.join(data_dir, f"round{r}_labels.pt"))
            
if __name__ == "__main__":
    args = Argument(explicit_bool=True).parse_args()
    args.process_args()
    trainer = Trainer(args)
    trainer.train()
