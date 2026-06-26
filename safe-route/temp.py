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
from transformers import get_linear_schedule_with_warmup
import wandb
from dataset import get_dataset
from models import BNN,  LlamaToxicClassifier, Guardian, SafetyGuard


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

        for r in range(args.num_rounds):
            train_features = torch.cat([train_features, torch.load(args.aug_features[r], map_location="cpu")])
            train_labels = torch.cat([train_labels, torch.load(args.aug_labels[r], map_location="cpu")])

        val_features = torch.load(args.val_features, map_location="cpu")
        val_labels = torch.load(args.val_labels, map_location="cpu").long()

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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
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

        for r in range(args.num_rounds):
            train_features = torch.cat([train_features, torch.load(args.aug_features[r], map_location="cpu")])
            train_labels = torch.cat([train_labels, torch.load(args.aug_labels[r], map_location="cpu")])

        val_features = torch.load(args.val_features, map_location="cpu")
        val_labels = torch.load(args.val_labels, map_location="cpu").long()

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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        t_total = len(self.tr_loader) * self.args.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, args.num_warmup_steps, t_total
        )

    def train(self):
        global_step = 1
        criterion = nn.BCEWithLogitsLoss()
        
        best_f1 = -1.0
        best_metrics = None
        
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

                total_loss = loss + 0.01 * kl_loss
                total_loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                wandb.log({"loss/train": loss.item(),
                          "kl/train": kl_loss.item()}, step=global_step)
                num_ones = torch.sum(y).item()
                t.set_description(
                    f"epoch: {epoch}, step: {global_step}, loss: {loss.item(): .4f} kl: {kl_loss.item(): .4f},  num ones: {num_ones}"
                )
                global_step += 1

            if self.val_loader is not None:
                metrics = self.eval(self.val_loader, split="val")
                wandb.log(metrics, step=global_step)
                
                # Check for best model
                if metrics["f1/val"] > best_f1:
                    best_f1 = metrics["f1/val"]
                    best_metrics = metrics
                    self.model.train()  # remove eps var
                    ckpt = {
                        "state_dict": self.model.state_dict(),
                        "layer_idx": self.args.layer_idx
                    }
                    torch.save(ckpt, save_file)
                    print(f"--> Saved new best model at epoch {epoch} with F1: {best_f1:.4f}")

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
        all_preds = []
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

                preds = (scores > 0.8).long().cpu()
                all_preds.append(preds)
                all_labels.append(y.cpu().long())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_scores = torch.cat(all_scores, dim=0).numpy()
        avg_loss = np.mean(all_loss)

    def eval(self, dataloader, split="val"):
        all_preds = []
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

                preds = (scores > 0.8).long().cpu()
                all_preds.append(preds)
                all_labels.append(y.cpu().long())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_scores = torch.cat(all_scores, dim=0).numpy()
        avg_loss = np.mean(all_loss)

        precision = precision_score(all_labels, all_preds, zero_division=1.0)
        recall = recall_score(all_labels, all_preds, zero_division=1.0)
        f1 = f1_score(all_labels, all_preds, zero_division=1.0)
        auc = average_precision_score(all_labels, all_scores)

        acc = np.mean(all_preds == all_labels)
        metrics = {
            f"loss/{split}": avg_loss,
            f"f1/{split}": f1,
            f"precision/{split}": precision,
            f"recall/{split}": recall,
            f"auprc/{split}": auc,
            f"acc/{split}": acc,
        }

        return metrics

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

        all_preds = []
        all_labels = []
        all_scores = []

        for x, y in dataloader:
            x = x.to(self.device)
            logits = self.model(x)
            scores = torch.sigmoid(logits)
            
            all_scores.append(scores.cpu())
            # Ngưỡng dự đoán hiện tại là 0.7 để tăng Precision
            preds = (scores > 0.8).long().cpu()
            all_preds.append(preds)
            all_labels.append(y.long().cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_scores = torch.cat(all_scores).numpy()

        acc = np.mean(all_preds == all_labels)
        precision = precision_score(all_labels, all_preds, zero_division=1.0)
        recall = recall_score(all_labels, all_preds, zero_division=1.0)
        f1 = f1_score(all_labels, all_preds, zero_division=1.0)
        auprc = average_precision_score(all_labels, all_scores)

        print("\n" + "*"*50)
        print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST CÂN BẰNG:")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"AUPRC    : {auprc:.4f}")
        print("*"*50 + "\n")
        
        wandb.log({
            "test/acc": acc,
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1,
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
    lr: float = 1e-3
    max_norm: float = 1.0
    num_warmup_steps: int = 100
    batch_size: int = 512
    eval_batch_size: int = 16
    epochs: int = 1000
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













# import os

# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.metrics import (average_precision_score, f1_score,
#                              precision_score, recall_score)
# from tap import Tap
# from torch.utils.data import (DataLoader, TensorDataset,
#                               WeightedRandomSampler)
# from tqdm import tqdm
# from transformers import get_linear_schedule_with_warmup
# import wandb
# from dataset import get_dataset
# from models import BNN,  LlamaToxicClassifier, Guardian, SafetyGuard


# class Trainer(object):
#     def __init__(self, args) -> None:
#         self.args = args

#         wandb.init(
#             reinit=True,
#             config=args.as_dict(),
#             project=args.wandb_project,
#             name=args.exp_name,
#         )

#         train_features = torch.load(args.train_features, map_location="cpu")
#         train_labels = torch.load(args.train_labels, map_location="cpu").long()

#         for r in range(args.num_rounds):
#             train_features = torch.cat([train_features, torch.load(args.aug_features[r], map_location="cpu")])
#             train_labels = torch.cat([train_labels, torch.load(args.aug_labels[r], map_location="cpu")])

#         val_features = torch.load(args.val_features, map_location="cpu")
#         val_labels = torch.load(args.val_labels, map_location="cpu").long()

#         val_ds = TensorDataset(val_features, val_labels)
#         self.val_loader = DataLoader(
#             val_ds, args.batch_size, shuffle=False)
      

#         train_ds = TensorDataset(train_features, train_labels)

#         class_sample_count = torch.tensor(
#             [(train_labels == t).sum() for t in torch.unique(train_labels, sorted=True)])
#         weight = 1.0 / class_sample_count.float()
#         samples_weight = torch.tensor([weight[t] for t in train_labels])
#         sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
#         self.tr_loader = DataLoader(train_ds, args.batch_size, sampler=sampler)

#         self.device = torch.cuda.current_device()

#         input_dim = train_features.size(1)
#         self.model = BNN(input_dim).to(self.device)
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
#     def __init__(self, args) -> None:
#         self.args = args

#         wandb.init(
#             reinit=True,
#             config=args.as_dict(),
#             project=args.wandb_project,
#             name=args.exp_name,
#         )

#         train_features = torch.load(args.train_features, map_location="cpu")
#         train_labels = torch.load(args.train_labels, map_location="cpu").long()

#         for r in range(args.num_rounds):
#             train_features = torch.cat([train_features, torch.load(args.aug_features[r], map_location="cpu")])
#             train_labels = torch.cat([train_labels, torch.load(args.aug_labels[r], map_location="cpu")])

#         val_features = torch.load(args.val_features, map_location="cpu")
#         val_labels = torch.load(args.val_labels, map_location="cpu").long()

#         val_ds = TensorDataset(val_features, val_labels)
#         self.val_loader = DataLoader(
#             val_ds, args.batch_size, shuffle=False)
      

#         train_ds = TensorDataset(train_features, train_labels)

#         class_sample_count = torch.tensor(
#             [(train_labels == t).sum() for t in torch.unique(train_labels, sorted=True)])
#         weight = 1.0 / class_sample_count.float()
#         samples_weight = torch.tensor([weight[t] for t in train_labels])
#         sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
#         self.tr_loader = DataLoader(train_ds, args.batch_size, sampler=sampler)

#         self.device = torch.cuda.current_device()

#         input_dim = train_features.size(1)
#         self.model = BNN(input_dim).to(self.device)
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

#         t_total = len(self.tr_loader) * self.args.epochs
#         self.scheduler = get_linear_schedule_with_warmup(
#             self.optimizer, args.num_warmup_steps, t_total
#         )

#     def train(self):
#         global_step = 1
#         criterion = nn.BCEWithLogitsLoss()
        
#         best_f1 = -1.0
#         best_metrics = None
        
#         save_dir = os.path.join(
#             f"save/{self.args.version}", "bnn_small")
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         save_file = os.path.join(save_dir, "model.pt")

#         for epoch in tqdm(range(self.args.epochs), dynamic_ncols=True):

#             t = tqdm(
#                 self.tr_loader,
#                 total=len(self.tr_loader),
#                 leave=False,
#                 dynamic_ncols=True,
#             )
#             for batch in t:
#                 x, y = batch
#                 x, y = x.to(self.device), y.to(self.device)

#                 self.model.train()
#                 self.model.zero_grad()

#                 logits = self.model(x)
#                 loss = criterion(logits, y.float())

#                 kl_loss = self.model.get_kl()

#                 total_loss = loss + 0.01 * kl_loss
#                 total_loss.backward()

#                 self.optimizer.step()
#                 self.scheduler.step()

#                 wandb.log({"loss/train": loss.item(),
#                           "kl/train": kl_loss.item()}, step=global_step)
#                 num_ones = torch.sum(y).item()
#                 t.set_description(
#                     f"epoch: {epoch}, step: {global_step}, loss: {loss.item(): .4f} kl: {kl_loss.item(): .4f},  num ones: {num_ones}"
#                 )
#                 global_step += 1

#             if self.val_loader is not None:
#                 metrics = self.eval(self.val_loader, split="val")
#                 wandb.log(metrics, step=global_step)
                
#                 # Check for best model
#                 if metrics["f1/val"] > best_f1:
#                     best_f1 = metrics["f1/val"]
#                     best_metrics = metrics
#                     self.model.train()  # remove eps var
#                     ckpt = {
#                         "state_dict": self.model.state_dict(),
#                         "layer_idx": self.args.layer_idx
#                     }
#                     torch.save(ckpt, save_file)
#                     print(f"--> Saved new best model at epoch {epoch} with F1: {best_f1:.4f}")

#         print("\n" + "="*50)
#         print("TRAINING COMPLETED. BEST VALIDATION RESULTS:")
#         if best_metrics is not None:
#             for k, v in best_metrics.items():
#                 print(f"  {k}: {v:.4f}")
#         print("="*50 + "\n")

#         if not self.args.skip_test:
#             if str(self.args.version) == "dynaguard_1p7b_8b":
#                 self.test_on_features()
#             else:
#                 self.test()
#         wandb.finish()
#     def eval(self, dataloader, split="val"):
#         all_preds = []
#         all_labels = []
#         all_loss = []
#         all_scores = []
#         criterion = nn.BCEWithLogitsLoss()
#         self.model.eval()
#         for batch in tqdm(
#             dataloader, leave=False, dynamic_ncols=True, desc=f"run {split}"
#         ):
#             x, y = batch
#             x, y = x.to(self.device), y.to(self.device)
#             with torch.no_grad():
#                 logits = self.model(x)
#                 loss = criterion(logits, y.float())

#                 scores = torch.sigmoid(logits)

#                 all_scores.append(scores.cpu())
#                 all_loss.append(loss.item())

#                 preds = (scores > 0.5).long().cpu()
#                 all_preds.append(preds)
#                 all_labels.append(y.cpu().long())

#         all_preds = torch.cat(all_preds, dim=0).numpy()
#         all_labels = torch.cat(all_labels, dim=0).numpy()
#         all_scores = torch.cat(all_scores, dim=0).numpy()
#         avg_loss = np.mean(all_loss)

#     def eval(self, dataloader, split="val"):
#         all_preds = []
#         all_labels = []
#         all_loss = []
#         all_scores = []
#         criterion = nn.BCEWithLogitsLoss()
#         self.model.eval()
#         for batch in tqdm(
#             dataloader, leave=False, dynamic_ncols=True, desc=f"run {split}"
#         ):
#             x, y = batch
#             x, y = x.to(self.device), y.to(self.device)
#             with torch.no_grad():
#                 logits = self.model(x)
#                 loss = criterion(logits, y.float())

#                 scores = torch.sigmoid(logits)

#                 all_scores.append(scores.cpu())
#                 all_loss.append(loss.item())

#                 preds = (scores > 0.5).long().cpu()
#                 all_preds.append(preds)
#                 all_labels.append(y.cpu().long())

#         all_preds = torch.cat(all_preds, dim=0).numpy()
#         all_labels = torch.cat(all_labels, dim=0).numpy()
#         all_scores = torch.cat(all_scores, dim=0).numpy()
#         avg_loss = np.mean(all_loss)

#         precision = precision_score(all_labels, all_preds, zero_division=1.0)
#         recall = recall_score(all_labels, all_preds, zero_division=1.0)
#         f1 = f1_score(all_labels, all_preds, zero_division=1.0)
#         auc = average_precision_score(all_labels, all_scores)

#         acc = np.mean(all_preds == all_labels)
#         metrics = {
#             f"loss/{split}": avg_loss,
#             f"f1/{split}": f1,
#             f"precision/{split}": precision,
#             f"recall/{split}": recall,
#             f"auprc/{split}": auc,
#             f"acc/{split}": acc,
#         }

#         return metrics

#     @torch.no_grad()
#     def test_on_features(self):
#         import os
#         from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
        
#         data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", str(self.args.version))
#         test_feat_path = os.path.join(data_dir, "test_features.pt")
#         test_lbl_path = os.path.join(data_dir, "test_labels.pt")
        
#         if not os.path.exists(test_feat_path):
#             print("Không tìm thấy file test_features.pt. Hãy chạy file split_val.py trước.")
#             return

#         test_features = torch.load(test_feat_path, map_location="cpu")
#         test_labels = torch.load(test_lbl_path, map_location="cpu").long()

#         ds = TensorDataset(test_features, test_labels)
#         dataloader = DataLoader(ds, batch_size=self.args.eval_batch_size, shuffle=False)

#         # Load best model from the checkpoint we just saved
#         save_dir = os.path.join(f"save/{self.args.version}", "bnn_small")
#         ckpt = torch.load(os.path.join(save_dir, "model.pt"), map_location="cpu")
#         self.model.load_state_dict(ckpt["state_dict"], strict=False)
#         self.model.eval()

#         all_labels = []
#         all_scores = []

#         for x, y in dataloader:
#             x = x.to(self.device)
#             logits = self.model(x)
#             scores = torch.sigmoid(logits)
            
#             all_scores.append(scores.cpu())
#             all_labels.append(y.long().cpu())

#         all_labels = torch.cat(all_labels).numpy()
#         all_scores = torch.cat(all_scores).numpy()

#         auprc = average_precision_score(all_labels, all_scores)

#         print("\n" + "*"*50)
#         print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST CÂN BẰNG TẠI CÁC NGƯỠNG (THRESHOLDS):")
        
#         best_f1 = -1
#         best_metrics = {}
        
#         for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
#             preds = (all_scores > thresh).astype(int)
#             acc = np.mean(preds == all_labels)
#             precision = precision_score(all_labels, preds, zero_division=1.0)
#             recall = recall_score(all_labels, preds, zero_division=1.0)
#             f1 = f1_score(all_labels, preds, zero_division=1.0)
            
#             print(f"--> Threshold {thresh:.1f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")
            
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_metrics = {"acc": acc, "precision": precision, "recall": recall, "f1": f1}

#         print(f"\nAUPRC chung: {auprc:.4f}")
#         print("*"*50 + "\n")
        
#         wandb.log({
#             "test/acc": best_metrics["acc"],
#             "test/precision": best_metrics["precision"],
#             "test/recall": best_metrics["recall"],
#             "test/f1": best_metrics["f1"],
#             "test/auprc": auprc
#         })

#     @torch.no_grad()
#     def test(self):
#         small_model = LlamaToxicClassifier(self.device, version="1b")

#         version_key = str(self.args.version)

#         if version_key == "guardian":
#             print("guardian")
#             large_model = Guardian(self.device)
#         elif version_key in {"1", "2", "3"}:
#             print("llama")
#             large_model = LlamaToxicClassifier(
#                 self.device, version=int(version_key))
#         else:
#             raise NotImplementedError(
#                 f"Unsupported version for test(): {self.args.version}. "
#                 "Use --skip_test when training a custom router dataset."
#             )
#         save_dir = os.path.join(
#             f"save/{version_key}", "bnn_small")
#         ckpt_path = os.path.join(save_dir, "model.pt")
#         model = SafetyGuard(
#             ckpt_path,
#             small_model,
#             large_model,
#         )
        
#         model.eval()

#         for dataset_name in ["wildguard-test-prompt", 
#                              "toxic-chat", "openai", "wildguard-test", "xstest", "harmbench"]:
#             dataset = get_dataset(dataset_name)
#             prompts = dataset["prompts"]
#             responses = dataset["responses"]
#             labels = dataset["labels"]

#             ds = TensorDataset(torch.arange(len(prompts)))
#             dataloader = DataLoader(
#                 ds, self.args.eval_batch_size, shuffle=False)

#             preds = []
#             probs = []
#             final_labels = []
#             num_large = 0
#             for batch in tqdm(dataloader, leave=False):
#                 ids = batch[0].tolist()
#                 batch_prompts = []
#                 batch_responses = []
#                 batch_labels = []

#                 for idx in ids:
#                     batch_prompts.append(prompts[idx])
#                     batch_labels.append(labels[idx])
#                     if responses is not None:
#                         batch_responses.append(responses[idx])
#                     else:
#                         batch_responses = None

#                 result = model(
#                     batch_prompts,
#                     batch_responses,
#                     batch_labels,
#                 )

#                 num_large += result["num_large"]
#                 probs.append(result["probs"])
#                 preds.append(result["preds"])
#                 final_labels.append(result["final_labels"])

#             probs = torch.cat(probs).numpy()
#             preds = torch.cat(preds).numpy()
#             final_labels = torch.cat(final_labels).numpy()

#             acc = np.mean(final_labels == preds)
#             f1 = f1_score(final_labels, preds)
#             precision = precision_score(final_labels, preds)
#             recall = recall_score(final_labels, preds)
#             auc = average_precision_score(final_labels, preds)

#             output_dir = os.path.join(
#                 "results", f"{dataset_name}", f"{self.args.version}")
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)

#             wandb.log({f"f1/{dataset_name}": f1,
#                        f"precision/{dataset_name}": precision,
#                        f"recall/{dataset_name}": recall,
#                        f"accuracy/{dataset_name}": acc,
#                        f"auc/{dataset_name}": auc,
#                        f"large_ratio/{dataset_name}": num_large / len(prompts)})

# class Argument(Tap):
#     save_dir: str = "./save"
#     layer_idx: int = -1

#     # Variables dependent on 'mode' (set to None initially)
#     train_features: str = None
#     train_labels: str = None
#     val_features: str = None
#     val_labels: str = None
#     num_rounds: int = 7
    
#     # Other optional arguments
#     lr: float = 1e-3
#     max_norm: float = 1.0
#     num_warmup_steps: int = 100
#     batch_size: int = 512
#     eval_batch_size: int = 16
#     epochs: int = 1000
#     version: str = "3"

#     num_layers: int = 2
#     skip_test: bool = False
#     # wandb
#     exp_name: str = "debug"
#     wandb_project: str = "saferoute"

#     def process_args(self):
#         """Dynamically set paths based on the mode argument."""
#         version_key = str(self.version)
#         data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", version_key)
#         self.train_features = os.path.join(data_dir, "train_features.pt")
#         self.train_labels = os.path.join(data_dir, "train_labels.pt")
#         self.val_features = os.path.join(data_dir, "val_features.pt")
#         self.val_labels = os.path.join(data_dir, "val_labels.pt")

#         self.aug_features = []
#         self.aug_labels = []
#         if self.num_rounds > 0:
#             for r in range(self.num_rounds):
#                 self.aug_features.append(os.path.join(data_dir, f"round{r}_features.pt"))
#                 self.aug_labels.append(os.path.join(data_dir, f"round{r}_labels.pt"))
            
# if __name__ == "__main__":
#     args = Argument(explicit_bool=True).parse_args()
#     args.process_args()
#     trainer = Trainer(args)
#     trainer.train()

    
