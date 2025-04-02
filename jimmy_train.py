# jimmy_train.py (Changes highlighted or noted)
import argparse
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import transformers
import wandb
import torch.nn as nn
from dataset.dcase25 import get_training_set, get_test_set # Assuming these exist and work
from helpers.init import worker_init_fn # Assuming this exists
from helpers.utils import mixstyle # Assuming this exists
from helpers import complexity # Assuming this exists
from jimmy_net import get_model, get_teachers, compute_fsp_matrix, get_feature_channel_dims, initialize_weights # <--- ADDED initialize_weights HERE


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config) # Save config
        self.config = config

        # -------- Preprocessing Pipeline --------
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.Resample(
                orig_freq=self.config.orig_sample_rate, # Use self.config here
                new_freq=self.config.sample_rate
            ),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                win_length=self.config.window_length,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                f_min=self.config.f_min,
                f_max=self.config.f_max
            )
        )
        self.mel_augment = torch.nn.Sequential(
            # Make sure config values are appropriate (e.g., freqm > 0)
            torchaudio.transforms.FrequencyMasking(self.config.freqm, iid_masks=True),
            torchaudio.transforms.TimeMasking(self.config.timem, iid_masks=True)
)

        # --- Student Model ---
        self.model = get_model(
            n_classes=config.n_classes,
            in_channels=config.in_channels,
            # Other params ignored by current get_model -> StudentResNet
        )

        # --- Teacher Models (No gradients needed) ---
        self.teachers = get_teachers(n_classes=config.n_classes, in_channels=config.in_channels)
        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
                
        try:
            student_c1 = self.model.layer1_channels # e.g., 16
            student_c2 = self.model.layer2_channels # e.g., 32
            student_fsp_in_dim = student_c1 * student_c2
            print(f"Student FSP features: C1={student_c1}, C2={student_c2}, Projection Input Dim={student_fsp_in_dim}")
        except AttributeError:
             print("ERROR: Could not determine student feature channel counts automatically.")
             print("Ensure StudentResNet exposes layer1_channels and layer2_channels attributes.")
             # Fallback or raise error
             student_fsp_in_dim = 16 * 32

        target_c1 = 256 # Example: ResNet50 layer1 output channels
        target_c2 = 512 # Example: ResNet50 layer2 output channels
        fsp_proj_out_dim = target_c1 * target_c2
        print(f"Target FSP Teacher dims (example): C1={target_c1}, C2={target_c2}, Projection Output Dim={fsp_proj_out_dim}")

        # Define the trainable projection layer for the student's FSP matrix
        self.fsp_proj = nn.Linear(student_fsp_in_dim, fsp_proj_out_dim)
        # Initialize weights for the projection layer
        initialize_weights(self.fsp_proj) # Use the same init helper

        self.student_fsp_pair = ("layer1", "layer2")
        # Define corresponding pairs for each teacher
        self.teacher_fsp_pairs = [
            ("layer1", "layer2"), # For TeacherResNet50
            ("block2", "block4")  # For TeacherEfficientNetB0
        ]

        self.teacher_target_dims = [
             (target_c1, target_c2), # For TeacherResNet50 pair
             # We need a projection per teacher pair if dimensions differ,
             # OR project student to ONE target dim and resize teacher FSPs.
             # For simplicity, let's assume fsp_proj targets the first teacher's dims.
             # We'll need to handle the second teacher carefully in the loss.
             (40, 80) # Example dimensions for EffNet pair
        ]
        print(f"Teacher FSP Pairs: {self.teacher_fsp_pairs}")
        print(f"Teacher Target Dims: {self.teacher_target_dims}")
        
        # Ensure number of pairs/dims matches number of teachers
        assert len(self.teacher_fsp_pairs) == len(self.teachers)
        assert len(self.teacher_target_dims) == len(self.teachers)
        
        # Check if student projection output matches first teacher target dim
        assert self.fsp_proj.out_features == self.teacher_target_dims[0][0] * self.teacher_target_dims[0][1] , \
            "Student projection output dim must match target teacher FSP dim"


        # -------- Device/Label Definitions --------
        # ... (keep existing definitions)
        self.device_ids = [...]
        self.label_ids = [...]
        self.device_groups = {...}

        # KD stage control
        self.kd_stage = "fsp" # Start with FSP

    def setup(self, stage=None):
        """Move teachers to the correct device."""
        if stage == "fit" or stage is None:
            print(f"Moving {len(self.teachers)} teachers to device: {self.device}")
            for i, teacher in enumerate(self.teachers):
                teacher.to(self.device)
                print(f"Teacher {i} ({type(teacher).__name__}) moved.")
            # No need to initialize projections here anymore
            # Ensure student's projection layer is also on the right device
            self.fsp_proj.to(self.device)
            print(f"Student FSP projection layer moved to device: {self.device}")


    def mel_forward(self, x):
        # ... (keep existing implementation)
        x = self.mel(x)
        # Ensure model is in training mode for augmentation
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log() # LogMel
        return x

    def forward(self, x):
        """Student forward pass for inference/validation/test"""
        x_mel = self.mel_forward(x)
        logits = self.model(x_mel)
        return logits

    def configure_optimizers(self):
        """
        Configure optimizers and LR scheduler, similar to the baseline,
        but optimizing only student + projection layer parameters.
        """
        print("Configuring optimizers...") # Add print statement for debugging
        # Combine parameters from the student model and the FSP projection layer
        parameters_to_optimize = list(self.model.parameters()) + list(self.fsp_proj.parameters())

        optimizer = torch.optim.AdamW(
            parameters_to_optimize,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        # Use estimated_stepping_batches directly, as in the baseline
        # This value is calculated by PyTorch Lightning after dataloader inspection
        try:
            num_training_steps = self.trainer.estimated_stepping_batches
            print(f"Using self.trainer.estimated_stepping_batches: {num_training_steps}")
            if num_training_steps is None or num_training_steps <= 0:
                print("Warning: estimated_stepping_batches is invalid. Using fallback.")
                # Provide a reasonable fallback if needed, though it shouldn't be necessary
                num_training_steps = 100000 # Large fallback
        except AttributeError:
            print("Warning: self.trainer.estimated_stepping_batches not available yet. Using fallback.")
            # Provide a reasonable fallback if needed
            num_training_steps = 100000 # Large fallback


        print(f"Scheduler total training steps set to: {num_training_steps}")

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps # Pass the determined value
        )
        print(f"Scheduler: Cosine with warmup ({self.config.warmup_steps} steps) to total steps ({num_training_steps})")

        # Return in the standard format expected by Lightning
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def training_step(self, train_batch, batch_idx):
        x, _, labels, _, _ = train_batch # Assuming this batch structure
        B = x.size(0) # Get batch size

        # 1. Preprocessing
        x_mel = self.mel_forward(x)

        # Optional: MixStyle Augmentation
        if self.config.mixstyle_p > 0 and self.training:
            x_mel = mixstyle(x_mel, self.config.mixstyle_p, self.config.mixstyle_alpha)

        # 2. Forward pass through student (hooks will capture features)
        student_logits = self.model(x_mel)
        student_features = self.model.feature_maps # {name: feature_map}

        # --- Loss Calculation ---
        if self.kd_stage == "fsp":
            total_fsp_loss = 0
            fsp_teachers_processed = 0

            # Check if student features needed for FSP are present
            s_key1, s_key2 = self.student_fsp_pair
            if s_key1 not in student_features or s_key2 not in student_features:
                 print(f"Warning: Student features '{s_key1}' or '{s_key2}' not found. Skipping FSP.")
                 # Fallback to Cross-Entropy or return 0 loss for this step?
                 loss = F.cross_entropy(student_logits, labels)
                 self.log("train/loss_ce_fallback", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
                 return loss # Or return 0? Decide on fallback behavior

            try:
                # Calculate student FSP matrix
                sfm1 = student_features[s_key1]
                sfm2 = student_features[s_key2]
                fsp_student = compute_fsp_matrix(sfm1, sfm2) # Shape: (B, Cs1, Cs2)

                # Project student FSP matrix
                fsp_student_flat = fsp_student.view(B, -1)
                fsp_student_proj = self.fsp_proj(fsp_student_flat) # Shape: (B, ProjOutDim)

                # Reshape projected student FSP to target teacher's FSP shape (C_t1, C_t2)
                # We use the dimensions from the *first* teacher pair as the target.
                target_ct1, target_ct2 = self.teacher_target_dims[0]
                fsp_student_proj_reshaped = fsp_student_proj.view(B, target_ct1, target_ct2)

            except Exception as e:
                print(f"Error computing or projecting student FSP: {e}")
                # Fallback to Cross-Entropy
                loss = F.cross_entropy(student_logits, labels)
                self.log("train/loss_ce_fallback", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
                return loss

            # Loop through teachers
            for i, teacher in enumerate(self.teachers):
                 t_key1, t_key2 = self.teacher_fsp_pairs[i]
                 teacher_ct1, teacher_ct2 = self.teacher_target_dims[i] # Get expected dims

                 try:
                     # Teacher forward pass (no gradients)
                     with torch.no_grad():
                         _ = teacher(x_mel) # Run forward to populate hooks
                         teacher_features = teacher.feature_maps

                     # Check if teacher features are present
                     if t_key1 not in teacher_features or t_key2 not in teacher_features:
                         print(f"Warning: Teacher {i} features '{t_key1}' or '{t_key2}' not found. Skipping.")
                         continue

                     # Calculate teacher FSP matrix
                     tfm1 = teacher_features[t_key1]
                     tfm2 = teacher_features[t_key2]
                     fsp_teacher = compute_fsp_matrix(tfm1, tfm2) # Shape: (B, T_Ci1, T_Ci2)
                     fsp_teacher = fsp_teacher.detach() # IMPORTANT: Detach teacher FSP

                     # --- Match teacher FSP to student projected FSP shape ---
                     # Option 1: If teacher FSP already matches target (e.g., first teacher)
                     if fsp_teacher.shape[1] == target_ct1 and fsp_teacher.shape[2] == target_ct2:
                         fsp_teacher_target = fsp_teacher
                     # Option 2: Resize teacher FSP if dimensions differ (e.g., second teacher)
                     # Use adaptive pooling on the flattened matrix for resizing channels x channels
                     else:
                         print(f"Teacher {i} FSP shape {fsp_teacher.shape} differs from target {(B, target_ct1, target_ct2)}. Resizing.")
                         fsp_teacher_flat = fsp_teacher.view(B, -1) # Flatten to (B, T_Ci1 * T_Ci2)
                         # Target size is target_ct1 * target_ct2
                         target_flat_dim = target_ct1 * target_ct2
                         # Use adaptive pooling in 1D to resize the feature dimension
                         fsp_teacher_resized_flat = F.adaptive_avg_pool1d(fsp_teacher_flat.unsqueeze(1), target_flat_dim).squeeze(1)
                         # Reshape back to target 3D FSP shape
                         fsp_teacher_target = fsp_teacher_resized_flat.view(B, target_ct1, target_ct2)


                     # Calculate MSE loss between projected student FSP and target teacher FSP
                     # Both should now be shape (B, target_ct1, target_ct2)
                     fsp_loss = F.mse_loss(fsp_student_proj_reshaped, fsp_teacher_target)

                     total_fsp_loss += fsp_loss
                     fsp_teachers_processed += 1
                     self.log(f"train/fsp_loss_t{i}", fsp_loss, on_step=True, on_epoch=False, batch_size=B)

                 except Exception as e:
                     print(f"Error processing teacher {i} ({type(teacher).__name__}) FSP: {e}")
                     import traceback
                     # traceback.print_exc() # Uncomment for detailed stack trace
                     continue # Skip this teacher if error occurs

            # Average FSP loss over processed teachers
            if fsp_teachers_processed > 0:
                 loss = total_fsp_loss / fsp_teachers_processed
                 self.log("train/fsp_loss_avg", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
            else:
                 print("Warning: No teachers processed successfully for FSP. Using CE loss.")
                 loss = F.cross_entropy(student_logits, labels)
                 self.log("train/loss_ce_fallback", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)

        else: # kd_stage == "classification" (or other stages)
            # Standard Cross-Entropy Loss using student logits
            loss = F.cross_entropy(student_logits, labels)
            self.log("train/loss_ce", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)

        # Log learning rate and epoch
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        self.log("epoch", self.current_epoch, on_step=True, on_epoch=True)

        # Store loss for epoch-level aggregation if needed
        # self.training_step_outputs.append(loss) # Careful with memory if loss tensor is large
        return loss

    def on_train_epoch_end(self):
        # Check if it's time to switch KD stage
        # Epochs are 0-indexed, so check against `kd_fsp_epochs - 1`
        if self.current_epoch == self.config.kd_fsp_epochs - 1:
            print(f"\nEpoch {self.current_epoch + 1}: Switching from FSP to Classification stage.\n")
            self.kd_stage = "classification"
        # Optional: Add logic for other stage transitions if needed

        # Optional: Aggregate and log epoch-level training loss average
        # avg_loss = torch.stack(self.training_step_outputs).mean()
        # self.log('train/loss_epoch', avg_loss, on_epoch=True, prog_bar=False)
        # self.training_step_outputs.clear() # Clear memory


    # --- Validation and Test Steps/Epoch Ends ---
    # Keep the existing validation_step, on_validation_epoch_end,
    # test_step, and on_test_epoch_end methods. They seem correct
    # for standard evaluation logging.
    # Inside PLModule class

    # Inside PLModule class

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, _ = val_batch
        y_hat = self.forward(x) # Use simple forward pass

        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        # Construct the results dictionary (as before)
        results = {
            "loss": samples_loss.mean(), # Aggregate batch loss
            "n_correct": n_correct,
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }
        # Per-device stats
        for d in self.device_ids:
            results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(devices):
            results[f"devloss.{d}"] += samples_loss[i]
            results[f"devcnt.{d}"] += 1
            results[f"devn_correct.{d}"] += n_correct_per_sample[i]
        # Per-label stats
        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)
        for i, lbl_index in enumerate(labels):
            lbl_name = self.label_ids[lbl_index]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        # Move results to CPU before returning (good practice)
        results_cpu = {k: v.cpu() for k, v in results.items()}

        # Return the dictionary - PL will collect these
        return results_cpu
        # REMOVE the line: self.validation_step_outputs.append(results)


   # Inside PLModule class

    # Add the 'outputs' argument here
    def on_validation_epoch_end(self, outputs):
        """
        Called at the end of the validation epoch. Aggregates step outputs and logs them.
        Uses the standard PL pattern with the 'outputs' argument.
        """
        if not outputs:
            print("Warning: validation outputs list is empty. Skipping validation epoch end.")
            return # Nothing to process

        # --- Start: Aggregation Logic ---
        # Initialize based on keys from the first dictionary element
        # Assuming all steps return dictionaries with the same keys
        aggregated_outputs = {k: [] for k in outputs[0].keys()}

        # Aggregate results from all steps
        for step_output in outputs:
            # Optional: Add a check here if steps might return different structures
            # if not isinstance(step_output, dict): continue
            for k, v in step_output.items():
                if k in aggregated_outputs: # Check key exists
                    aggregated_outputs[k].append(torch.as_tensor(v)) # Ensure tensor
                else:
                    print(f"Warning: Key '{k}' found in step output but not in first step's keys. Initializing.")
                    aggregated_outputs[k] = [torch.as_tensor(v)]


        # Stack tensors for each key
        try:
            final_outputs = {}
            for k, v_list in aggregated_outputs.items():
                if v_list: # Check if list is not empty
                    final_outputs[k] = torch.stack(v_list)
                else:
                    print(f"Warning: Aggregated list for key '{k}' is empty.")
                    # Assign a default or skip the key
                    final_outputs[k] = torch.tensor([]) # Example: empty tensor

        except Exception as e:
            print(f"Error stacking tensors in on_validation_epoch_end: {e}")
            return
        # --- End: Aggregation Logic ---


        # --- Compute metrics (using 'final_outputs' dictionary) ---
        # Check keys exist and tensors are not empty before calculation
        if "loss" in final_outputs and final_outputs["loss"].numel() > 0:
            avg_loss = final_outputs["loss"].mean()
        else:
            print("Warning: 'loss' key not found or empty in aggregated outputs. Cannot compute avg_loss.")
            avg_loss = torch.tensor(float('nan'))

        if ("n_correct" in final_outputs and "n_pred" in final_outputs and
            final_outputs["n_correct"].numel() > 0 and final_outputs["n_pred"].numel() > 0):
            total_preds = final_outputs["n_pred"].sum()
            if total_preds > 0:
                acc = final_outputs["n_correct"].sum() / total_preds
            else:
                print("Warning: Total predictions (n_pred) is zero. Accuracy set to NaN.")
                acc = torch.tensor(float('nan'))
        else:
            print("Warning: 'n_correct' or 'n_pred' keys not found or empty. Cannot compute accuracy.")
            acc = torch.tensor(float('nan'))

        logs = {"acc": acc, "loss": avg_loss}

        # --- Per-device stats ---
        for d in self.device_ids:
            dev_loss_key = f"devloss.{d}"
            dev_cnt_key = f"devcnt.{d}"
            dev_correct_key = f"devn_correct.{d}"

            # Check if keys exist and have data
            if (dev_loss_key in final_outputs and dev_cnt_key in final_outputs and dev_correct_key in final_outputs and
                final_outputs[dev_loss_key].numel() > 0 and final_outputs[dev_cnt_key].numel() > 0 and final_outputs[dev_correct_key].numel() > 0):

                dev_loss = final_outputs[dev_loss_key].sum()
                dev_cnt = final_outputs[dev_cnt_key].sum()
                dev_correct = final_outputs[dev_correct_key].sum()

                if dev_cnt > 0:
                    logs[f"loss.{d}"] = dev_loss / dev_cnt
                    logs[f"acc.{d}"] = dev_correct / dev_cnt
                else:
                    logs[f"loss.{d}"] = torch.tensor(float('nan'))
                    logs[f"acc.{d}"] = torch.tensor(float('nan'))
                logs[f"cnt.{d}"] = dev_cnt

                # Group stats aggregation
                group_name = self.device_groups[d]
                logs[f"acc.{group_name}_sum"] = logs.get(f"acc.{group_name}_sum", 0.) + dev_correct
                logs[f"count.{group_name}"] = logs.get(f"count.{group_name}", 0.) + dev_cnt
                logs[f"lloss.{group_name}_sum"] = logs.get(f"lloss.{group_name}_sum", 0.) + dev_loss
            else:
                print(f"Warning: Missing or empty data for device {d} in validation. Skipping stats.")
                # Assign default NaN values if keys are missing
                logs[f"loss.{d}"] = torch.tensor(float('nan'))
                logs[f"acc.{d}"] = torch.tensor(float('nan'))
                logs[f"cnt.{d}"] = torch.tensor(0.)


        # Reduce group stats safely (same as before)
        for grp in set(self.device_groups.values()):
            grp_count = logs.get(f"count.{grp}", 0.)
            if grp_count > 0:
                logs[f"acc.{grp}"] = logs.get(f"acc.{grp}_sum", 0.) / grp_count
                logs[f"lloss.{grp}"] = logs.get(f"lloss.{grp}_sum", 0.) / grp_count
            else:
                logs[f"acc.{grp}"] = torch.tensor(float('nan'))
                logs[f"lloss.{grp}"] = torch.tensor(float('nan'))
            logs.pop(f"acc.{grp}_sum", None)
            logs.pop(f"lloss.{grp}_sum", None)

        # --- Per-label stats ---
        label_accs = []
        for lbl in self.label_ids:
            lbl_loss_key = f"lblloss.{lbl}"
            lbl_cnt_key = f"lblcnt.{lbl}"
            lbl_correct_key = f"lbln_correct.{lbl}"

            if (lbl_loss_key in final_outputs and lbl_cnt_key in final_outputs and lbl_correct_key in final_outputs and
                final_outputs[lbl_loss_key].numel() > 0 and final_outputs[lbl_cnt_key].numel() > 0 and final_outputs[lbl_correct_key].numel() > 0):

                lbl_loss = final_outputs[lbl_loss_key].sum()
                lbl_cnt = final_outputs[lbl_cnt_key].sum()
                lbl_correct = final_outputs[lbl_correct_key].sum()

                if lbl_cnt > 0:
                    logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
                    logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
                    label_accs.append(logs[f"acc.{lbl}"]) # Collect valid accuracies
                else:
                    logs[f"loss.{lbl}"] = torch.tensor(float('nan'))
                    logs[f"acc.{lbl}"] = torch.tensor(float('nan'))

                logs[f"cnt.{lbl}"] = lbl_cnt.float() # Log count regardless
            else:
                print(f"Warning: Missing or empty data for label {lbl} in validation. Skipping stats.")
                logs[f"loss.{lbl}"] = torch.tensor(float('nan'))
                logs[f"acc.{lbl}"] = torch.tensor(float('nan'))
                logs[f"cnt.{lbl}"] = torch.tensor(0.)

        # Compute macro-average accuracy safely (same as before)
        if label_accs:
            logs["macro_avg_acc"] = torch.mean(torch.stack(label_accs))
        else:
            print("Warning: No valid label accuracies found for macro average.")
            logs["macro_avg_acc"] = torch.tensor(float('nan'))

        # Log results
        self.log_dict({f"val/{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True)
        # No need to clear self.validation_step_outputs as it's no longer used

    def test_step(self, test_batch, batch_idx):
         # ... (Keep existing code - similar to validation_step) ...
        x, files, labels, devices, _ = test_batch
        y_hat = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        # ... rest of test_step ...
        self.test_step_outputs.append({k: v.cpu() for k, v in results.items()}) # Move to CPU

    def on_test_epoch_end(self):
         # ... (Keep existing aggregation and logging code) ...
        outputs = {k: [] for k in self.test_step_outputs[0]}
        # ... rest of aggregation ...
        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        self.test_step_outputs.clear()


# --- Main Training Function ---
def train(config):
    torch.set_float32_matmul_precision('medium') # Or 'high'

    # Logger
    wandb_logger = WandbLogger(
        project=config.project_name,
        # notes="Baseline System with FSP KD for DCASE'25 Task 1.", # Updated notes
        tags=["DCASE25", "KD", "FSP"], # Added FSP tag
        config=vars(config), # Log hyperparameters from args
        name=config.experiment_name
    )

    # DataLoaders
    assert config.subset == 25, "DCASE'25 subset must be 25%."
    roll_samples = int(config.orig_sample_rate * config.roll_sec) if config.roll_sec > 0 else 0
    train_ds = get_training_set(config.subset, device=None, roll=roll_samples) # device=None assumed means load to RAM/CPU
    train_dl = DataLoader(
        dataset=train_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers, # Use config value
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True # Helps speed up CPU->GPU transfer if workers > 0
    )
    test_ds = get_test_set(device=None)
    test_dl = DataLoader(
        dataset=test_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers, # Use config value
        batch_size=config.batch_size,
        pin_memory=True
    )
    print(f"Train DataLoader: {len(train_dl)} batches, Batch Size: {config.batch_size}")
    print(f"Test DataLoader: {len(test_dl)} batches, Batch Size: {config.batch_size}")


    # Model Module
    pl_module = PLModule(config)

    # --- Optional: Model Complexity ---
    # Run a test forward pass to get Mel shape AFTER PLModule init
    try:
        print("Running dummy forward pass for shape check...")
        # Get a sample batch from the DataLoader
        dummy_batch = next(iter(train_dl)) # Use train_dl for realistic input
        dummy_input = dummy_batch[0][0].unsqueeze(0) # Get first audio sample, add batch dim
        # Move dummy input to expected device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dummy_input = dummy_input.to(device)
        pl_module.to(device) # Ensure module is on device too

        print(f"Dummy input audio shape: {dummy_input.shape}")
        mel_shape = pl_module.mel_forward(dummy_input).size()
        print(f"Calculated Mel Spectrogram shape: {mel_shape}") # B, C, H, W

        # Calculate complexity (ensure model is on CPU/GPU as expected by complexity tool)
        # Move model to CPU for reliable complexity calculation if using certain tools
        # pl_module.to('cpu')
        macs, params_bytes = complexity.get_torch_macs_memory(pl_module.model, input_size=mel_shape)
        # pl_module.to(device) # Move back to training device

        print(f"Model Complexity: MACs={macs/1e9:.2f} G, ParamsSize={params_bytes/1e6:.2f} MB")
        wandb_logger.experiment.config["MACs"] = macs
        wandb_logger.experiment.config["Parameters_Bytes"] = params_bytes
    except Exception as e:
        print(f"DataLoader test or complexity calculation failed: {e}")
        import traceback
        traceback.print_exc()
        # Decide if you want to raise the error or continue
        # raise # Reraise the exception to stop execution


    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None, # Specify devices=1 for single GPU
        precision=config.precision, # e.g., "32-true", "16-mixed"
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        # Add gradient clipping if needed
        # gradient_clip_val=1.0,
        # Log gradients, weights
        # log_every_n_steps=50, # Adjust frequency
        callbacks=[
             pl.callbacks.ModelCheckpoint(
                 save_last=True, # Save last checkpoint
                 # Optional: save best checkpoint based on validation metric
                 # monitor="val/acc", # Example: monitor validation accuracy
                 # mode="max",
                 # filename="best-checkpoint-{epoch}-{val/acc:.2f}"
                 ),
             pl.callbacks.LearningRateMonitor(logging_interval='step') # Log LR
             ]
    )

    # --- Run Training ---
    print("Starting training...")
    trainer.fit(pl_module, train_dl, test_dl) # Pass train and val dataloaders

    # --- Run Testing ---
    print("Starting testing...")
    # Load best checkpoint if saved, otherwise uses last implicitly with ckpt_path="last"
    trainer.test(model=pl_module, dataloaders=test_dl, ckpt_path="last") # Or path to best ckpt

    wandb.finish()
    print("Training and Testing finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 25 FSP KD argument parser')

    # General arguments
    parser.add_argument("--project_name", type=str, default="DCASE25_Task1")
    parser.add_argument("--experiment_name", type=str, default="ResNetStudent_FSP_KD") # More specific name
    parser.add_argument("--num_workers", type=int, default=4) # Default to a reasonable number
    parser.add_argument("--precision", type=str, default="32-true", choices=["32-true", "16-mixed"]) # Use PL precision flags
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1) # Validate every epoch
    parser.add_argument("--orig_sample_rate", type=int, default=44100)

    # Subset
    parser.add_argument("--subset", type=int, default=25)

    # Model hyperparameters (Student specific - others ignored by current get_model)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1)
    # parser.add_argument("--base_channels", type=int, default=32) # Ignored by StudentResNet
    # parser.add_argument("--channels_multiplier", type=float, default=1.8) # Ignored
    # parser.add_argument("--expansion_rate", type=float, default=2.1) # Ignored

    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=100) # Increase epochs
    parser.add_argument("--batch_size", type=int, default=128) # Adjust based on GPU memory
    parser.add_argument("--mixstyle_p", type=float, default=0.4)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--roll_sec", type=float, default=0.0) # Disable roll by default? Check dataset needs

    # KD specific hyperparameters
    parser.add_argument("--kd_fsp_epochs", type=int, default=30, help="Number of epochs for FSP training (Stage 1)")
    # parser.add_argument("--lambda_fsp", type=float, default=1.0, help="Weight for FSP loss") # Optional: Add loss weighting

    # Learning rate schedule
    parser.add_argument("--lr", type=float, default=0.001) # Often lower LR works better with AdamW
    parser.add_argument("--warmup_steps", type=int, default=1000) # Adjust warmup

    # Spectrogram parameters
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--window_length", type=int, default=1024) # Common values
    parser.add_argument("--hop_length", type=int, default=320)   # ~10ms hop for 32kHz
    parser.add_argument("--n_fft", type=int, default=1024)      # Often same as window_length
    parser.add_argument("--n_mels", type=int, default=128)      # Common value
    parser.add_argument("--freqm", type=int, default=24)        # Freq masking amount (~20% of n_mels)
    parser.add_argument("--timem", type=int, default=0)         # Time masking often not used if roll/mixup used
    parser.add_argument("--f_min", type=int, default=50)        # Min freq
    parser.add_argument("--f_max", type=int, default=14000)     # Max freq (~Nyquist/2 for 32kHz)

    args = parser.parse_args()

    # --- Run Training ---
    train(args)