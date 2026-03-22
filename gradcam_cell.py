# ============================================================
# Grad-CAM Visualisation for AlphaBreast V4
# ============================================================
# Paste this cell AFTER training completes (after Section 8/9).
# It will:
#   1. Train a fresh AlphaBreast V4 on the full training set
#   2. Run Grad-CAM on test samples
#   3. Generate side-by-side heatmap overlays for CC and MLO views
#   4. Save figures to OUTPUT_DIR
#
# Works with BOTH notebook versions (GroupKFold and StratifiedKFold)
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ── Grad-CAM Implementation ────────────────────────────────────

class GradCAM:
    """Grad-CAM for Swin Transformer inside AlphaBreast V4.

    Hooks into the last layer of the Swin encoder (before global
    average pooling) to capture spatial feature maps and their
    gradients. The weighted combination produces a heatmap showing
    which regions most influenced the predicted class.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap for a single image.

        Args:
            input_tensor: [1, 1, 224, 224] image tensor
            class_idx: target class (None = predicted class)

        Returns:
            heatmap: [224, 224] numpy array in [0, 1]
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass through the swin backbone only
        output = self.target_layer(input_tensor)

        # We need to get the full model output for the gradient
        # So we do a full forward pass instead
        return None  # placeholder - see generate_for_pair below

    def generate_for_pair(self, cc_tensor, mlo_tensor, class_idx=None):
        """Generate Grad-CAM heatmaps for both CC and MLO views.

        Args:
            cc_tensor: [1, 1, 224, 224]
            mlo_tensor: [1, 1, 224, 224]
            class_idx: target class (None = predicted class)

        Returns:
            cc_heatmap, mlo_heatmap: [224, 224] numpy arrays in [0, 1]
            predicted_class: int
            confidence: float
        """
        self.model.eval()

        # Enable gradients for this inference
        cc_tensor = cc_tensor.clone().detach().requires_grad_(True)
        mlo_tensor = mlo_tensor.clone().detach().requires_grad_(True)

        # We need to hook into the swin backbone's layers.norm
        # (the final LayerNorm before pooling) to get spatial features.
        # But the model pools internally. Instead, we'll use a trick:
        # hook the last Swin stage and reshape.

        # Forward pass
        self.activations = None
        self.gradients = None

        # Run CC through the shared encoder's swin backbone
        # The hook captures activations at target_layer
        output = self.model(cc_tensor, mlo_tensor)

        # Get predicted class
        probs = F.softmax(output, dim=1)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        confidence = probs[0, class_idx].item()

        # Backward pass for the target class
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            print("Warning: hooks did not capture activations/gradients")
            return None, None, class_idx, confidence

        # Compute Grad-CAM
        # activations shape depends on swin stage: [B, H*W, C] or [B, C, H, W]
        acts = self.activations
        grads = self.gradients

        # Handle different tensor shapes from Swin
        if acts.dim() == 3:
            # [B, num_tokens, channels] - reshape to spatial
            B, N, C = acts.shape
            H = W = int(N ** 0.5)
            acts = acts.permute(0, 2, 1).reshape(B, C, H, W)
            grads = grads.permute(0, 2, 1).reshape(B, C, H, W)

        # Global average pooling of gradients -> channel weights
        weights = grads.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)  # Only positive contributions

        # Resize to input size
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx, confidence

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


# ── Better approach: separate CC and MLO Grad-CAM ──────────────

def compute_gradcam_heatmaps(model, cc_tensor, mlo_tensor, device):
    """Compute Grad-CAM heatmaps for both CC and MLO views.

    Uses the Swin encoder's final norm layer to capture spatial
    features before global average pooling.

    Returns:
        cc_cam, mlo_cam: numpy arrays [224, 224] in [0, 1]
        pred_class: predicted class index
        pred_prob: probability of predicted class
    """
    model.eval()

    # Storage for hooks
    cc_activations = {}
    cc_gradients = {}
    mlo_activations = {}
    mlo_gradients = {}

    # The shared encoder is model.encoder.swin
    # We hook into the last norm layer which has spatial features
    # For Swin-Tiny via timm, this is model.encoder.swin.norm
    # (the LayerNorm applied to the final stage output before pooling)
    target = model.encoder.swin.norm

    call_count = {'n': 0}

    def fwd_hook(module, inp, out):
        if call_count['n'] == 0:
            cc_activations['value'] = out.detach()
        else:
            mlo_activations['value'] = out.detach()
        call_count['n'] += 1

    def bwd_hook(module, grad_in, grad_out):
        # Backward is called in reverse order, so MLO first then CC
        if 'value' not in mlo_gradients:
            mlo_gradients['value'] = grad_out[0].detach()
        else:
            cc_gradients['value'] = grad_out[0].detach()

    fh = target.register_forward_hook(fwd_hook)
    bh = target.register_full_backward_hook(bwd_hook)

    try:
        # Forward pass
        cc_in = cc_tensor.to(device).requires_grad_(True)
        mlo_in = mlo_tensor.to(device).requires_grad_(True)
        output = model(cc_in, mlo_in)

        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()

        # Backward for predicted class
        model.zero_grad()
        output[0, pred_class].backward()

        # Process CC heatmap
        cc_cam = _make_cam(cc_activations.get('value'), cc_gradients.get('value'))
        mlo_cam = _make_cam(mlo_activations.get('value'), mlo_gradients.get('value'))

    finally:
        fh.remove()
        bh.remove()

    return cc_cam, mlo_cam, pred_class, pred_prob


def _make_cam(activations, gradients):
    """Convert activations + gradients into a Grad-CAM heatmap."""
    if activations is None or gradients is None:
        return np.zeros((224, 224))

    acts = activations
    grads = gradients

    # Swin norm output is [B, num_tokens, C]
    if acts.dim() == 3:
        B, N, C = acts.shape
        H = W = int(N ** 0.5)
        if H * W != N:
            # Not a perfect square - try common Swin output sizes
            # For 224 input with patch_size=4: final stage is 7x7=49 tokens
            H = W = 7
            if H * W != N:
                print(f"Warning: cannot reshape {N} tokens to square grid")
                return np.zeros((224, 224))
        acts = acts.permute(0, 2, 1).reshape(B, C, H, W)
        grads = grads.permute(0, 2, 1).reshape(B, C, H, W)
    elif acts.dim() == 4:
        pass  # already [B, C, H, W]

    # Channel-wise global average pooling of gradients
    weights = grads.mean(dim=[2, 3], keepdim=True)

    # Weighted sum
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # Upsample to input size
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Normalize
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    return cam


# ── Visualisation ──────────────────────────────────────────────

def visualise_gradcam(model, dataset, device, num_samples=8,
                      save_dir=None, show_malignant=True, show_benign=True):
    """Generate and display Grad-CAM visualisations.

    Args:
        model: trained AlphaBreastV4 model
        dataset: CBISDDSMDatasetV4 with test transforms
        device: torch device
        num_samples: how many samples to show
        save_dir: where to save figures (None = don't save)
        show_malignant: include malignant examples
        show_benign: include benign examples
    """
    model.eval()

    # Collect samples by class
    malignant_indices = [i for i in range(len(dataset))
                         if dataset.valid_samples[i]['label'] == 1]
    benign_indices = [i for i in range(len(dataset))
                      if dataset.valid_samples[i]['label'] == 0]

    indices = []
    if show_malignant:
        n_mal = min(num_samples // 2, len(malignant_indices))
        indices += malignant_indices[:n_mal]
    if show_benign:
        n_ben = min(num_samples // 2, len(benign_indices))
        indices += benign_indices[:n_ben]

    if not indices:
        print("No samples found!")
        return

    print(f"Generating Grad-CAM for {len(indices)} samples...")

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    class_names = ['Benign', 'Malignant']

    for row, idx in enumerate(indices):
        sample = dataset.valid_samples[idx]
        true_label = sample['label']

        # Get the images
        cc_img, mlo_img, label = dataset[idx]
        cc_tensor = cc_img.unsqueeze(0)  # [1, 1, 224, 224]
        mlo_tensor = mlo_img.unsqueeze(0)

        # Compute Grad-CAM
        cc_cam, mlo_cam, pred_class, pred_prob = compute_gradcam_heatmaps(
            model, cc_tensor, mlo_tensor, device
        )

        # Convert images for display (undo normalisation)
        cc_display = cc_img.squeeze().cpu().numpy() * 0.5 + 0.5
        mlo_display = mlo_img.squeeze().cpu().numpy() * 0.5 + 0.5
        cc_display = np.clip(cc_display, 0, 1)
        mlo_display = np.clip(mlo_display, 0, 1)

        # Determine if prediction is correct
        correct = pred_class == true_label
        color = 'green' if correct else 'red'

        # Column 0: CC original
        axes[row, 0].imshow(cc_display, cmap='gray')
        axes[row, 0].set_title(f'CC View\nTrue: {class_names[true_label]}', fontsize=10)
        axes[row, 0].axis('off')

        # Column 1: CC with Grad-CAM overlay
        axes[row, 1].imshow(cc_display, cmap='gray')
        axes[row, 1].imshow(cc_cam, cmap='jet', alpha=0.4)
        axes[row, 1].set_title(f'CC Grad-CAM\nPred: {class_names[pred_class]} ({pred_prob:.1%})',
                               fontsize=10, color=color)
        axes[row, 1].axis('off')

        # Column 2: MLO original
        axes[row, 2].imshow(mlo_display, cmap='gray')
        axes[row, 2].set_title(f'MLO View\nTrue: {class_names[true_label]}', fontsize=10)
        axes[row, 2].axis('off')

        # Column 3: MLO with Grad-CAM overlay
        axes[row, 3].imshow(mlo_display, cmap='gray')
        axes[row, 3].imshow(mlo_cam, cmap='jet', alpha=0.4)
        axes[row, 3].set_title(f'MLO Grad-CAM\nPred: {class_names[pred_class]} ({pred_prob:.1%})',
                               fontsize=10, color=color)
        axes[row, 3].axis('off')

    plt.suptitle('AlphaBreast V4 - Grad-CAM Visualisation\n'
                 'Red/Yellow = high attention regions | Green text = correct | Red text = incorrect',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, 'v4_gradcam_visualisation.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")

    plt.show()


# ── Run it ─────────────────────────────────────────────────────

# Train a quick model on full training data for visualisation
# (or reuse the last fold's model if you saved it)
print("Training a model on full training data for Grad-CAM visualisation...")
print("(This uses the same hyperparameters as the CV runs)\n")

# Create train dataset with transforms
from torch.utils.data import DataLoader

viz_train_dataset = CBISDDSMDatasetV4(
    paired_train, JPEG_PATH,
    geo_transform=train_geo_transform,
    intensity_transform=train_intensity_transform,
    final_transform=train_final_transform,
    use_cropped=True
)

viz_train_loader = DataLoader(
    viz_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True
)

viz_test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True
)

# Train the model
viz_model = AlphaBreastV4(
    num_classes=2, embed_dim=256, num_heads=8,
    num_evoformer_blocks=2, pretrained=True
).to(device)

viz_metrics = train_single_fold(
    viz_model, viz_train_loader, viz_test_loader, device,
    model_type='multi', epochs=EPOCHS, lr=LR, patience=PATIENCE
)
print(f"\nViz model test metrics: Acc={viz_metrics['accuracy']:.1f}%, AUC={viz_metrics['auc']:.3f}")

# Generate visualisations
print("\nGenerating Grad-CAM heatmaps...")
visualise_gradcam(
    viz_model, test_dataset, device,
    num_samples=8, save_dir=OUTPUT_DIR,
    show_malignant=True, show_benign=True
)

print("\nDone! Check the heatmaps above.")
print("Red/yellow regions = areas the model focuses on for its prediction")
print("These can be included in your report Chapter 5 and demo video")
