# jimmy_net.py (Changes highlighted)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops.misc import Conv2dNormActivation
# from torchvision.ops.misc import Conv2dNormActivation # Not used
# from models.helpers.utils import make_divisible # Not used

def initialize_weights(m):
    # ... (keep existing implementation)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        # Kaiming init for linear layers as well can be good
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.normal_(m.weight, 0, 0.01) # Original
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# FSP matrix computation - Handles spatial mismatch via interpolation BEFORE bmm
def compute_fsp_matrix(feature_map1, feature_map2):
    """
    Compute FSP matrix between two feature maps. Handles spatial mismatches.
    FSP = (1/N) * (FM1 * FM2^T) where N is spatial size (H*W)
    Output shape: (Batch, Channels1, Channels2)
    """
    B, C1, H1, W1 = feature_map1.size()
    _, C2, H2, W2 = feature_map2.size()

    # Interpolate feature_map2 to match feature_map1's spatial dimensions if needed
    if (H1, W1) != (H2, W2):
        feature_map2 = F.interpolate(feature_map2, size=(H1, W1), mode='bilinear', align_corners=False)

    # Reshape for matrix multiplication
    # Pool spatially BEFORE reshape can sometimes be more stable / less memory
    # feature_map1 = F.adaptive_avg_pool2d(feature_map1, (1,1)) # Example spatial pooling
    # feature_map2 = F.adaptive_avg_pool2d(feature_map2, (1,1)) # Example spatial pooling
    # H1, W1 = 1, 1 # Update spatial size if pooled

    feature_map1 = feature_map1.view(B, C1, H1 * W1)  # B x C1 x (H*W)
    feature_map2 = feature_map2.view(B, C2, H1 * W1)  # B x C2 x (H*W)

    # Compute FSP matrix: B x C1 x C2
    fsp = torch.bmm(feature_map1, feature_map2.transpose(1, 2)) / (H1 * W1) # Normalize by spatial size

    return fsp

# Lightweight ResNet as Student Model
class StudentResNet(nn.Module):
    def __init__(self, n_classes=10, in_channels=1):
        super(StudentResNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Initial conv layer
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False) # Stride 1 initially? Often 2
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Add pooling like standard ResNet

        # Layer 1 (Channels: 16 -> 16) - Stride 1
        self.layer1_conv1 = Conv2dNormActivation(16, 16, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.layer1_conv2 = Conv2dNormActivation(16, 16, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=None) # No ReLU before residual add
        self.layer1_activation = nn.ReLU(inplace=True)
        # Store output channel count for easy access
        self.layer1_channels = 16

        # Layer 2 (Channels: 16 -> 32) - Stride 2
        self.layer2_conv1 = Conv2dNormActivation(16, 32, kernel_size=3, stride=2, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
        self.layer2_conv2 = Conv2dNormActivation(32, 32, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d, activation_layer=None) # No ReLU before residual add
        self.layer2_downsample = Conv2dNormActivation(16, 32, kernel_size=1, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=None) # Projection for residual
        self.layer2_activation = nn.ReLU(inplace=True)
        # Store output channel count for easy access
        self.layer2_channels = 32

        # Pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.layer2_channels, n_classes)

        # Feature maps for KD - reset in forward pass
        self.feature_maps = {} # Use a dict for clarity {layer_name: feature_map}

        # Register hooks for feature extraction - Hook the *output* of the main block/activation
        self.hook_handles = []
        # Hook after the activation of layer 1
        self.hook_handles.append(self.layer1_activation.register_forward_hook(self._get_hook("layer1")))
        # Hook after the activation of layer 2
        self.hook_handles.append(self.layer2_activation.register_forward_hook(self._get_hook("layer2")))

        self.apply(initialize_weights)

    # Use a factory function for hooks to capture layer name
    def _get_hook(self, name):
        def hook(module, input, output):
            self.feature_maps[name] = output
        return hook

    def forward(self, x):
        # Clear feature maps at the start of each forward pass
        self.feature_maps = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # Added maxpool

        # Layer 1
        identity = x
        out = self.layer1_conv1(x)
        out = self.layer1_conv2(out)
        out += identity # Add residual
        x = self.layer1_activation(out) # Apply activation (hooked here)

        # Layer 2
        identity = x
        out = self.layer2_conv1(x)
        out = self.layer2_conv2(out)
        identity = self.layer2_downsample(identity) # Downsample residual
        out += identity # Add residual
        x = self.layer2_activation(out) # Apply activation (hooked here)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Use flatten instead of view
        logits = self.fc(x)
        return logits

    # Optional: Remove hooks on deletion
    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()

# --- Teacher Models ---
# Use a base class for teachers to share hook logic
class TeacherBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_maps = {}
        self.hook_handles = []
        self._hook_layers = [] # To be defined by subclasses

    def _get_hook(self, name):
        def hook(module, input, output):
            self.feature_maps[name] = output
        return hook

    def _register_hooks(self):
        # Clear existing hooks first if any
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # Register new hooks based on subclass definition
        for name, layer in self._hook_layers:
             self.hook_handles.append(layer.register_forward_hook(self._get_hook(name)))

    def forward(self, x):
        self.feature_maps = {} # Clear features on each forward pass
        # Subclasses should implement the actual model forward pass
        raise NotImplementedError

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()

# Teacher ResNet50
class TeacherResNet50(TeacherBase):
    def __init__(self, n_classes=10, in_channels=1):
        super().__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        # Adapt input channel
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adapt output classes
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

        # Define layers to hook and their names {name: layer}
        # Hook the output of entire ResNet blocks
        self._hook_layers = [
            ("layer1", self.model.layer1),
            ("layer2", self.model.layer2),
            # ("layer3", self.model.layer3), # Optional: Add more layers if needed
            # ("layer4", self.model.layer4)
        ]
        self._register_hooks() # Register hooks during init

    def forward(self, x):
        self.feature_maps = {} # Clear features
        return self.model(x)

# Teacher EfficientNet-B0
class TeacherEfficientNetB0(TeacherBase):
    def __init__(self, n_classes=10, in_channels=1):
        super().__init__()
        self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        # Adapt input channel
        self.model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Adapt output classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, n_classes)

        # Define layers to hook and their names
        # Hook outputs of specific blocks (MBConv blocks)
        self._hook_layers = [
             ("block2", self.model.features[2]), # After block 2 (MBConv stride 2)
             ("block4", self.model.features[4]), # After block 4 (MBConv stride 2)
             # ("block6", self.model.features[6]), # Optional
             # ("block8", self.model.features[8])  # Optional (final features before head)
        ]
        self._register_hooks() # Register hooks

    def forward(self, x):
        self.feature_maps = {} # Clear features
        return self.model(x)


# --- Factory Functions ---
def get_model(n_classes=10, in_channels=1, **kwargs):
    """Returns the student model."""
    # kwargs like base_channels etc are ignored for this specific StudentResNet
    return StudentResNet(n_classes=n_classes, in_channels=in_channels)

def get_teachers(n_classes=10, in_channels=1):
    """Returns a list of teacher models for KD."""
    # NOTE: Ensure teacher models are compatible with the FSP approach
    # (i.e., produce feature maps that can be meaningfully compared)
    return [
        TeacherResNet50(n_classes=n_classes, in_channels=in_channels),
        TeacherEfficientNetB0(n_classes=n_classes, in_channels=in_channels)
    ]

# Helper to get channel counts from hooked layers (run AFTER model init)
def get_feature_channel_dims(model):
    dims = {}
    if isinstance(model, TeacherBase) or isinstance(model, StudentResNet):
         # Need a dummy forward pass to populate feature_maps IF hooks based on activations
         # Or inspect layers directly if hooks are on modules with known channel outputs
        if isinstance(model, StudentResNet):
            # For StudentResNet, we defined attributes for channel counts
             if "layer1" in model.feature_maps: # Check if dummy pass ran
                 dims["layer1"] = model.feature_maps["layer1"].size(1)
             else: # Estimate from layer def
                 dims["layer1"] = model.layer1_channels
             if "layer2" in model.feature_maps:
                 dims["layer2"] = model.feature_maps["layer2"].size(1)
             else: # Estimate from layer def
                 dims["layer2"] = model.layer2_channels

        elif hasattr(model, '_hook_layers'):
             # Try to infer from the hooked layer's output channels if possible
             # This is complex; requires knowing layer types (e.g., last conv/bn in block)
             # A dummy forward pass is more reliable
             print(f"Warning: Channel dims for {type(model)} need dummy pass or manual spec.")
             # Example for ResNet based on known block output channels
             if isinstance(model, TeacherResNet50):
                 try:
                    dims["layer1"] = model.model.layer1[-1].conv3.out_channels # Bottleneck block
                 except:
                    dims["layer1"] = model.model.layer1[-1].conv2.out_channels # Basic block
                 try:
                    dims["layer2"] = model.model.layer2[-1].conv3.out_channels
                 except:
                    dims["layer2"] = model.model.layer2[-1].conv2.out_channels
                 # ... etc for layer3, layer4
    return dims