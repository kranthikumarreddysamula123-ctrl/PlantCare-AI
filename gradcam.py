
import torch
import torch.nn.functional as F
import numpy as np
import cv2


def generate_gradcam(model, input_tensor, original_image, device):

    model.eval()
    model.zero_grad(set_to_none=True)

    activations = {}

    # Use layer3 for a more detailed feature map
    target_layer = model.layer3[-1]

    def forward_hook(module, inputs, output):
        activations["features"] = output
        output.retain_grad()

    hook = target_layer.register_forward_hook(forward_hook)

    try:
        # Enable gradients for Grad-CAM
        with torch.enable_grad():

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted_class = torch.max(
                probabilities, dim=1
            )

            # Calculate gradients for predicted class
            class_score = outputs[0, predicted_class.item()]
            class_score.backward()

        # Get feature maps and gradients
        features = activations["features"]
        gradients = features.grad

        # Calculate channel weights
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Generate CAM
        cam = (weights * features).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        # Resize heatmap to image dimensions
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize heatmap
        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        # Convert heatmap to image
        heatmap = np.uint8(255 * cam)

        # Apply color map
        colored_heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        colored_heatmap = cv2.cvtColor(
            colored_heatmap,
            cv2.COLOR_BGR2RGB
        )

        # Prepare original image
        original = original_image.convert("RGB").resize(
            (224, 224)
        )

        original = np.array(original)

        # Combine original image and heatmap
        overlay = cv2.addWeighted(
            original,
            0.6,
            colored_heatmap,
            0.4,
            0
        )

        return (
            predicted_class.item(),
            confidence.item() * 100,
            heatmap,
            overlay
        )

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
