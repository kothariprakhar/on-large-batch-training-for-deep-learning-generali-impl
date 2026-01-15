import unittest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Assuming the user's code is imported as 'module' or classes are available.
# For this standalone test, we assume the classes SimpleVGG and function compute_sharpness_curve are available.
# We will reference the class SimpleVGG and function compute_sharpness_curve from the provided code context.

# --- Mocks/Re-definitions for Testing if not directly importable ---
# Ideally, one would import these. We replicate the minimal necessary structure if imports fail in a real scenario.
# Here we test the provided class structure directly.

class TestGeneralizationGapCode(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = SimpleVGG().to(self.device)

    def test_model_architecture_shapes(self):
        """Test if SimpleVGG handles CIFAR-10 shape (3, 32, 32) correctly."""
        batch_size = 4
        # CIFAR images are 3x32x32
        dummy_input = torch.randn(batch_size, 3, 32, 32).to(self.device)
        output = self.model(dummy_input)
        
        # Expected output: [batch_size, 10]
        self.assertEqual(output.shape, (batch_size, 10), 
                         f"Output shape mismatch. Expected ({batch_size}, 10), got {output.shape}")

    def test_forward_backward_pass(self):
        """Test valid gradient flow."""
        dummy_input = torch.randn(2, 3, 32, 32).to(self.device)
        dummy_target = torch.tensor([0, 1]).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        self.model.train()
        optimizer.zero_grad()
        outputs = self.model(dummy_input)
        loss = criterion(outputs, dummy_target)
        loss.backward()
        optimizer.step()
        
        # Check if gradients were generated
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient.")

    def test_sharpness_curve_logic(self):
        """Test if sharpness calculation runs and restores parameters."""
        # Create a dummy loader
        dummy_data = torch.randn(10, 3, 32, 32)
        dummy_labels = torch.randint(0, 10, (10,))
        dataset = TensorDataset(dummy_data, dummy_labels)
        loader = DataLoader(dataset, batch_size=5)
        
        # Clone original params to verify restoration
        orig_params = [p.clone() for p in self.model.parameters()]
        
        # Run function
        alphas = [-0.1, 0.0, 0.1]
        losses = compute_sharpness_curve(self.model, loader, alphas)
        
        # 1. Check output type/length
        self.assertIsInstance(losses, list)
        self.assertEqual(len(losses), 3)
        
        # 2. Check parameter restoration
        for p_orig, p_curr in zip(orig_params, self.model.parameters()):
            self.assertTrue(torch.allclose(p_orig, p_curr), "Parameters were not restored after sharpness check.")

    def test_normalization_logic(self):
        """Verify the filter normalization logic handles shapes correctly without crashing."""
        # We manually invoke the logic used in compute_sharpness_curve to ensure no broadcasting errors
        direction = []
        for p in self.model.parameters():
            d = torch.randn_like(p)
            if len(p.shape) == 4: # Conv
                n_filters = p.shape[0]
                for i in range(n_filters):
                    n = p[i].norm()
                    dn = d[i].norm()
                    # This operation is the critical shape check
                    d[i] = d[i] * (n / (dn + 1e-6))
            else: # FC
                 n = p.norm()
                 dn = d.norm()
                 d = d * (n / (dn + 1e-6))
            direction.append(d)
        self.assertTrue(len(direction) == len(list(self.model.parameters())))

if __name__ == '__main__':
    unittest.main()