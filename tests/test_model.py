import unittest
import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model
from src.csrnet_implementation import CSRNet


class TestCSRNetModel(unittest.TestCase):
    def setUp(self):
        # Create a small model for testing
        self.model = CSRNet(load_weights=False)

        # Create a dummy input
        self.dummy_input = torch.randn(1, 3, 224, 224)

    def test_model_initialization(self):
        """Test model initialization"""
        # Check if model is created successfully
        self.assertIsNotNone(self.model)

        # Check if model has correct frontend and backend
        self.assertTrue(hasattr(self.model, 'frontend'))
        self.assertTrue(hasattr(self.model, 'backend'))
        self.assertTrue(hasattr(self.model, 'output_layer'))

    def test_model_forward_pass(self):
        """Test model forward pass"""
        # Set model to evaluation mode
        self.model.eval()

        # Perform forward pass
        with torch.no_grad():
            output = self.model(self.dummy_input)

        # Check output shape (should be smaller than input due to downsampling)
        # VGG16 frontend has 3 max pooling layers (1/8 of original size)
        expected_shape = (1, 1, 224 // 8, 224 // 8)
        self.assertEqual(output.shape, expected_shape)

        # Check output values (should be non-negative for density map)
        self.assertTrue(torch.all(output >= 0))

    def test_model_parameters(self):
        """Test model parameters"""
        # Check if model has parameters
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)

        # Check if parameters are trainable
        for param in params:
            self.assertTrue(param.requires_grad)

    def test_model_with_pretrained_weights(self):
        """Test model with pretrained weights (if available)"""
        # Check if pretrained VGG16 weights are available (requires internet connection)
        try:
            pretrained_model = CSRNet(load_weights=True)

            # VGG16 frontend parameters should be initialized
            for name, param in pretrained_model.frontend.named_parameters():
                if 'weight' in name:
                    # Weights should not be near zero if pretrained
                    self.assertGreater(param.abs().mean().item(), 1e-5)
        except:
            # Skip test if pretrained weights cannot be loaded
            print("Skipping pretrained weights test (could not load weights)")
            pass

    def test_output_layer(self):
        """Test output layer configuration"""
        # Output layer should be 1x1 convolution to produce density map
        self.assertEqual(self.model.output_layer.kernel_size[0], 1)
        self.assertEqual(self.model.output_layer.kernel_size[1], 1)
        self.assertEqual(self.model.output_layer.out_channels, 1)

    def test_model_with_batch(self):
        """Test model with batch input"""
        # Create batch input
        batch_size = 4
        batch_input = torch.randn(batch_size, 3, 224, 224)

        # Set model to evaluation mode
        self.model.eval()

        # Perform forward pass
        with torch.no_grad():
            output = self.model(batch_input)

        # Check output shape
        expected_shape = (batch_size, 1, 224 // 8, 224 // 8)
        self.assertEqual(output.shape, expected_shape)

    def test_model_with_different_input_sizes(self):
        """Test model with different input sizes"""
        input_sizes = [(320, 240), (640, 480), (480, 320)]

        # Set model to evaluation mode
        self.model.eval()

        for size in input_sizes:
            # Create input with specific size
            h, w = size
            test_input = torch.randn(1, 3, h, w)

            # Perform forward pass
            with torch.no_grad():
                output = self.model(test_input)

            # Check output shape
            expected_shape = (1, 1, h // 8, w // 8)
            self.assertEqual(output.shape, expected_shape)

            # Sum output to get count
            count = output.sum().item()

            # Count should be a non-negative float
            self.assertGreaterEqual(count, 0)
            self.assertIsInstance(count, float)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_on_gpu(self):
        """Test model on GPU (if available)"""
        # Move model to GPU
        model = self.model.to('cuda')

        # Create input on GPU
        input_gpu = self.dummy_input.to('cuda')

        # Perform forward pass
        with torch.no_grad():
            output = model(input_gpu)

        # Check output device
        self.assertEqual(output.device.type, 'cuda')

        # Check output shape
        expected_shape = (1, 1, 224 // 8, 224 // 8)
        self.assertEqual(output.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
