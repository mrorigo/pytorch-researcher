"""
Tests for the _get_input_size_from_config function.

Tests the helper function that extracts input dimensions from planning LLM configurations
to enable dynamic sandbox validation with correct input sizes.
"""

from pytorch_researcher.src.planning_llm.client import _get_input_size_from_config


class TestInputSizeExtraction:
    """Test the input size extraction from various model config formats."""
    
    def test_mnist_config_extraction(self):
        """Test extracting input size from MNIST CNN configuration."""
        mnist_config = {
            "architecture": "CNN",
            "layers": [
                {
                    "type": "Conv2D",
                    "filters": 32,
                    "kernel_size": 3,
                    "activation": "relu",
                    "input_shape": [28, 28, 1]
                },
                {
                    "type": "MaxPooling2D",
                    "pool_size": 2
                },
                {
                    "type": "Conv2D",
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "relu"
                },
                {
                    "type": "MaxPooling2D",
                    "pool_size": 2
                },
                {
                    "type": "Flatten"
                },
                {
                    "type": "Dense",
                    "units": 128,
                    "activation": "relu"
                },
                {
                    "type": "Dense",
                    "units": 10,
                    "activation": "softmax"
                }
            ],
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy"
        }
        
        result = _get_input_size_from_config(mnist_config)
        expected = (1, 1, 28, 28)  # batch_size, channels, height, width
        
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_cifar10_config_extraction(self):
        """Test extracting input size from CIFAR-10 CNN configuration."""
        cifar10_config = {
            "architecture": "CNN",
            "layers": [
                {
                    "type": "Conv2D",
                    "filters": 32,
                    "kernel_size": 3,
                    "activation": "relu",
                    "input_shape": [32, 32, 3]
                },
                {
                    "type": "MaxPooling2D",
                    "pool_size": 2
                },
                {
                    "type": "Conv2D",
                    "filters": 64,
                    "kernel_size": 3,
                    "activation": "relu"
                },
                {
                    "type": "MaxPooling2D",
                    "pool_size": 2
                },
                {
                    "type": "Flatten"
                },
                {
                    "type": "Dense",
                    "units": 128,
                    "activation": "relu"
                },
                {
                    "type": "Dense",
                    "units": 10,
                    "activation": "softmax"
                }
            ],
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy"
        }
        
        result = _get_input_size_from_config(cifar10_config)
        expected = (1, 3, 32, 32)  # batch_size, channels, height, width
        
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_config_with_input_shape_at_top_level(self):
        """Test extracting input size when input_shape is at top level."""
        config = {
            "architecture": "CNN",
            "input_shape": [64, 64, 3],  # Top-level input shape
            "layers": [
                {
                    "type": "Conv2D",
                    "filters": 32,
                    "kernel_size": 3,
                    "activation": "relu"
                },
                {
                    "type": "Flatten"
                },
                {
                    "type": "Dense",
                    "units": 128,
                    "activation": "relu"
                },
                {
                    "type": "Dense",
                    "units": 10,
                    "activation": "softmax"
                }
            ],
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy"
        }
        
        result = _get_input_size_from_config(config)
        expected = (1, 3, 64, 64)  # batch_size, channels, height, width
        
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_config_with_multiple_layers_first_has_input_shape(self):
        """Test that we pick the first layer with input_shape."""
        config = {
            "architecture": "CNN",
            "layers": [
                {
                    "type": "Conv2D",
                    "filters": 16,
                    "kernel_size": 5,
                    "activation": "relu",
                    "input_shape": [224, 224, 3]  # First layer has input shape
                },
                {
                    "type": "MaxPooling2D",
                    "pool_size": 2
                },
                {
                    "type": "Conv2D",
                    "filters": 32,
                    "kernel_size": 3,
                    "activation": "relu",
                    "input_shape": [112, 112, 16]  # Later layer also has input shape
                }
            ],
            "optimizer": "adam",
            "loss": "categorical_crossentropy"
        }
        
        result = _get_input_size_from_config(config)
        expected = (1, 3, 224, 224)  # Should use first layer's input_shape
        
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_config_with_no_input_shape_fallback(self):
        """Test fallback to default when no input_shape found."""
        config = {
            "architecture": "CNN",
            "layers": [
                {
                    "type": "Conv2D",
                    "filters": 32,
                    "kernel_size": 3,
                    "activation": "relu"
                    # No input_shape
                },
                {
                    "type": "Flatten"
                },
                {
                    "type": "Dense",
                    "units": 128,
                    "activation": "relu"
                },
                {
                    "type": "Dense",
                    "units": 10,
                    "activation": "softmax"
                }
            ],
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy"
        }
        
        result = _get_input_size_from_config(config)
        expected = (1, 3, 32, 32)  # Default CIFAR-10 size
        
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_empty_config_fallback(self):
        """Test fallback behavior with empty or invalid config."""
        # Empty config
        result = _get_input_size_from_config({})
        expected = (1, 3, 32, 32)
        assert result == expected, f"Expected {expected}, got {result}"
        
        # None config
        result = _get_input_size_from_config(None)
        expected = (1, 3, 32, 32)
        assert result == expected, f"Expected {expected}, got {result}"
        
        # Config with no layers
        result = _get_input_size_from_config({"architecture": "CNN"})
        expected = (1, 3, 32, 32)
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_invalid_input_shape_formats(self):
        """Test handling of invalid input_shape formats."""
        # Too few dimensions
        config1 = {
            "layers": [
                {
                    "type": "Conv2D",
                    "input_shape": [28, 28]  # Only 2D, missing channels
                }
            ]
        }
        result1 = _get_input_size_from_config(config1)
        expected = (1, 3, 32, 32)  # Should fall back to default
        assert result1 == expected, f"Expected {expected}, got {result1}"
        
        # Wrong type
        config2 = {
            "layers": [
                {
                    "type": "Conv2D",
                    "input_shape": "28x28x3"  # String instead of list
                }
            ]
        }
        result2 = _get_input_size_from_config(config2)
        expected = (1, 3, 32, 32)  # Should fall back to default
        assert result2 == expected, f"Expected {expected}, got {result2}"
        
        # Empty list
        config3 = {
            "layers": [
                {
                    "type": "Conv2D",
                    "input_shape": []  # Empty list
                }
            ]
        }
        result3 = _get_input_size_from_config(config3)
        expected = (1, 3, 32, 32)  # Should fall back to default
        assert result3 == expected, f"Expected {expected}, got {result3}"
    
    def test_4d_input_shape(self):
        """Test handling of 4D input shapes (batch, height, width, channels)."""
        config = {
            "layers": [
                {
                    "type": "Conv2D",
                    "input_shape": [None, 28, 28, 1]  # 4D with batch dimension
                }
            ]
        }
        
        result = _get_input_size_from_config(config)
        expected = (1, 1, 28, 28)  # Should extract last 3 dimensions
        
        assert result == expected, f"Expected {expected}, got {result}"
    
    def test_grayscale_vs_rgb_extraction(self):
        """Test that we correctly distinguish between grayscale (1 channel) and RGB (3 channel)."""
        # Grayscale MNIST
        grayscale_config = {
            "layers": [
                {
                    "type": "Conv2D",
                    "input_shape": [28, 28, 1]
                }
            ]
        }
        grayscale_result = _get_input_size_from_config(grayscale_config)
        assert grayscale_result == (1, 1, 28, 28)
        
        # RGB ImageNet-like
        rgb_config = {
            "layers": [
                {
                    "type": "Conv2D",
                    "input_shape": [224, 224, 3]
                }
            ]
        }
        rgb_result = _get_input_size_from_config(rgb_config)
        assert rgb_result == (1, 3, 224, 224)
        
        # Multi-spectral (4 channels)
        multi_config = {
            "layers": [
                {
                    "type": "Conv2D",
                    "input_shape": [64, 64, 4]
                }
            ]
        }
        multi_result = _get_input_size_from_config(multi_config)
        assert multi_result == (1, 4, 64, 64)


if __name__ == "__main__":
    # Run the tests
    test_instance = TestInputSizeExtraction()
    
    # Test each method
    print("Testing MNIST config extraction...")
    test_instance.test_mnist_config_extraction()
    print("✓ MNIST config extraction passed")
    
    print("Testing CIFAR-10 config extraction...")
    test_instance.test_cifar10_config_extraction()
    print("✓ CIFAR-10 config extraction passed")
    
    print("Testing top-level input_shape...")
    test_instance.test_config_with_input_shape_at_top_level()
    print("✓ Top-level input_shape passed")
    
    print("Testing multiple layers with input_shape...")
    test_instance.test_config_with_multiple_layers_first_has_input_shape()
    print("✓ Multiple layers test passed")
    
    print("Testing fallback behavior...")
    test_instance.test_config_with_no_input_shape_fallback()
    print("✓ Fallback behavior passed")
    
    print("Testing empty config fallback...")
    test_instance.test_empty_config_fallback()
    print("✓ Empty config fallback passed")
    
    print("Testing invalid input_shape formats...")
    test_instance.test_invalid_input_shape_formats()
    print("✓ Invalid formats test passed")
    
    print("Testing 4D input shapes...")
    test_instance.test_4d_input_shape()
    print("✓ 4D input shapes passed")
    
    print("Testing grayscale vs RGB extraction...")
    test_instance.test_grayscale_vs_rgb_extraction()
    print("✓ Grayscale vs RGB extraction passed")
    
    print("\n🎉 All tests passed! Input size extraction is working correctly.")