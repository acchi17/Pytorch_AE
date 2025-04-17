import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from ae_torch import Autoencoder

# Data preprocessing settings (using same settings as training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load test data
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Model construction (using same settings as training)
AE = Autoencoder(
    image_channels = 1,
    encoder_conv_filters = [32,64,64,64],
    encoder_conv_kernel_size = [3,3,3,3],
    encoder_conv_strides = [1,2,2,1],
    decoder_conv_t_filters = [64,64,32,1],
    decoder_conv_t_kernel_size = [3,3,3,3],
    decoder_conv_t_strides = [1,2,2,1],
    z_dim = 2,
    use_batch_norm=True,
    use_dropout=True
).to(device)

# Initialize dynamic layers by performing a forward pass with dummy data
dummy_input = torch.randn(1, 1, 28, 28).to(device)
_ = AE(dummy_input)

# Load trained model
checkpoint = torch.load('best_autoencoder.pth')
AE.load_state_dict(checkpoint['model_state_dict'])
AE.eval()  # Evaluation mode (Default is training mode)

def denormalize(x):
    """Function to denormalize values (from -1,1 to 0,1)"""
    return (x + 1) / 2

def visualize_reconstruction(model, image_idx=0):
    """Function to display image reconstruction results"""
    # Get input image
    image, _ = test_dataset[image_idx]
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Reconstruction
    with torch.no_grad():
        reconstructed = model(image)
    
    # Process for display
    image = denormalize(image.cpu().squeeze())
    reconstructed = denormalize(reconstructed.cpu().squeeze())
    
    # Display results
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Reconstructed')
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Test with multiple images
for i in range(5):  # Display 5 images
    visualize_reconstruction(AE, i)
    input("Press Enter to see next image...")  # Press Enter to view next image
