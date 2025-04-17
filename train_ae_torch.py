import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torchinfo
import torch.nn as nn
import torch.optim as optim
from ae_torch import Autoencoder

EPOCHS = 100
BATCH_SIZE = 128  # Increased batch size for faster learning

# Data preprocessing settings
transform = transforms.Compose([
    transforms.ToTensor(),               # Convert to Tensor and scale pixel values ​​to range 0 to 1
    transforms.Normalize((0.5,), (0.5,)) # Scale pixel values ​​to range -1 to 1
])
# Load training data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# print(f'train data num:{len(train_dataset)}') # 60000
# image, label = train_dataset[0]
# print(f'train image type:{type(image)}') # torch.Tensor
# print(f'train label size:{type(label)}') # int
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# data_iter = iter(train_loader)
# data, target = next(data_iter)
# print(data.size())   # torch.Size([BATCH_SIZE, 1, 28, 28])
# print(target.size()) # torch.Size([BATCH_SIZE])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

AE = Autoencoder(
    image_channels = 1,
    encoder_conv_filters = [32,64,64,64],
    encoder_conv_kernel_size = [3,3,3,3],
    encoder_conv_strides = [1,2,2,1],
    decoder_conv_t_filters = [64,64,32,1],
    decoder_conv_t_kernel_size = [3,3,3,3],
    decoder_conv_t_strides = [1,2,2,1],
    z_dim = 2,
    use_batch_norm=True,   # Enable batch normalization
    use_dropout=True       # Enable dropout
).to(device)

torchinfo.summary(AE, input_size=(32, 1, 28, 28))

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(AE.parameters(), lr=0.001, weight_decay=1e-5)

# Add learning rate scheduler to adjust learning rate during training
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=5
)

# Initialize best loss for model saving
best_loss = float('inf')
best_model_path = 'best_autoencoder.pth'

# Training loop
for epoch in range(EPOCHS):
    AE.train()
    running_loss = 0.0
    for data, _ in train_loader:
        data = data.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = AE(data)
        
        # Compute loss
        loss = criterion(outputs, data)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate average loss for this epoch
    epoch_loss = running_loss/len(train_loader)
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Learning Rate: {current_lr:.6f}')
    
    # Adjust learning rate based on loss
    scheduler.step(epoch_loss)
    
    # Save model if it has the best loss
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': AE.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, best_model_path)
        print(f'Model saved with loss: {best_loss:.4f}')

# Save final model
final_model_path = 'final_autoencoder.pth'
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': AE.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
}, final_model_path)
print(f'Final model saved at: {final_model_path}')
