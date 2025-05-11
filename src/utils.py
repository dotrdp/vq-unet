def log_message(message):
    print(f"[LOG] {message}")

def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    log_message(f"Checkpoint saved at {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    log_message(f"Checkpoint loaded from {filepath}, epoch: {epoch}, loss: {loss}")
    return epoch, loss

def visualize_reconstruction(original, reconstructed, num_images=8):
    plt.figure(figsize=(20, 5))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original[i].cpu().numpy().transpose(1, 2, 0), cmap='gray')
        plt.axis('off')
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed[i].cpu().numpy().transpose(1, 2, 0), cmap='gray')
        plt.axis('off')
    plt.show()

def calculate_psnr(original, reconstructed):
    mse = F.mse_loss(original, reconstructed)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()