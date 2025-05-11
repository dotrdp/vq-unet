import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, num_epochs, checkpoint_dir, vq_loss_weight=1.0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion # This will be for reconstruction loss, e.g., MSELoss
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.vq_loss_weight = vq_loss_weight

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            running_recon_loss = 0.0
            running_vq_loss = 0.0
            running_total_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f'Training Epoch {epoch + 1}/{self.num_epochs}')
            for images, targets in progress_bar: # Assuming targets are also images for reconstruction
                images, targets = images.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                
                reconstructions, vq_loss = self.model(images)
                
                recon_loss = self.criterion(reconstructions, targets)
                total_loss = recon_loss + self.vq_loss_weight * vq_loss
                
                total_loss.backward()
                self.optimizer.step()

                running_recon_loss += recon_loss.item()
                running_vq_loss += vq_loss.item()
                running_total_loss += total_loss.item()
                
                progress_bar.set_postfix({
                    'Recon Loss': f'{recon_loss.item():.4f}',
                    'VQ Loss': f'{vq_loss.item():.4f}',
                    'Total Loss': f'{total_loss.item():.4f}'
                })

            avg_recon_loss = running_recon_loss / len(self.train_loader)
            avg_vq_loss = running_vq_loss / len(self.train_loader)
            avg_total_loss = running_total_loss / len(self.train_loader)
            
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Avg Recon Loss: {avg_recon_loss:.4f}, Avg VQ Loss: {avg_vq_loss:.4f}, Avg Total Loss: {avg_total_loss:.4f}')
            
            if self.val_loader:
                self.validate(epoch)

            self.save_checkpoint(epoch)

    def validate(self, epoch): # Added epoch for context in print
        self.model.eval()
        val_recon_loss = 0.0
        val_vq_loss = 0.0
        val_total_loss = 0.0
        
        progress_bar = tqdm(self.val_loader, desc=f'Validating Epoch {epoch + 1}/{self.num_epochs}')
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                reconstructions, vq_loss_val = self.model(images)
                
                recon_loss_val = self.criterion(reconstructions, targets)
                total_loss_val = recon_loss_val + self.vq_loss_weight * vq_loss_val

                val_recon_loss += recon_loss_val.item()
                val_vq_loss += vq_loss_val.item()
                val_total_loss += total_loss_val.item()

                progress_bar.set_postfix({
                    'Val Recon Loss': f'{recon_loss_val.item():.4f}',
                    'Val VQ Loss': f'{vq_loss_val.item():.4f}'
                })

        avg_val_recon_loss = val_recon_loss / len(self.val_loader)
        avg_val_vq_loss = val_vq_loss / len(self.val_loader)
        avg_val_total_loss = val_total_loss / len(self.val_loader)

        print(f'Validation Epoch [{epoch + 1}/{self.num_epochs}], Avg Recon Loss: {avg_val_recon_loss:.4f}, Avg VQ Loss: {avg_val_vq_loss:.4f}, Avg Total Loss: {avg_val_total_loss:.4f}')
        return avg_val_total_loss


    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        # Save more than just model state if needed (e.g., optimizer, epoch)
        save_obj = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_obj, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f'Checkpoint loaded: {checkpoint_path}, starting from epoch {start_epoch}')
            return start_epoch
        else:
            print(f'Checkpoint not found: {checkpoint_path}')
            return 0