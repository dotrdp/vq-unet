import torch.nn as nn
import torch
import numpy as np

class VQUNet(nn.Module):
    def __init__(self, in_channels, out_channels, codebook_size, encoder_channel_dims, commitment_cost=0.25):
        super(VQUNet, self).__init__()
        self.commitment_cost = commitment_cost
        self.encoder_channel_dims = encoder_channel_dims # e.g., [64, 128, 256, 512]

        # Encoder
        self.enc_blocks = nn.ModuleList()
        current_channels = in_channels
        for i, channels in enumerate(encoder_channel_dims):
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels = channels
        
        self.codebooks = nn.ModuleList()
        for emb_dim in encoder_channel_dims:
            self.codebooks.append(nn.Embedding(codebook_size, emb_dim))

        # Decoder
        self.dec_blocks = nn.ModuleList()
        # Decoder stages: len(encoder_channel_dims)
        # Bottleneck is encoder_channel_dims[-1]
        # First decoder block upsamples from bottleneck, concatenates with skip from encoder_channel_dims[-2]
        
        reversed_enc_dims = encoder_channel_dims[::-1] # [512, 256, 128, 64] for example

        current_channels = reversed_enc_dims[0] # Bottleneck channels
        for i in range(len(reversed_enc_dims)):
            is_last_block = (i == len(reversed_enc_dims) - 1)
            
            # Channels for ConvTranspose2d
            # Input: current_channels from previous decoder layer (or bottleneck)
            # Output: target_out_channels for this upsampling stage
            # If not the first decoder block (i.e., not from bottleneck directly),
            # current_channels already includes concatenated skip from previous stage.
            # This needs careful thought. Let's use a U-Net like structure: Upsample -> Concat -> Conv
            
            # Upsampling layer
            up_in_channels = current_channels
            if i == 0: # Bottleneck
                 up_out_channels = reversed_enc_dims[i] # Stays same, or target next skip dim
            else: # After concat and conv
                 up_out_channels = reversed_enc_dims[i]


            upsample_layer = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=4, stride=2, padding=1)
            
            # Convolution layer after concatenation
            # Input to conv: up_out_channels (from upsample) + skip_channels (reversed_enc_dims[i])
            # (Exception: first decoder block uses reversed_enc_dims[i+1] as skip if i=0)
            # Output of conv: reversed_enc_dims[i] (target for this stage)
            # If it's the final block, output is `out_channels`
            
            conv_in_channels = up_out_channels
            if not is_last_block: # If there's a skip connection to concat
                 skip_channel_dim = reversed_enc_dims[i+1] # Skip from corresponding encoder level
                 conv_in_channels += skip_channel_dim
            
            conv_out_channels = up_out_channels # Maintain dim before final layer
            
            if is_last_block: # Final output layer
                # The last upsample should go to out_channels directly if no further conv
                # Let's make the last ConvTranspose2d output `out_channels`
                upsample_layer = nn.ConvTranspose2d(up_in_channels, out_channels, kernel_size=4, stride=2, padding=1)
                conv_layer = nn.Tanh() # Activation after final upsample
            else:
                conv_layer = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(conv_in_channels, up_out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )

            self.dec_blocks.append(nn.ModuleDict({
                'upsample': upsample_layer,
                'conv': conv_layer
            }))
            
            current_channels = up_out_channels # Output of conv block becomes input for next upsample

    def _quantize(self, z_e, codebook_idx):
        codebook = self.codebooks[codebook_idx]
        b, c, h, w = z_e.shape
        
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, c)
        
        distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) + \
                    torch.sum(codebook.weight**2, dim=1) - \
                    2 * torch.matmul(z_e_flat, codebook.weight.t())
        
        indices = torch.argmin(distances, dim=1)
        z_q_flat = codebook(indices)
        z_q = z_q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        
        loss = self.commitment_cost * torch.mean((z_q.detach() - z_e)**2) + \
               torch.mean((z_q - z_e.detach())**2)
        
        z_q = z_e + (z_q - z_e).detach() # Straight-through estimator
        return z_q, loss

    def forward(self, x):
        skip_connections_quantized = []
        total_vq_loss = 0.0

        # Encoder pass & Quantization
        current_features = x
        for i, enc_block in enumerate(self.enc_blocks):
            encoded_features = enc_block(current_features)
            quantized_features, vq_loss = self._quantize(encoded_features, i)
            skip_connections_quantized.append(quantized_features)
            total_vq_loss += vq_loss
            current_features = encoded_features # Pass unquantized features to next encoder stage

        # The last element in skip_connections_quantized is the quantized bottleneck
        decoder_input = skip_connections_quantized[-1]
        
        # Decoder pass
        # Skips are used in reverse order, and the bottleneck itself is the first input to decoder
        # So, actual skips to be concatenated are from skip_connections_quantized[:-1] in reverse
        
        reversed_quantized_skips_for_concat = skip_connections_quantized[:-1][::-1]

        current_features = decoder_input # Start with quantized bottleneck
        
        for i, dec_m_dict in enumerate(self.dec_blocks):
            upsample_layer = dec_m_dict['upsample']
            conv_block = dec_m_dict['conv']

            current_features = upsample_layer(current_features)

            if i < len(reversed_quantized_skips_for_concat): # If there's a skip to concat
                skip_to_concat = reversed_quantized_skips_for_concat[i]
                # Ensure spatial dimensions match for concatenation
                if current_features.shape[2:] != skip_to_concat.shape[2:]:
                    # This can happen if padding/stride isn't perfect.
                    # A common fix is to crop the larger feature map.
                    # For simplicity, we assume ConvTranspose2d output matches skip dimensions.
                    # If not, an adaptive pooling or cropping might be needed.
                    # For now, let's assume they match.
                    pass # Add cropping/padding if necessary
                current_features = torch.cat((current_features, skip_to_concat), dim=1)
            
            current_features = conv_block(current_features) # This is either Conv layers or Tanh for last

        reconstructed_x = current_features
        return reconstructed_x, total_vq_loss