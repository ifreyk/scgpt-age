"""
Improved ClsDecoder - Enhanced version of ClsDecoder with additional features:
- Residual connections (skip connections)
- Dropout for regularization
- Configurable hidden dimensions
- Optional batch normalization
- Better initialization
"""

import torch
from torch import nn
from typing import Optional, Callable


class ImprovedCls(nn.Module):
    """
    Improved classification decoder with residual connections, dropout, and better architecture.
    
    Features:
    - Residual connections for better gradient flow
    - Dropout for regularization
    - Configurable hidden dimensions
    - Optional batch normalization
    - Better weight initialization
    """
    
    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 5,
        activation: Callable = nn.GELU,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_batch_norm: bool = False,
        hidden_dim: Optional[int] = None,
        layer_norm_eps: float = 1e-5,
    ):
        """
        Args:
            d_model: Input embedding dimension
            n_cls: Number of output classes
            nlayers: Number of hidden layers (excluding output layer)
            activation: Activation function class (e.g., nn.GELU, nn.ReLU)
            dropout: Dropout probability
            use_residual: Whether to use residual connections
            use_batch_norm: Whether to use batch normalization (instead of layer norm)
            hidden_dim: Hidden layer dimension (default: same as d_model)
            layer_norm_eps: Epsilon for layer normalization
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_cls = n_cls
        self.nlayers = nlayers
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.hidden_dim = hidden_dim if hidden_dim is not None else d_model
        
        # Build decoder layers
        self._decoder = nn.ModuleList()
        
        for i in range(nlayers):
            # Input dimension for this layer
            in_dim = d_model if i == 0 else self.hidden_dim
            out_dim = self.hidden_dim
            
            # Linear layer
            linear = nn.Linear(in_dim, out_dim)
            self._decoder.append(linear)
            
            # Normalization
            if use_batch_norm:
                self._decoder.append(nn.BatchNorm1d(out_dim))
            else:
                self._decoder.append(nn.LayerNorm(out_dim, eps=layer_norm_eps))
            
            # Activation
            self._decoder.append(activation())
            
            # Dropout (except for last layer)
            if i < nlayers - 1:
                self._decoder.append(nn.Dropout(dropout))
        
        # Output layer
        self.out_layer = nn.Linear(self.hidden_dim, n_cls)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the improved classifier.
        
        Args:
            x: Input tensor of shape [batch_size, d_model]
        
        Returns:
            Output tensor of shape [batch_size, n_cls]
        """
        residual = x if self.use_residual and self.hidden_dim == self.d_model else None
        
        # Process through decoder layers
        layer_idx = 0
        for i in range(self.nlayers):
            # Linear layer
            x = self._decoder[layer_idx](x)
            layer_idx += 1
            
            # Normalization
            norm_layer = self._decoder[layer_idx]
            if self.use_batch_norm:
                # BatchNorm expects (N, C) or (N, C, ...)
                x = norm_layer(x)
            else:
                x = norm_layer(x)
            layer_idx += 1
            
            # Activation
            x = self._decoder[layer_idx](x)
            layer_idx += 1
            
            # Residual connection (if enabled and dimensions match)
            if self.use_residual and residual is not None and i == 0:
                # Only apply residual on first layer if dimensions match
                if x.shape == residual.shape:
                    x = x + residual
            
            # Dropout (if not last layer)
            if layer_idx < len(self._decoder):
                x = self._decoder[layer_idx](x)
                layer_idx += 1
        
        # Output layer
        return self.out_layer(x)

class SimpleCls(nn.Module):
    def __init__(self, d_model, n_cls, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_cls),
        )

    def forward(self, x):  # x: [B, d_model] (cell_emb)
        return self.net(x)

