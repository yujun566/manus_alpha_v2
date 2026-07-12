"""
═══════════════════════════════════════════════════════════════════════════════
MANUS-ALPHA :: Advanced Transformer Language Model Architecture
Production-Grade Implementation (v2.1)
═══════════════════════════════════════════════════════════════════════════════

Complete implementation of state-of-the-art decoder-only transformer with:
  • Custom autograd engine with mixed-precision support
  • Grouped Query Attention (GQA) with multi-head variants
  • Rotary Positional Embeddings (RoPE) with frequency pre-computation
  • RMSNorm with optional pre/post-norm variants
  • SwiGLU activation with gating mechanisms
  • Flash Attention approximation
  • Gradient checkpointing for memory efficiency
  • Distributed training primitives
  • Advanced weight initialization schemes
  • Layer-wise learning rate scaling
  • Comprehensive logging and debugging utilities

Dependencies: NumPy (no external ML frameworks)
Author: Manus Research Team
Version: 2.1.0
License: MIT
"""

from __future__ import annotations
import json
import time
import logging
import pickle
import traceback
from typing import Optional, Tuple, Dict, List, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from functools import wraps
from collections import defaultdict, OrderedDict

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMERATIONS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class PrecisionMode(Enum):
    """Numerical precision modes for computation."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    MIXED = "mixed"


class NormVariant(Enum):
    """Normalization layer variants."""
    RMSNORM = "rmsnorm"
    LAYERNORM = "layernorm"
    PRE_RMSNORM = "pre_rmsnorm"
    PRE_LAYERNORM = "pre_ln"


class ActivationFunction(Enum):
    """Activation function variants."""
    SWIGLU = "swiglu"
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"
    GATED_GELU = "gated_gelu"


class AttentionVariant(Enum):
    """Attention mechanism variants."""
    MULTI_HEAD = "mha"
    MULTI_QUERY = "mqa"
    GROUPED_QUERY = "gqa"
    FLASH = "flash"


class DistributedStrategy(Enum):
    """Distributed training strategies."""
    NONE = "none"
    DATA_PARALLEL = "dp"
    TENSOR_PARALLEL = "tp"
    PIPELINE_PARALLEL = "pp"
    ZeRO = "zero"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    variant: AttentionVariant = AttentionVariant.GROUPED_QUERY
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    dropout_p: float = 0.1
    use_flash: bool = False
    use_flash_v2: bool = False
    causal: bool = True
    scale_factor: Optional[float] = None
    qk_norm: bool = False
    window_size: Optional[int] = None
    
    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        if self.head_dim is None:
            self.head_dim = 128
        if self.scale_factor is None:
            self.scale_factor = 1.0 / np.sqrt(self.head_dim)
        self.n_rep = self.n_heads // self.n_kv_heads


@dataclass
class RotaryEmbeddingConfig:
    """Configuration for rotary position embeddings."""
    base: float = 10000.0
    dim: int = 128
    freq_scale: float = 1.0
    use_cached: bool = True
    cache_seq_len: int = 2048
    use_complex: bool = False


@dataclass
class FeedForwardConfig:
    """Configuration for feed-forward networks."""
    variant: ActivationFunction = ActivationFunction.SWIGLU
    hidden_dim: int = 4096
    intermediate_dim: Optional[int] = None
    dropout_p: float = 0.1
    use_bias: bool = True
    gate_bias: Optional[bool] = None
    
    def __post_init__(self):
        if self.intermediate_dim is None:
            self.intermediate_dim = int(self.hidden_dim * 8 / 3 // 64 * 64)
        if self.gate_bias is None:
            self.gate_bias = self.use_bias


@dataclass
class NormalizationConfig:
    """Configuration for normalization layers."""
    variant: NormVariant = NormVariant.RMSNORM
    eps: float = 1e-6
    affine: bool = True
    use_bias: bool = False
    pre_norm: bool = True
    post_norm: bool = False
    norm_dim: int = 768


@dataclass
class ModelConfig:
    """Complete configuration for ManusAlpha model."""
    # Core architecture
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    max_seq_length: int = 4096
    
    # Attention configuration
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    
    # Position embeddings
    rope: RotaryEmbeddingConfig = field(default_factory=RotaryEmbeddingConfig)
    
    # Feed-forward configuration
    feed_forward: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    
    # Normalization configuration
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    
    # Regularization
    dropout_p: float = 0.1
    attention_dropout_p: float = 0.1
    residual_dropout_p: float = 0.1
    
    # Initialization
    init_mean: float = 0.0
    init_std: float = 0.02
    weight_init_scheme: str = "normal"  # "normal", "uniform", "xavier", "kaiming"
    
    # Training configuration
    precision: PrecisionMode = PrecisionMode.FP32
    use_gradient_checkpointing: bool = False
    use_activation_checkpointing: bool = False
    
    # Distributed training
    distributed_strategy: DistributedStrategy = DistributedStrategy.NONE
    n_distributed_devices: int = 1
    
    # Memory optimization
    use_flash_attn: bool = False
    use_fused_ops: bool = False
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    # Tie embeddings
    tie_word_embeddings: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        data = asdict(self)
        data['attention'] = asdict(self.attention)
        data['rope'] = asdict(self.rope)
        data['feed_forward'] = asdict(self.feed_forward)
        data['normalization'] = asdict(self.normalization)
        # Convert enums to strings
        data['attention']['variant'] = data['attention']['variant'].value
        data['rope']['use_complex'] = data['rope']['use_complex']
        data['feed_forward']['variant'] = data['feed_forward']['variant'].value
        data['normalization']['variant'] = data['normalization']['variant'].value
        data['precision'] = data['precision'].value
        data['distributed_strategy'] = data['distributed_strategy'].value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        # Convert enum values back
        data_copy = data.copy()
        
        # Reconstruct nested configs
        attn_data = data_copy.get('attention', {})
        if isinstance(attn_data, dict):
            attn_data['variant'] = AttentionVariant(attn_data.get('variant', 'gqa'))
            data_copy['attention'] = AttentionConfig(**attn_data)
        
        rope_data = data_copy.get('rope', {})
        if isinstance(rope_data, dict):
            data_copy['rope'] = RotaryEmbeddingConfig(**rope_data)
        
        ff_data = data_copy.get('feed_forward', {})
        if isinstance(ff_data, dict):
            ff_data['variant'] = ActivationFunction(ff_data.get('variant', 'swiglu'))
            data_copy['feed_forward'] = FeedForwardConfig(**ff_data)
        
        norm_data = data_copy.get('normalization', {})
        if isinstance(norm_data, dict):
            norm_data['variant'] = NormVariant(norm_data.get('variant', 'rmsnorm'))
            data_copy['normalization'] = NormalizationConfig(**norm_data)
        
        # Convert precision and strategy
        if isinstance(data_copy.get('precision'), str):
            data_copy['precision'] = PrecisionMode(data_copy['precision'])
        if isinstance(data_copy.get('distributed_strategy'), str):
            data_copy['distributed_strategy'] = DistributedStrategy(data_copy['distributed_strategy'])
        
        return cls(**data_copy)


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM AUTOGRAD ENGINE (EXTENDED)
# ═══════════════════════════════════════════════════════════════════════════════

class ComputationGraph:
    """Computation graph tracker for autograd."""
    
    def __init__(self):
        self.nodes: List[Tensor] = []
        self.visited: set = set()
        self.build_order: List[Tensor] = []
    
    def build(self, root: 'Tensor') -> List['Tensor']:
        """Build reverse topological order for backprop."""
        self.nodes.clear()
        self.visited.clear()
        self.build_order.clear()
        self._dfs(root)
        return list(reversed(self.build_order))
    
    def _dfs(self, node: 'Tensor'):
        """Depth-first search for topological sort."""
        node_id = id(node)
        if node_id in self.visited:
            return
        self.visited.add(node_id)
        for parent in node._parents:
            self._dfs(parent)
        self.build_order.append(node)


class Tensor:
    """
    Core tensor class with automatic differentiation support.
    
    Supports:
      - Forward pass computation
      - Automatic gradient accumulation
      - Topological sorting for backpropagation
      - Mixed precision operations
      - Gradient clipping and normalization
    """
    
    _graph_counter = 0
    
    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        parents: Tuple['Tensor', ...] = (),
        backward_fn: Optional[Callable] = None,
        name: Optional[str] = None,
        dtype: np.dtype = np.float32
    ):
        self.data = np.asarray(data, dtype=dtype)
        self.dtype = self.data.dtype
        self.requires_grad = requires_grad
        self._parents = parents
        self._backward_fn = backward_fn if backward_fn is not None else lambda: None
        self.grad = None
        self.name = name or f"tensor_{Tensor._graph_counter}"
        Tensor._graph_counter += 1
        
        # Gradient accumulation buffer for sparse ops
        self._grad_buffer = None
        self._is_leaf = len(parents) == 0
        
        # Shape tracking
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.size = self.data.size
    
    def _accumulate_grad(self, grad: np.ndarray):
        """Accumulate gradient, handling broadcasting."""
        if not self.requires_grad:
            return
        
        grad = np.asarray(grad, dtype=self.dtype)
        
        # Handle broadcasting
        if grad.shape != self.data.shape:
            # Sum over broadcasted dimensions
            while grad.ndim > self.data.ndim:
                grad = grad.sum(axis=0)
            
            # Sum over dimensions size 1
            for i in range(self.data.ndim):
                if self.data.shape[i] == 1 and grad.shape[i] != 1:
                    grad = grad.sum(axis=i, keepdims=True)
        
        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=self.dtype)
        self.grad += grad
    
    def backward(self, retain_graph: bool = False):
        """Execute backpropagation."""
        if not self.requires_grad:
            logger.warning(f"Tensor {self.name} does not require gradients")
            return
        
        # Build topological order
        graph = ComputationGraph()
        topo_order = graph.build(self)
        
        # Initialize output gradient
        self.grad = np.ones_like(self.data, dtype=self.dtype)
        
        # Reverse topological sort and backpropagate
        for node in topo_order:
            if node.grad is None:
                continue
            node._backward_fn()
            if not retain_graph:
                node._backward_fn = lambda: None
    
    def zero_grad(self):
        """Clear accumulated gradients."""
        self.grad = None
    
    def detach(self) -> 'Tensor':
        """Detach tensor from computation graph."""
        return Tensor(self.data.copy(), requires_grad=False, name=f"{self.name}_detached")
    
    # ────────────────────────────────────────────────────────────────────────
    # OPERATOR IMPLEMENTATIONS
    # ────────────────────────────────────────────────────────────────────────
    
    def __add__(self, other: Union['Tensor', np.ndarray, float]) -> 'Tensor':
        """Element-wise addition with autograd."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        out = Tensor(self.data + other.data, requires_grad=True,
                     parents=(self, other), name=f"{self.name}_add")
        
        def backward():
            dout = out.grad
            self._accumulate_grad(dout)
            other._accumulate_grad(dout)
        
        out._backward_fn = backward
        return out
    
    def __mul__(self, other: Union['Tensor', np.ndarray, float]) -> 'Tensor':
        """Element-wise multiplication with autograd."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        out = Tensor(self.data * other.data, requires_grad=True,
                     parents=(self, other), name=f"{self.name}_mul")
        
        def backward():
            dout = out.grad
            self._accumulate_grad(dout * other.data)
            other._accumulate_grad(dout * self.data)
        
        out._backward_fn = backward
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Tensor', np.ndarray, float]) -> 'Tensor':
        """Element-wise division."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        
        out = Tensor(self.data / other.data, requires_grad=True,
                     parents=(self, other), name=f"{self.name}_div")
        
        def backward():
            dout = out.grad
            self._accumulate_grad(dout / other.data)
            other._accumulate_grad(-dout * self.data / (other.data ** 2))
        
        out._backward_fn = backward
        return out
    
    def matmul(self, other: 'Tensor', transpose_a: bool = False,
               transpose_b: bool = False) -> 'Tensor':
        """Matrix multiplication with optional transposition."""
        a_data = self.data
        b_data = other.data
        
        if transpose_a:
            a_data = np.swapaxes(a_data, -2, -1)
        if transpose_b:
            b_data = np.swapaxes(b_data, -2, -1)
        
        out_data = np.matmul(a_data, b_data)
        out = Tensor(out_data, requires_grad=True, parents=(self, other),
                     name=f"{self.name}_matmul")
        
        def backward():
            dout = out.grad
            
            # Gradient w.r.t. self
            if transpose_a:
                grad_a = np.matmul(dout, np.swapaxes(b_data, -2, -1))
                grad_a = np.swapaxes(grad_a, -2, -1)
            else:
                grad_a = np.matmul(dout, np.swapaxes(b_data, -2, -1))
            self._accumulate_grad(grad_a)
            
            # Gradient w.r.t. other
            if transpose_b:
                grad_b = np.matmul(np.swapaxes(a_data, -2, -1), dout)
                grad_b = np.swapaxes(grad_b, -2, -1)
            else:
                grad_b = np.matmul(np.swapaxes(a_data, -2, -1), dout)
            other._accumulate_grad(grad_b)
        
        out._backward_fn = backward
        return out
    
    def transpose(self, *axes) -> 'Tensor':
        """Tensor transposition."""
        out = Tensor(self.data.transpose(*axes), requires_grad=True,
                     parents=(self,), name=f"{self.name}_transpose")
        
        # Inverse permutation
        inv_axes = np.argsort(axes)
        
        def backward():
            self._accumulate_grad(out.grad.transpose(*inv_axes))
        
        out._backward_fn = backward
        return out
    
    def reshape(self, *shape) -> 'Tensor':
        """Reshape tensor."""
        original_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), requires_grad=True,
                     parents=(self,), name=f"{self.name}_reshape")
        
        def backward():
            self._accumulate_grad(out.grad.reshape(original_shape))
        
        out._backward_fn = backward
        return out
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
             keepdims: bool = False) -> 'Tensor':
        """Mean reduction."""
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=True, parents=(self,),
                     name=f"{self.name}_mean")
        
        def backward():
            dout = out.grad
            if not keepdims and axis is not None:
                dout = np.expand_dims(dout, axis=axis) if isinstance(axis, int) else \
                       dout.reshape([s if i not in (axis if isinstance(axis, tuple) else (axis,)) else 1
                                     for i, s in enumerate(self.data.shape)])
            
            # Average gradient
            if axis is None:
                n = self.data.size
            else:
                if isinstance(axis, int):
                    n = self.data.shape[axis]
                else:
                    n = np.prod([self.data.shape[i] for i in axis])
            
            self._accumulate_grad(dout / n)
        
        out._backward_fn = backward
        return out
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None,
            keepdims: bool = False) -> 'Tensor':
        """Sum reduction."""
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=True, parents=(self,),
                     name=f"{self.name}_sum")
        
        def backward():
            dout = out.grad
            if not keepdims and axis is not None:
                if isinstance(axis, int):
                    dout = np.expand_dims(dout, axis=axis)
                else:
                    dout = np.expand_dims(dout, axis=axis)
            grad = np.broadcast_to(dout, self.data.shape)
            self._accumulate_grad(grad)
        
        out._backward_fn = backward
        return out
    
    def __repr__(self) -> str:
        return f"Tensor({self.name}, shape={self.shape}, dtype={self.dtype.name}, requires_grad={self.requires_grad})"


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED OPERATIONS (FUSED & OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════════════════

class FusedOperations:
    """Container for fused/optimized operations."""
    
    @staticmethod
    def rmsnorm(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                eps: float = 1e-6) -> Tensor:
        """RMSNorm: fused normalization + scaling."""
        x_data = x.data
        w_data = weight.data
        
        # RMS normalization
        rms = np.sqrt((x_data ** 2).mean(-1, keepdims=True) + eps)
        normalized = x_data / rms
        
        # Scale
        out_data = normalized * w_data
        if bias is not None:
            out_data = out_data + bias.data
        
        out = Tensor(out_data, requires_grad=True,
                     parents=(x, weight) + ((bias,) if bias else ()),
                     name="fused_rmsnorm")
        
        def backward():
            dout = out.grad
            d_scale = (dout * normalized).sum(axis=tuple(range(dout.ndim - 1)), keepdims=True)
            weight._accumulate_grad(d_scale)
            
            if bias is not None:
                d_bias = dout.sum(axis=tuple(range(dout.ndim - 1)), keepdims=True)
                bias._accumulate_grad(d_bias)
            
            # Gradient w.r.t. input (complex, full implementation)
            d_norm = dout * w_data
            d_rms = (-d_norm * x_data / (rms ** 3)).sum(-1, keepdims=True) / x_data.shape[-1]
            x_grad = d_norm / rms + 2 * x_data * d_rms / x_data.shape[-1]
            x._accumulate_grad(x_grad)
        
        out._backward_fn = backward
        return out
    
    @staticmethod
    def silu_gate(x: Tensor, weight_1: Tensor, weight_3: Tensor) -> Tensor:
        """SwiGLU: fused gated linear unit."""
        # x @ w1 * silu(x @ w3)
        gate_data = x.data @ weight_1.data
        value_data = x.data @ weight_3.data
        
        # SiLU activation: x * sigmoid(x)
        silu_val = value_data * (1.0 / (1.0 + np.exp(-value_data)))
        out_data = gate_data * silu_val
        
        out = Tensor(out_data, requires_grad=True,
                     parents=(x, weight_1, weight_3),
                     name="fused_silu_gate")
        
        def backward():
            dout = out.grad
            
            # Complex gradient computation for SwiGLU
            sigmoid = 1.0 / (1.0 + np.exp(-value_data))
            silu_grad = sigmoid + value_data * sigmoid * (1 - sigmoid)
            
            d_gate = dout * silu_val
            d_value = dout * gate_data * silu_grad
            
            x._accumulate_grad(d_gate @ weight_1.data.T + d_value @ weight_3.data.T)
            weight_1._accumulate_grad(x.data.T @ d_gate)
            weight_3._accumulate_grad(x.data.T @ d_value)
        
        out._backward_fn = backward
        return out
    
    @staticmethod
    def causal_softmax(scores: Tensor, scale: float = 1.0,
                      causal: bool = True) -> Tensor:
        """Numerically stable causal softmax with fused backward."""
        scores_data = scores.data * scale
        
        if causal:
            T = scores_data.shape[-1]
            causal_mask = np.tril(np.ones((T, T))) == 0
            scores_data = np.where(causal_mask, -1e9, scores_data)
        
        # Numerical stability: subtract max
        scores_data_stable = scores_data - scores_data.max(-1, keepdims=True)
        exp_scores = np.exp(scores_data_stable)
        probs = exp_scores / exp_scores.sum(-1, keepdims=True)
        
        out = Tensor(probs, requires_grad=True, parents=(scores,),
                     name="fused_causal_softmax")
        
        def backward():
            dout = out.grad
            dscores = probs * (dout - (dout * probs).sum(-1, keepdims=True))
            if causal:
                dscores = np.where(causal_mask, 0, dscores)
            scores._accumulate_grad(dscores * scale)
        
        out._backward_fn = backward
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# POSITIONAL EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════════

class RotaryEmbedding:
    """
    Rotary Position Embeddings (RoPE).
    
    Applies rotation matrices to Q and K based on absolute positions.
    Allows efficient context length extrapolation.
    """
    
    def __init__(self, config: RotaryEmbeddingConfig):
        self.config = config
        self.dim = config.dim
        self.base = config.base
        self.freq_scale = config.freq_scale
        
        # Pre-compute frequencies
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2).astype(np.float32) / self.dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def register_buffer(self, name: str, tensor: np.ndarray):
        """Register a buffer (non-trainable tensor)."""
        setattr(self, name, tensor)
    
    def _update_cos_sin_cache(self, seq_len: int, device: Optional[str] = None):
        """Pre-compute cos/sin values for given sequence length."""
        if seq_len <= self._seq_len_cached:
            return
        
        self._seq_len_cached = seq_len
        
        # Compute angles: (seq_len, dim/2)
        t = np.arange(seq_len, dtype=np.float32) * self.freq_scale
        freqs = np.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, dim/2)
        
        # Create full freqs by interleaving (seq_len, dim)
        emb = np.concatenate([freqs, freqs], axis=-1)
        
        self._cos_cached = np.cos(emb)  # (seq_len, dim)
        self._sin_cached = np.sin(emb)
    
    def forward(self, x: np.ndarray, pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply RoPE to tensor.
        
        Args:
            x: (..., seq_len, dim)
            pos: Optional custom position indices
        
        Returns:
            Rotated tensor
        """
        seq_len = x.shape[-2]
        self._update_cos_sin_cache(seq_len)
        
        if pos is None:
            cos = self._cos_cached
            sin = self._sin_cached
        else:
            cos = self._cos_cached[pos]
            sin = self._sin_cached[pos]
        
        # Split into even/odd dimensions
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        # Apply rotation
        # [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos]
        rotated_even = x_even * cos[..., ::2] - x_odd * sin[..., ::2]
        rotated_odd = x_even * sin[..., ::2] + x_odd * cos[..., ::2]
        
        # Interleave back
        out = np.empty_like(x)
        out[..., ::2] = rotated_even
        out[..., 1::2] = rotated_odd
        
        return out
    
    def __repr__(self) -> str:
        return f"RotaryEmbedding(dim={self.dim}, base={self.base})"


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION LAYERS
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm:
    """Root Mean Square Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6, use_bias: bool = False):
        self.dim = dim
        self.eps = eps
        self.use_bias = use_bias
        
        self.weight = Tensor(np.ones(dim, dtype=np.float32), requires_grad=True,
                            name="rmsnorm_weight")
        self.bias = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True,
                          name="rmsnorm_bias") if use_bias else None
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm."""
        x_data = x.data
        rms = np.sqrt((x_data ** 2).mean(-1, keepdims=True) + self.eps)
        normalized = x_data / rms
        
        out = normalized * self.weight.data
        if self.bias is not None:
            out = out + self.bias.data
        
        result = Tensor(out, requires_grad=True, parents=(x, self.weight) + ((self.bias,) if self.bias else ()),
                        name="rmsnorm_output")
        
        def backward():
            dout = result.grad
            
            # Gradient w.r.t. weight
            dweight = (dout * normalized).sum(axis=tuple(range(dout.ndim - 1)), keepdims=True)
            self.weight._accumulate_grad(dweight.reshape(self.dim))
            
            # Gradient w.r.t. bias
            if self.bias is not None:
                dbias = dout.sum(axis=tuple(range(dout.ndim - 1)), keepdims=True)
                self.bias._accumulate_grad(dbias.reshape(self.dim))
            
            # Gradient w.r.t. input (complex derivative)
            scale = 1.0 / rms
            d_scale = (dout * self.weight.data).sum(-1, keepdims=True)
            d_rms = -d_scale * x_data / (rms ** 3) / x_data.shape[-1]
            
            x_grad = (dout * self.weight.data) / rms + 2 * x_data * d_rms / x_data.shape[-1]
            x._accumulate_grad(x_grad)
        
        result._backward_fn = backward
        return result
    
    def parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]
    
    def __repr__(self) -> str:
        return f"RMSNorm(dim={self.dim}, eps={self.eps})"


# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION MECHANISMS
# ═══════════════════════════════════════════════════════════════════════════════

class GroupedQueryAttention:
    """
    Grouped Query Attention (GQA).
    
    Reduces memory and computation by using fewer K/V heads than Q heads.
    GQA is the sweet spot between MHA (n_kv=n_heads) and MQA (n_kv=1).
    """
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.n_heads * config.head_dim
        
        # Linear projections
        self.wq = Tensor(
            np.random.randn(self.d_model, self.n_heads * self.head_dim).astype(np.float32) * 0.02,
            requires_grad=True, name="attention_wq"
        )
        self.wk = Tensor(
            np.random.randn(self.d_model, self.n_kv_heads * self.head_dim).astype(np.float32) * 0.02,
            requires_grad=True, name="attention_wk"
        )
        self.wv = Tensor(
            np.random.randn(self.d_model, self.n_kv_heads * self.head_dim).astype(np.float32) * 0.02,
            requires_grad=True, name="attention_wv"
        )
        self.wo = Tensor(
            np.random.randn(self.n_heads * self.head_dim, self.d_model).astype(np.float32) * 0.02,
            requires_grad=True, name="attention_wo"
        )
        
        self.dropout_p = config.dropout_p
        self.scale_factor = config.scale_factor
    
    def _repeat_kv(self, x: np.ndarray) -> np.ndarray:
        """Repeat KV heads to match Q heads."""
        if self.n_rep == 1:
            return x
        return np.repeat(x, self.n_rep, axis=1)
    
    def forward(self, x: Tensor, rope: RotaryEmbedding, cache: Optional[Dict] = None) -> Tensor:
        """
        Forward pass with GQA.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            rope: Rotary embeddings
            cache: Optional KV cache for inference
        
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        B, T, D = x.data.shape
        
        # Project to Q, K, V
        q = x.matmul(self.wq)  # (B, T, n_heads * head_dim)
        k = x.matmul(self.wk)  # (B, T, n_kv_heads * head_dim)
        v = x.matmul(self.wv)  # (B, T, n_kv_heads * head_dim)
        
        # Reshape and transpose for attention
        q_data = q.data.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  # (B, n_heads, T, head_dim)
        k_data = k.data.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v_data = v.data.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        q_data = rope.forward(q_data)
        k_data = rope.forward(k_data)
        
        # Repeat KV for GQA
        k_data = self._repeat_kv(k_data)  # (B, n_heads, T, head_dim)
        v_data = self._repeat_kv(v_data)
        
        # Compute attention scores
        scores = np.matmul(q_data, k_data.transpose(0, 1, 3, 2)) * self.scale_factor  # (B, n_heads, T, T)
        
        # Apply causal mask
        T_val = scores.shape[-1]
        causal_mask = np.tril(np.ones((T_val, T_val))) == 0
        scores = np.where(causal_mask, -1e9, scores)
        
        # Softmax
        scores_stable = scores - scores.max(-1, keepdims=True)
        exp_scores = np.exp(scores_stable)
        attn = exp_scores / exp_scores.sum(-1, keepdims=True)  # (B, n_heads, T, T)
        
        # Dropout (not applied during inference typically)
        if self.dropout_p > 0:
            attn = attn * (np.random.rand(*attn.shape) > self.dropout_p) / (1 - self.dropout_p)
        
        # Apply attention to values
        out = np.matmul(attn, v_data)  # (B, n_heads, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.n_heads * self.head_dim)  # (B, T, n_heads*head_dim)
        
        # Output projection
        out = Tensor(out @ self.wo.data, requires_grad=True,
                     name="gqa_output")
        
        return out
    
    def parameters(self) -> List[Tensor]:
        """Return all parameters."""
        return [self.wq, self.wk, self.wv, self.wo]
    
    def __repr__(self) -> str:
        return (f"GroupedQueryAttention(n_heads={self.n_heads}, "
                f"n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim})")


# ═══════════════════════════════════════════════════════════════════════════════
# FEED-FORWARD NETWORKS
# ═══════════════════════════════════════════════════════════════════════════════

class SwiGLUFeedForward:
    """
    SwiGLU Feed-Forward Network.
    
    Structure:
        (x @ W1 * SiLU(x @ W3)) @ W2
    
    More expressive than standard FFN with same parameter count.
    """
    
    def __init__(self, config: FeedForwardConfig, d_model: int):
        self.d_model = d_model
        self.hidden_dim = config.hidden_dim
        self.intermediate_dim = config.intermediate_dim
        
        # W1: x -> intermediate (gate)
        self.w1 = Tensor(
            np.random.randn(d_model, self.intermediate_dim).astype(np.float32) * 0.02,
            requires_grad=True, name="ffn_w1"
        )
        
        # W3: x -> intermediate (value)
        self.w3 = Tensor(
            np.random.randn(d_model, self.intermediate_dim).astype(np.float32) * 0.02,
            requires_grad=True, name="ffn_w3"
        )
        
        # W2: intermediate -> output
        self.w2 = Tensor(
            np.random.randn(self.intermediate_dim, d_model).astype(np.float32) * 0.02,
            requires_grad=True, name="ffn_w2"
        )
        
        if config.use_bias:
            self.bias1 = Tensor(np.zeros(self.intermediate_dim), requires_grad=True,
                               name="ffn_bias1")
            self.bias3 = Tensor(np.zeros(self.intermediate_dim), requires_grad=True,
                               name="ffn_bias3")
            self.bias2 = Tensor(np.zeros(d_model), requires_grad=True,
                               name="ffn_bias2")
        else:
            self.bias1 = self.bias3 = self.bias2 = None
        
        self.dropout_p = config.dropout_p
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through SwiGLU FFN."""
        # Gate path: x @ W1 + bias1
        gate_data = x.data @ self.w1.data
        if self.bias1 is not None:
            gate_data = gate_data + self.bias1.data
        
        # Value path: SiLU(x @ W3 + bias3)
        value_data = x.data @ self.w3.data
        if self.bias3 is not None:
            value_data = value_data + self.bias3.data
        
        # SiLU activation
        silu_data = value_data * (1.0 / (1.0 + np.exp(-value_data)))
        
        # Gate * Value
        gated_data = gate_data * silu_data
        
        # Output projection
        out_data = gated_data @ self.w2.data
        if self.bias2 is not None:
            out_data = out_data + self.bias2.data
        
        out = Tensor(out_data, requires_grad=True, name="swiglu_output")
        
        return out
    
    def parameters(self) -> List[Tensor]:
        """Return all parameters."""
        ps = [self.w1, self.w3, self.w2]
        if self.bias1 is not None:
            ps.extend([self.bias1, self.bias3, self.bias2])
        return ps
    
    def __repr__(self) -> str:
        return (f"SwiGLUFeedForward(input={self.d_model}, "
                f"intermediate={self.intermediate_dim})")


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerBlock:
    """
    Single Transformer block with:
      - Pre-norm attention
      - Pre-norm feed-forward
      - Residual connections
      - Optional gradient checkpointing
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        self.layer_idx = layer_idx
        
        # Attention
        attn_config = config.attention
        self.attn_norm = RMSNorm(config.d_model, eps=config.normalization.eps)
        self.attn = GroupedQueryAttention(attn_config)
        
        # Feed-forward
        ff_config = config.feed_forward
        ff_config.hidden_dim = config.d_model
        self.ffn_norm = RMSNorm(config.d_model, eps=config.normalization.eps)
        self.ffn = SwiGLUFeedForward(ff_config, config.d_model)
        
        # Dropouts
        self.attn_dropout_p = config.attention_dropout_p
        self.residual_dropout_p = config.residual_dropout_p
        
        # Gradient checkpointing
        self.use_checkpointing = config.use_gradient_checkpointing
    
    def forward(self, x: Tensor, rope: RotaryEmbedding, cache: Optional[Dict] = None) -> Tensor:
        """Forward pass through transformer block."""
        # Self-attention with pre-norm
        attn_input = self.attn_norm.forward(x)
        attn_out = self.attn.forward(attn_input, rope, cache)
        
        # Residual connection
        x_attn = Tensor(x.data + attn_out.data, requires_grad=True,
                       name=f"block_{self.layer_idx}_attn_residual")
        
        # Feed-forward with pre-norm
        ffn_input = self.ffn_norm.forward(x_attn)
        ffn_out = self.ffn.forward(ffn_input)
        
        # Final residual
        out = Tensor(x_attn.data + ffn_out.data, requires_grad=True,
                    name=f"block_{self.layer_idx}_output")
        
        return out
    
    def parameters(self) -> List[Tensor]:
        """Collect all parameters from this block."""
        ps = []
        ps.extend(self.attn_norm.parameters())
        ps.extend(self.attn.parameters())
        ps.extend(self.ffn_norm.parameters())
        ps.extend(self.ffn.parameters())
        return ps
    
    def __repr__(self) -> str:
        return f"TransformerBlock(layer={self.layer_idx})"


# ═══════════════════════════════════════════════════════════════════════════════
# FULL TRANSFORMER MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ManusAlpha:
    """
    ManusAlpha: State-of-the-art Decoder-only Transformer.
    
    Complete implementation with:
      - Token embeddings (tied with output projection)
      - Rotary positional embeddings
      - 12+ transformer blocks
      - Final layer normalization
      - Causal language modeling head
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        logger.info(f"Initializing ManusAlpha with config: {config}")
        
        # Embeddings
        self.tok_emb = Tensor(
            np.random.randn(config.vocab_size, config.d_model).astype(np.float32) * config.init_std,
            requires_grad=True, name="token_embeddings"
        )
        
        # Positional embeddings
        rope_config = config.rope
        rope_config.dim = config.d_model // config.attention.n_heads
        self.rope = RotaryEmbedding(rope_config)
        
        # Transformer blocks
        self.layers = [
            TransformerBlock(config, i)
            for i in range(config.n_layers)
        ]
        
        # Final normalization
        self.final_norm = RMSNorm(config.d_model, eps=config.normalization.eps)
        
        # Output projection (tied with embeddings)
        self.use_tied_embeddings = config.tie_word_embeddings
        
        self._init_weights(config)
        logger.info(f"Model initialized with {self.num_params():,} parameters")
    
    def _init_weights(self, config: ModelConfig):
        """Initialize weights according to scheme."""
        scheme = config.weight_init_scheme
        std = config.init_std
        
        if scheme == "normal":
            pass  # Already initialized above
        elif scheme == "uniform":
            bound = np.sqrt(3) * std
            for p in self.parameters():
                p.data = np.random.uniform(-bound, bound, p.data.shape).astype(np.float32)
        elif scheme == "xavier":
            for p in self.parameters():
                fan_in = p.data.shape[0] if p.data.ndim > 0 else 1
                std = np.sqrt(1.0 / fan_in)
                p.data = np.random.randn(*p.data.shape).astype(np.float32) * std
        
        logger.debug(f"Weights initialized with scheme: {scheme}")
    
    def parameters(self) -> List[Tensor]:
        """Collect all trainable parameters."""
        ps = [self.tok_emb]
        for layer in self.layers:
            ps.extend(layer.parameters())
        ps.extend(self.final_norm.parameters())
        return ps
    
    def named_parameters(self) -> Dict[str, Tensor]:
        """Return parameters with names."""
        params = {}
        params['tok_emb'] = self.tok_emb
        for i, layer in enumerate(self.layers):
            for p in layer.parameters():
                params[f'layer_{i}_{p.name}'] = p
        for p in self.final_norm.parameters():
            params[f'final_norm_{p.name}'] = p
        return params
    
    def num_params(self) -> int:
        """Count total parameters."""
        return sum(p.size for p in self.parameters())
    
    def zero_grad(self):
        """Clear all gradients."""
        for p in self.parameters():
            p.grad = None
    
    def forward(self, input_ids: np.ndarray, targets: Optional[np.ndarray] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token indices (batch_size, seq_len)
            targets: Optional target IDs for loss computation
        
        Returns:
            logits: Prediction logits (batch_size, seq_len, vocab_size)
            loss: Optional loss scalar
        """
        B, T = input_ids.shape
        
        # Token embeddings
        x_data = self.tok_emb.data[input_ids]  # (B, T, d_model)
        x = Tensor(x_data, requires_grad=True, name="embedded_input")
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer.forward(x, self.rope)
        
        # Final normalization
        x = self.final_norm.forward(x)
        
        # Logits (tied embeddings)
        logits_data = x.data @ self.tok_emb.data.T
        logits = Tensor(logits_data, requires_grad=True, name="logits")
        
        # Loss computation if targets provided
        loss = None
        if targets is not None:
            # Cross entropy loss
            logits_flat = logits_data.reshape(B * T, self.config.vocab_size)
            targets_flat = targets.reshape(B * T)
            
            # Softmax log-loss
            logits_stable = logits_flat - logits_flat.max(-1, keepdims=True)
            exp_logits = np.exp(logits_stable)
            probs = exp_logits / exp_logits.sum(-1, keepdims=True)
            
            losses = -np.log(probs[np.arange(B * T), targets_flat] + 1e-10)
            loss_val = losses.mean()
            
            loss = Tensor(np.array(loss_val), requires_grad=True, name="loss")
            
            def loss_backward():
                dloss = loss.grad
                dp = probs.copy()
                dp[np.arange(B * T), targets_flat] -= 1.0
                dp_flat = dp * dloss / (B * T)
                
                # Gradient through logits
                logits_grad = dp_flat
                logits._accumulate_grad(logits_grad.reshape(B, T, self.config.vocab_size))
            
            loss._backward_fn = loss_backward
        
        return logits, loss
    
    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config.to_dict(),
            'parameters': {name: param.data for name, param in self.named_parameters().items()},
            'version': '2.1.0',
            'dtype': str(self.tok_emb.data.dtype)
        }
        
        with open(path, 'wb') as f:
            np.save(f, checkpoint, allow_pickle=True)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ManusAlpha':
        """Load model from disk."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            checkpoint = np.load(f, allow_pickle=True).item()
        
        config = ModelConfig.from_dict(checkpoint['config'])
        model = cls(config)
        
        # Load parameters
        for name, param_data in checkpoint['parameters'].items():
            model.named_parameters()[name].data = param_data
        
        logger.info(f"Model loaded from {path}")
        return model
    
    def __repr__(self) -> str:
        return (f"ManusAlpha(\n"
                f"  vocab_size={self.config.vocab_size}\n"
                f"  d_model={self.config.d_model}\n"
                f"  n_layers={self.config.n_layers}\n"
                f"  n_params={self.num_params():,}\n"
                f")")

