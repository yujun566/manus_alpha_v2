"""
═══════════════════════════════════════════════════════════════════════════════
MANUS-ALPHA :: Tokenizer Implementation
Byte-Pair Encoding (BPE) with Advanced Vocabulary Management
═══════════════════════════════════════════════════════════════════════════════

Professional-grade BPE tokenizer with:
  • Byte-level encoding for UTF-8 text
  • Efficient vocabulary building with frequency-based merging
  • Subword caching for fast encoding
  • Regex-based pre-tokenization
  • Special token management
  • Vocabulary statistics and analysis
  • Token ID compression and decompression
  • Serialization with backward compatibility

Dependencies: None (pure Python)
Author: Manus Research Team
"""

from __future__ import annotations
import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from dataclasses import dataclass, field, asdict
import pickle

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenizerConfig:
    """Configuration for BPE tokenizer."""
    vocab_size: int = 32768
    min_frequency: int = 2
    max_merges: int = 40000
    lower_case: bool = False
    add_prefix_space: bool = False
    
    # Regex patterns for pre-tokenization
    pattern: str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Special tokens
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    mask_token: str = "<mask>"
    
    # Unicode normalization
    use_unicode_norm: bool = False
    
    # Cache settings
    use_cache: bool = True
    cache_size: int = 100000
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenizerConfig:
        return cls(**data)


class VocabularyIndex:
    """Efficient vocabulary storage with forward/reverse lookup."""
    
    def __init__(self, vocab_size: int = 32768):
        self.vocab_size = vocab_size
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._next_id = 0
        self._frozen = False
    
    def add(self, token: str) -> int:
        """Add token to vocabulary, return its ID."""
        if self._frozen and token not in self._token_to_id:
            raise ValueError(f"Vocabulary is frozen. Cannot add token '{token}'")
        
        if token not in self._token_to_id:
            if self._next_id >= self.vocab_size:
                raise OverflowError(f"Vocabulary size exceeded: {self.vocab_size}")
            
            token_id = self._next_id
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
            self._next_id += 1
            
            return token_id
        
        return self._token_to_id[token]
    
    def get_id(self, token: str) -> Optional[int]:
        """Get token ID, return None if not found."""
        return self._token_to_id.get(token)
    
    def get_token(self, token_id: int) -> Optional[str]:
        """Get token from ID."""
        return self._id_to_token.get(token_id)
    
    def freeze(self):
        """Freeze vocabulary (no more additions allowed)."""
        self._frozen = True
        logger.info(f"Vocabulary frozen at size {self._next_id}")
    
    def unfreeze(self):
        """Unfreeze vocabulary."""
        self._frozen = False
    
    def __len__(self) -> int:
        return self._next_id
    
    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
    
    def items(self):
        """Iterate over (token, id) pairs."""
        return self._token_to_id.items()
    
    def save(self, path: str):
        """Save vocabulary to file."""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self._token_to_id,
            'id_to_token': self._id_to_token,
            'next_id': self._next_id,
            'frozen': self._frozen
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Vocabulary saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> VocabularyIndex:
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        vocab = cls(data['vocab_size'])
        vocab._token_to_id = {k: int(v) for k, v in data['token_to_id'].items()}
        vocab._id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        vocab._next_id = data['next_id']
        vocab._frozen = data['frozen']
        
        return vocab


@dataclass
class MergeRule:
    """Single BPE merge rule."""
    rank: int
    left: int  # Left token ID
    right: int  # Right token ID
    merged: int  # Resulting token ID
    frequency: int  # How many times this pair appeared
    
    def __repr__(self) -> str:
        return f"MergeRule(rank={self.rank}, freq={self.frequency})"


# ═══════════════════════════════════════════════════════════════════════════════
# BYTE-LEVEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

class ByteEncoder:
    """Convert text to/from byte sequences."""
    
    @staticmethod
    def encode_text_to_bytes(text: str) -> List[int]:
        """Encode UTF-8 text to list of byte values."""
        return list(text.encode('utf-8'))
    
    @staticmethod
    def decode_bytes_to_text(bytes_seq: List[int]) -> str:
        """Decode byte sequence to text."""
        return bytes(bytes_seq).decode('utf-8', errors='replace')
    
    @staticmethod
    def bytes_to_unicode_tokens(byte_list: List[int]) -> List[str]:
        """
        Convert bytes to string tokens, handling special cases.
        This allows BPE to work with Unicode codepoints naturally.
        """
        # Create mapping: 0-255 as printable if possible, otherwise use unicode
        bs = list(range(256))
        cs = bs[:]
        n = 0
        
        # Skip whitespace and special chars, map them to unused unicode space
        for b in bs:
            if b not in (9, 10, 13):  # Keep tab, newline, carriage return
                if 32 <= b <= 126 or b >= 160:  # Skip problematic ASCII
                    cs.append(b)
                else:
                    cs.append(256 + n)
                    n += 1
        
        cs = [chr(n) for n in cs]
        return [cs[b] for b in byte_list]


# ═══════════════════════════════════════════════════════════════════════════════
# BPE TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

class BPETokenizer:
    """
    Byte-Pair Encoding Tokenizer.
    
    Training process:
    1. Encode text to bytes
    2. Find most frequent adjacent byte pair
    3. Create new token for this pair, replace all occurrences
    4. Repeat until vocabulary size reached
    
    This results in a compact, hierarchical tokenization that handles
    any UTF-8 text, including out-of-vocabulary characters gracefully.
    """
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.vocab = VocabularyIndex(self.config.vocab_size)
        self.merges: List[Tuple[int, int]] = []  # List of merge operations
        self.merge_cache: Dict[Tuple[int, int], int] = {}  # Merge -> result token
        self.encoder_cache: Dict[str, List[int]] = {}  # String -> token IDs
        
        # Special tokens
        self._add_special_tokens()
        
        # Statistics
        self.merge_stats: List[Tuple[int, int, int]] = []  # (pair, freq, token_id)
        self.training_steps = 0
        
        logger.info(f"Initialized BPETokenizer: vocab_size={self.config.vocab_size}, "
                   f"max_merges={self.config.max_merges}")
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [
            self.config.pad_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.unk_token,
            self.config.mask_token,
        ]
        
        for token in special_tokens:
            self.vocab.add(token)
            logger.debug(f"Added special token: {token}")
        
        self.special_token_ids = {token: self.vocab.get_id(token) for token in special_tokens}
    
    def train(self, texts: List[str], verbose: bool = True) -> Dict[str, Any]:
        """
        Train BPE tokenizer on corpus.
        
        Args:
            texts: List of training texts
            verbose: Print progress
        
        Returns:
            Training statistics
        """
        logger.info(f"Starting BPE training on {len(texts)} texts")
        
        # Count byte frequencies
        byte_freqs: Counter = Counter()
        word_freqs: Dict[Tuple[int, ...], int] = defaultdict(int)
        
        # Encode all words and count frequencies
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                byte_seq = ByteEncoder.encode_text_to_bytes(word)
                if byte_seq:
                    word_freqs[tuple(byte_seq)] += 1
        
        logger.info(f"Created {len(word_freqs)} unique subwords from training corpus")
        
        # Initialize vocabulary with individual bytes
        for b in range(256):
            self.vocab.add(bytes([b]).decode('latin-1'))
        
        # Merge loop
        num_merges = min(self.config.max_merges, self.config.vocab_size - self.vocab._next_id)
        
        for merge_step in range(num_merges):
            if merge_step % max(1, num_merges // 10) == 0 and verbose:
                logger.info(f"Merge step {merge_step + 1}/{num_merges} "
                           f"(vocab_size={len(self.vocab)})")
            
            # Find most frequent adjacent pair
            pair_freqs: Counter = Counter()
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                logger.warning("No more pairs to merge")
                break
            
            best_pair = pair_freqs.most_common(1)[0]
            pair, pair_freq = best_pair
            
            # Create new token for this pair
            left_token = bytes([pair[0]]).decode('latin-1') if isinstance(pair[0], int) else self.vocab.get_token(pair[0])
            right_token = bytes([pair[1]]).decode('latin-1') if isinstance(pair[1], int) else self.vocab.get_token(pair[1])
            new_token = left_token + right_token
            new_token_id = self.vocab.add(new_token)
            
            # Record merge rule
            self.merges.append(pair)
            self.merge_cache[pair] = new_token_id
            self.merge_stats.append((pair, pair_freq, new_token_id))
            
            # Update word frequencies with merged token
            new_word_freqs: Dict[Tuple[int, ...], int] = defaultdict(int)
            for word_tuple, freq in word_freqs.items():
                new_word = list(word_tuple)
                
                # Replace all occurrences of pair in word
                i = 0
                while i < len(new_word) - 1:
                    if (new_word[i], new_word[i + 1]) == pair:
                        new_word = new_word[:i] + [new_token_id] + new_word[i + 2:]
                    else:
                        i += 1
                
                new_word_freqs[tuple(new_word)] += freq
            
            word_freqs = new_word_freqs
            self.training_steps += 1
        
        self.vocab.freeze()
        
        stats = {
            'vocab_size': len(self.vocab),
            'num_merges': len(self.merges),
            'num_training_texts': len(texts),
            'training_steps': self.training_steps
        }
        
        logger.info(f"Training complete: {stats}")
        return stats
    
    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text using regex pattern.
        
        Splits text into words before BPE encoding.
        """
        # Simple splitting: split on whitespace and punctuation
        # In production, use more sophisticated regex patterns
        if self.config.add_prefix_space and not text.startswith(' '):
            text = ' ' + text
        
        # Basic split: keep alphanumeric chunks separate from punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text, re.UNICODE)
        return [w for w in words if w.strip()]
    
    def encode(self, text: str, add_bos: bool = False,
              add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs using trained BPE.
        
        Args:
            text: Input text
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token
        
        Returns:
            List of token IDs
        """
        # Check cache first
        cache_key = (text, add_bos, add_eos)
        if self.config.use_cache and cache_key in self.encoder_cache:
            return self.encoder_cache[cache_key]
        
        token_ids = []
        
        if add_bos:
            token_ids.append(self.special_token_ids['bos'])
        
        # Pre-tokenize
        words = self._pre_tokenize(text)
        
        for word in words:
            # Encode word to bytes, then apply BPE
            word_tokens = self._encode_word(word)
            token_ids.extend(word_tokens)
        
        if add_eos:
            token_ids.append(self.special_token_ids['eos'])
        
        # Cache result if enabled
        if self.config.use_cache and len(self.encoder_cache) < self.config.cache_size:
            self.encoder_cache[cache_key] = token_ids
        
        return token_ids
    
    def _encode_word(self, word: str) -> List[int]:
        """Encode single word using BPE."""
        # Convert to bytes
        byte_seq = ByteEncoder.encode_text_to_bytes(word)
        if not byte_seq:
            return [self.special_token_ids['unk']]
        
        # Convert bytes to token IDs (initially each byte is its own token)
        tokens = list(byte_seq)
        
        # Apply merge rules
        for merge_left, merge_right in self.merges:
            # Find and merge all occurrences
            i = 0
            new_tokens = []
            
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == merge_left and tokens[i + 1] == merge_right:
                    # Apply merge
                    merged_token_id = self.merge_cache[(merge_left, merge_right)]
                    new_tokens.append(merged_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special: Skip special tokens in output
        
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            token = self.vocab.get_token(token_id)
            if token is None:
                token = self.config.unk_token
            
            if skip_special and token in self.special_token_ids:
                continue
            
            tokens.append(token)
        
        # Join and decode
        text = ''.join(tokens)
        
        try:
            # Try to decode as byte sequence
            byte_seq = [ord(c) if ord(c) < 256 else ord(c) - 256 for c in text]
            return bytes(byte_seq).decode('utf-8', errors='replace')
        except:
            return text
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.vocab)
    
    def get_token_id(self, token: str) -> Optional[int]:
        """Get token ID."""
        return self.vocab.get_id(token)
    
    def get_token(self, token_id: int) -> Optional[str]:
        """Get token from ID."""
        return self.vocab.get_token(token_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tokenizer statistics."""
        return {
            'vocab_size': self.get_vocab_size(),
            'num_merges': len(self.merges),
            'special_tokens': self.special_token_ids,
            'training_steps': self.training_steps,
            'cache_size': len(self.encoder_cache)
        }
    
    def save(self, path: str):
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config.to_dict(),
            'vocab': dict(self.vocab.items()),
            'merges': [(int(l), int(r)) for l, r in self.merges],
            'special_tokens': self.special_token_ids,
            'training_steps': self.training_steps
        }
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        """Load tokenizer from file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        config = TokenizerConfig.from_dict(checkpoint['config'])
        tokenizer = cls(config)
        
        # Restore vocabulary
        for token, token_id in checkpoint['vocab'].items():
            while len(tokenizer.vocab) <= token_id:
                tokenizer.vocab.add(f"_pad_{len(tokenizer.vocab)}")
            if token_id < len(tokenizer.vocab):
                tokenizer.vocab._token_to_id[token] = token_id
                tokenizer.vocab._id_to_token[token_id] = token
        
        # Restore merges
        tokenizer.merges = [(l, r) for l, r in checkpoint['merges']]
        tokenizer.merge_cache = {(l, r): i + 256 for i, (l, r) in enumerate(tokenizer.merges)}
        
        # Restore special tokens
        tokenizer.special_token_ids = checkpoint['special_tokens']
        
        tokenizer.training_steps = checkpoint['training_steps']
        tokenizer.vocab.freeze()
        
        logger.info(f"Tokenizer loaded from {path}")
        return tokenizer
    
    def __repr__(self) -> str:
        return (f"BPETokenizer(\n"
                f"  vocab_size={self.get_vocab_size()}\n"
                f"  num_merges={len(self.merges)}\n"
                f"  training_steps={self.training_steps}\n"
                f")")
