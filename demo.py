"""
═══════════════════════════════════════════════════════════════════════════════
MANUS-ALPHA :: Complete Integration Demo
═══════════════════════════════════════════════════════════════════════════════

End-to-end demonstration including:
  • Tokenizer training
  • Model initialization  
  • Training loop with checkpointing
  • Inference and generation
  • RLAIF self-evolution
  • Quality metrics and evaluation

This demo showcases the complete production pipeline.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Import core modules
try:
    from model import ManusAlpha, ModelConfig, AttentionConfig, FeedForwardConfig
    from tokenizer import BPETokenizer, TokenizerConfig
    from trainer import Trainer, TrainerConfig, DataLoader, DataCurator, AdamWConfig
    from inference import InferenceEngine, GenerationConfig, RLAIF, GenerationMetrics
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are in the same directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Training corpus
TRAINING_CORPUS = [
    "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n\n",
    "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n\n",
    "정리: 모든 자연수 n에 대해 1+2+...+n = n(n+1)/2이다.\n",
    "증명: 수학적 귀납법을 사용한다. 기저 단계: n=1일 때 성립한다.\n",
    "따라서 명제가 성립한다.\n\n",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    right = [x for x in arr[1:] if x >= pivot]\n    return quicksort(left) + [pivot] + quicksort(right)\n\n",
    "질문: 피보나치 수열이란 무엇인가?\n",
    "답변: 피보나치 수열은 F(n) = F(n-1) + F(n-2)로 정의되는 수열이다.\n\n",
    "class DataLoader:\n    def __init__(self, data, batch_size):\n        self.data = data\n        self.batch_size = batch_size\n\n",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\n",
] * 6  # Repeat for more data

def setup_tokenizer():
    """Initialize and train tokenizer."""
    logger.info("="*70)
    logger.info("STEP 1: TOKENIZER TRAINING")
    logger.info("="*70)
    
    config = TokenizerConfig(
        vocab_size=512,
        max_merges=256,
        lower_case=False
    )
    
    tokenizer = BPETokenizer(config)
    
    logger.info(f"Training BPE tokenizer on {len(TRAINING_CORPUS)} texts...")
    stats = tokenizer.train(TRAINING_CORPUS, verbose=True)
    
    logger.info(f"Tokenizer stats: {stats}")
    logger.info(f"Final vocab size: {tokenizer.get_vocab_size()}")
    
    # Test encoding/decoding
    test_text = "def hello(): return 42"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    logger.info(f"Test encode-decode:")
    logger.info(f"  Original: {repr(test_text)}")
    logger.info(f"  Tokens: {len(encoded)} tokens")
    logger.info(f"  Decoded: {repr(decoded[:50])}")
    
    return tokenizer


def setup_model():
    """Initialize model."""
    logger.info("="*70)
    logger.info("STEP 2: MODEL INITIALIZATION")
    logger.info("="*70)
    
    config = ModelConfig(
        vocab_size=512,
        d_model=96,
        n_layers=3,
        max_seq_length=128,
        attention=AttentionConfig(
            n_heads=6,
            n_kv_heads=2,
            head_dim=16
        ),
        feed_forward=FeedForwardConfig(
            hidden_dim=96,
            intermediate_dim=256
        )
    )
    
    model = ManusAlpha(config)
    logger.info(f"Model created:\n{model}")
    logger.info(f"Total parameters: {model.num_params():,}")
    
    return model


def train_model(model, tokenizer):
    """Train the model."""
    logger.info("="*70)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("="*70)
    
    # Tokenize corpus
    all_ids = []
    for text in TRAINING_CORPUS:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
    
    logger.info(f"Total tokens: {len(all_ids)}")
    
    # Create data loader
    loader = DataLoader(all_ids, seq_len=32, batch_size=4, seed=42)
    
    # Setup trainer
    trainer_config = TrainerConfig(
        optimizer_config=AdamWConfig(learning_rate=3e-3),
        warmup_steps=5,
        total_steps=50,
        log_interval=5,
        save_interval=100,
        save_dir="./checkpoints"
    )
    
    trainer = Trainer(model, trainer_config)
    
    # Training loop
    logger.info("Starting training...")
    for step in range(50):
        x, y = loader.get_batch()
        _, loss = model.forward(x, y)
        loss_val = float(loss.data)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        from trainer import GradientProcessor
        grad_norm = GradientProcessor.clip_grad_norm(model.parameters(), 1.0)
        
        # Optimizer step
        trainer.optimizer.step()
        model.zero_grad()
        
        # Update LR
        lr = trainer.scheduler.step()
        
        if step % 5 == 0 or step == 49:
            logger.info(f"Step {step:3d} | Loss: {loss_val:.4f} | GradNorm: {grad_norm:.4f} | LR: {lr:.2e}")
    
    logger.info("Training complete!")
    return trainer, loader


def test_generation(model, tokenizer):
    """Test text generation."""
    logger.info("="*70)
    logger.info("STEP 4: TEXT GENERATION")
    logger.info("="*70)
    
    engine = InferenceEngine(model, tokenizer)
    
    prompts = ["def ", "class ", "질문: "]
    
    for prompt in prompts:
        logger.info(f"\nPrompt: {repr(prompt)}")
        
        config = GenerationConfig(
            max_length=64,
            temperature=0.7,
            top_p=0.9,
            top_k=20
        )
        
        generated = engine.generate(prompt, config, seed=42)
        logger.info(f"Generated: {repr(generated[:80])}")


def test_rlaif(model, tokenizer):
    """Test RLAIF self-evolution."""
    logger.info("="*70)
    logger.info("STEP 5: RLAIF SELF-EVOLUTION")
    logger.info("="*70)
    
    engine = InferenceEngine(model, tokenizer)
    rlaif = RLAIF(engine, tokenizer)
    
    prompts = ["def ", "정리: "]
    
    for prompt in prompts:
        logger.info(f"\nPrompt: {repr(prompt)}")
        logger.info("Generating 4 candidates...")
        
        best_text, ranked = rlaif.best_of_n(prompt, n=4)
        
        logger.info(f"Candidate scores:")
        for reward, text in ranked:
            logger.info(f"  Reward {reward:.3f}: {repr(text[:40])}")
        
        logger.info(f"Best: {repr(best_text[:60])}")


def test_metrics(model, tokenizer):
    """Test evaluation metrics."""
    logger.info("="*70)
    logger.info("STEP 6: EVALUATION METRICS")
    logger.info("="*70)
    
    # Generate some text
    engine = InferenceEngine(model, tokenizer)
    config = GenerationConfig(max_length=64, temperature=0.5)
    text = engine.generate("def ", config)
    
    logger.info(f"Generated text: {repr(text[:80])}")
    
    # Compute metrics
    length = GenerationMetrics.length(text)
    diversity = GenerationMetrics.diversity(text)
    repetition = GenerationMetrics.repetition_ratio(text)
    
    logger.info(f"Metrics:")
    logger.info(f"  Length (tokens): {length}")
    logger.info(f"  Diversity (TTR): {diversity:.3f}")
    logger.info(f"  Repetition ratio: {repetition:.3f}")


def main():
    """Run complete pipeline."""
    logger.info("╔" + "="*68 + "╗")
    logger.info("║" + " MANUS-ALPHA: PRODUCTION-GRADE TRANSFORMER PIPELINE ".center(68) + "║")
    logger.info("╚" + "="*68 + "╝")
    
    try:
        # Step 1: Tokenizer
        tokenizer = setup_tokenizer()
        
        # Step 2: Model
        model = setup_model()
        
        # Step 3: Training
        trainer, loader = train_model(model, tokenizer)
        
        # Step 4: Generation
        test_generation(model, tokenizer)
        
        # Step 5: RLAIF
        test_rlaif(model, tokenizer)
        
        # Step 6: Metrics
        test_metrics(model, tokenizer)
        
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETE - ALL STEPS SUCCESSFUL")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
