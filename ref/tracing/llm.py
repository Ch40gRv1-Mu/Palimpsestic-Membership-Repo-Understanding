"""
Core LLM utilities: training, evaluation, generation, and distillation.

This module provides the building blocks for the experiments:
  - evaluate_model: compute per-token log-probabilities (the "signal" we test)
  - train_model: train a small LLM from scratch with controlled data ordering
  - distill_model: knowledge distillation (teacher→student)
  - generate: sample text from a trained model
"""

import torch                                                        # [HOUSEWORK] tensor operations
from transformers import LlamaForCausalLM                           # [HOUSEWORK] model class
from torch.utils.data import DataLoader                             # [HOUSEWORK] batching utility
# tqdm — full name "taqaddum" (تقدّم, Arabic for "progress") — provides a
# terminal progress bar for long-running loops. Pure housework: removing it
# changes nothing about the computation, only loses the progress display.
from tqdm import tqdm                                               # [HOUSEWORK] progress bar
import os                                                           # [HOUSEWORK] filesystem paths
import wandb                                                        # [HOUSEWORK] experiment logging
import random                                                       # [HOUSEWORK] RNG for shuffling
from vllm import LLM                                                # [HOUSEWORK] fast batched inference engine

def put(obj, device):                                               # [HOUSEWORK] recursive device-move helper
    """Recursively move tensors/dicts/lists to a device. Used for optimizer state."""
    if isinstance(obj, torch.Tensor):                               # [HOUSEWORK]
        return obj.to(device)                                       # [HOUSEWORK]
    elif isinstance(obj, dict):                                     # [HOUSEWORK]
        return {k: put(v, device) for k, v in obj.items()}          # [HOUSEWORK]
    elif isinstance(obj, list):                                     # [HOUSEWORK]
        return [put(v, device) for v in obj]                        # [HOUSEWORK]
    else:                                                           # [HOUSEWORK]
        return obj                                                  # [HOUSEWORK]

def evaluate_model(model, tokenizer, texts, prompts=None, metric=None, batch_size=1):
    """Compute per-token log-probabilities for each text under the model.

    This is THE core measurement function. For each text, it computes
    log P(token_t | token_1, ..., token_{t-1}) for every token position.

    These per-token logprobs are the raw signal that gets aggregated into
    the test statistic: sequences the model memorized better will have
    higher (less negative) logprobs.

    Args:
        model: the language model to evaluate
        texts: list of text strings
        prompts: optional prompt prefixes (metric only computed on completion part)
        metric: function(tokenized_text, logprobs, tokenized_prompt) → scalar
        batch_size: inference batch size

    Returns:
        predictions: list of per-token logprob tensors (one per text)
        metrics: list of scalar metrics (one per text), or None if no metric
    """
    if prompts is None:                                             # [HOUSEWORK] default empty prompts
        prompts = [""] * len(texts)                                 # [HOUSEWORK]
    else:                                                           # [HOUSEWORK]
        assert len(texts) == len(prompts), "texts and prompts must have the same length"  # [HOUSEWORK] input validation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # [HOUSEWORK] device selection
    model.to(device)                                                # [HOUSEWORK] move model to GPU
    model.eval()                                                    # [HOUSEWORK] eval mode

    train_dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)  # [HOUSEWORK] batch data
    batch_iterator = tqdm(train_dataloader)                         # [HOUSEWORK] progress bar

    predictions = []                                                # [HOUSEWORK] accumulator
    with torch.no_grad():                                           # [HOUSEWORK] disable gradients
        for batch_idx, batch in enumerate(batch_iterator):          # [HOUSEWORK] batch loop
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")  # [HOUSEWORK] tokenize
            inputs = {k: v.to(device) for k, v in inputs.items()}   # [HOUSEWORK] move to device

            outputs = model(**inputs)                               # [IMPORTANT] forward pass — the actual model inference
            logits = outputs.logits[:, :-1, :]                      # [IMPORTANT] shift logits: position i predicts token i+1
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)  # [IMPORTANT] convert to log-probabilities
            next_tokens = inputs['input_ids'][:, 1:]                # [IMPORTANT] the actual next tokens (ground truth)
            # Gather the logprob of the actual next token at each position
            next_logprobs = torch.gather(logprobs, -1, next_tokens.unsqueeze(-1)).squeeze(-1)  # [IMPORTANT] per-token logprob — THE raw signal

            for i in range(len(batch)):                             # [HOUSEWORK] per-sequence extraction
                # Only keep logprobs for non-padding positions
                predictions.append(next_logprobs[i][:inputs['attention_mask'][i,1:].sum()].cpu())  # [HOUSEWORK] strip padding, move to CPU

    if metric is not None:                                          # [HOUSEWORK] optional metric aggregation
        # Apply the metric function to aggregate per-token logprobs into a scalar per text
        metrics = []                                                # [HOUSEWORK]
        for text,prompt,prediction in zip(texts,prompts,predictions):  # [HOUSEWORK]
            metrics.append(                                         # [IMPORTANT] apply metric to summarize logprobs → scalar per sequence
                metric(
                    tokenizer.encode(text,truncation=True,return_tensors="pt").squeeze(0),
                    prediction,
                    tokenizer.encode(prompt,truncation=True,return_tensors="pt").squeeze(0)
                )
            )
    else:                                                           # [HOUSEWORK]
        metrics = None                                              # [HOUSEWORK]

    return predictions,metrics                                      # [HOUSEWORK] return results

def model_exists(save_path, epoch=0):                               # [HOUSEWORK] checkpoint existence check
    """Check if a saved model checkpoint exists at the given path."""
    return os.path.exists(os.path.join(save_path, f'epoch-{epoch}'))  # [HOUSEWORK]

def load_model_and_optimizer(save_path, epoch=0):                   # [HOUSEWORK] checkpoint loading
    """Load a saved LLaMA model and its optimizer state from a checkpoint."""
    model = LlamaForCausalLM.from_pretrained(os.path.join(save_path, f'epoch-{epoch}'))  # [HOUSEWORK]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)      # [HOUSEWORK]
    optimizer.load_state_dict(torch.load(os.path.join(save_path, f'epoch-{epoch}', "optimizer.pt")))  # [HOUSEWORK]

    return model, optimizer                                         # [HOUSEWORK]

def train_model(
    texts,
    tokenizer,
    index=None,
    save_path=None,
    batch_size=1,
    epochs=1,
    reshuffle=False,
    config=None,
    model=None,
    optimizer=None,
    shuffle=True,
    optimizer_params=None,
):
    """Train a LLaMA model on texts with controlled data ordering.

    This is critical for the experiments: the DATA ORDER during training is what
    creates the palimpsestic memorization signal. By controlling `index` (the
    random seed for shuffling) and `shuffle`, we can:
      - Train with a specific known order (shuffle=False → sequential)
      - Train with a reproducible random order (shuffle=True, index=seed)
      - Create reshuffled variants for permutation tests

    Args:
        texts: training texts in their canonical order
        index: random seed for shuffling (also serves as experiment identifier)
        shuffle: whether to shuffle the data (False → train in given order)
        reshuffle: whether to re-shuffle each epoch (vs. same order every epoch)
        model: existing model to continue training (None → create new from config)
        optimizer: existing optimizer state (None → create new)
    """
    if index is None:                                               # [HOUSEWORK] parameter validation
        assert shuffle == False, "index must be provided if shuffle is False"  # [HOUSEWORK]
    else:                                                           # [HOUSEWORK]
        random.seed(index)                                          # [IMPORTANT] seed RNG — controls the training order, which IS the palimpsestic signal

    if model is None:                                               # [HOUSEWORK] model initialization
        model = LlamaForCausalLM(config)                            # [HOUSEWORK] create fresh model from config
    if optimizer_params is None:                                    # [HOUSEWORK] default hyperparams
        optimizer_params = {"lr": 1e-5}                             # [HOUSEWORK]
    if optimizer is None:                                           # [HOUSEWORK]
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)  # [HOUSEWORK]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # [HOUSEWORK] device selection
    model.to(device)                                                # [HOUSEWORK]
    model.train()                                                   # [HOUSEWORK] training mode
    for k,v in optimizer.state.items():                             # [HOUSEWORK]
        optimizer.state[k] = put(v, device)                         # [HOUSEWORK] move optimizer state to GPU

    # Pre-compute shuffle orders for all epochs
    if shuffle:                                                     # [IMPORTANT] generate the training order — this is what Alice knows and the test statistic correlates against
        shuffle_orders = [random.sample(list(range(len(texts))), len(texts)) for _ in range(epochs)]  # [IMPORTANT]
    else:                                                           # [HOUSEWORK]
        shuffle_orders = [list(range(len(texts))) for _ in range(epochs)]  # [HOUSEWORK] sequential order

    for epoch in range(epochs):                                     # [HOUSEWORK] epoch loop
        if (epoch == 0) or reshuffle:                               # [HOUSEWORK]
            shuffle_order = shuffle_orders[epoch]                   # [IMPORTANT] select this epoch's training order
        shuffled_texts = [texts[i] for i in shuffle_order]          # [IMPORTANT] reorder texts according to the shuffle — data ordering is the experiment

        train_dataloader = DataLoader(shuffled_texts, batch_size=batch_size, shuffle=False)  # [HOUSEWORK] batch (shuffle=False: order already set above)
        batch_iterator = tqdm(train_dataloader)                     # [HOUSEWORK] progress bar

        # Standard causal LM training loop
        for batch_idx, batch in enumerate(batch_iterator):          # [HOUSEWORK] batch loop
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")  # [HOUSEWORK] tokenize
            inputs = {k: v.to(device) for k, v in inputs.items()}   # [HOUSEWORK] move to device
            inputs['labels'] = inputs['input_ids'].clone()          # [HOUSEWORK] causal LM: predict next token

            outputs = model(**inputs)                               # [IMPORTANT] forward pass
            loss = outputs.loss                                     # [IMPORTANT] cross-entropy loss

            optimizer.zero_grad()                                   # [HOUSEWORK] clear gradients
            loss.backward()                                         # [IMPORTANT] backpropagation
            optimizer.step()                                        # [IMPORTANT] weight update — each step deepens memorization of recent data

            wandb.log({                                             # [HOUSEWORK] experiment tracking
                "batch_loss": loss.item(),                          # [HOUSEWORK]
                "batch": batch_idx + epoch * len(train_dataloader), # [HOUSEWORK]
                "epoch": epoch,                                     # [HOUSEWORK]
            })

        if save_path is not None:                                   # [HOUSEWORK] checkpoint saving
            # Save model, tokenizer, and optimizer state for later resuming
            model.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))      # [HOUSEWORK]
            tokenizer.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))  # [HOUSEWORK]
            torch.save(optimizer.state_dict(), os.path.join(save_path, f'epoch-{epoch}', "optimizer.pt"))  # [HOUSEWORK]

    return model, optimizer, shuffle_orders                         # [IMPORTANT] returns shuffle_orders — needed to build the document_index for testing


def distill_model(
    teacher_model,
    texts,
    config,
    tokenizer,
    save_path,
    index,
    batch_size=1,
    epochs=1,
    temperature=1.0,
    hard_targets=False,
    optimizer_params=None,
):
    """Knowledge distillation: train a student model to mimic a teacher.

    This is one of Bob's possible "derivative" operations: instead of fine-tuning
    Alice's model directly, Bob could distill it into a new architecture.
    The question is whether the palimpsestic signal survives distillation.

    Two modes:
      - hard_targets=True: student learns from teacher's argmax predictions
      - hard_targets=False: student learns from teacher's softmax distribution
        (with temperature scaling for softer distributions)
    """
    student_model = LlamaForCausalLM(config)                        # [HOUSEWORK] create fresh student
    if optimizer_params is None:                                    # [HOUSEWORK]
        optimizer_params = {"lr": 1e-5}                             # [HOUSEWORK]
    optimizer = torch.optim.AdamW(student_model.parameters(), **optimizer_params)  # [HOUSEWORK]
    criterion = torch.nn.CrossEntropyLoss()                         # [HOUSEWORK] loss function

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # [HOUSEWORK]
    student_model.to(device)                                        # [HOUSEWORK]
    teacher_model.to(device)                                        # [HOUSEWORK]

    student_model.train()                                           # [HOUSEWORK]
    teacher_model.eval()                                            # [HOUSEWORK] teacher is frozen

    random.seed(index)                                              # [HOUSEWORK] reproducibility

    for epoch in range(epochs):                                     # [HOUSEWORK] epoch loop
        train_dataloader = DataLoader(texts, batch_size=batch_size, shuffle=True)  # [HOUSEWORK]
        batch_iterator = tqdm(train_dataloader)                     # [HOUSEWORK] progress bar

        for batch_idx, batch in enumerate(batch_iterator):          # [HOUSEWORK] batch loop
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")  # [HOUSEWORK]
            inputs = {k: v.to(device) for k, v in inputs.items()}   # [HOUSEWORK]

            # Get targets from teacher (no gradient needed)
            with torch.no_grad():                                   # [HOUSEWORK]
                teacher_outputs = teacher_model(**inputs).logits     # [IMPORTANT] teacher forward pass — source of the distilled knowledge
                if hard_targets:                                    # [IMPORTANT] target mode selection
                    targets = torch.argmax(teacher_outputs, dim=-1)  # [IMPORTANT] hard: argmax token
                else:
                    targets = torch.nn.functional.softmax(teacher_outputs / temperature, dim=-1)  # [IMPORTANT] soft: temperature-scaled distribution

            # Get student predictions
            student_outputs = student_model(**inputs).logits         # [IMPORTANT] student forward pass

            # Distillation loss: student vs teacher
            if hard_targets:                                        # [HOUSEWORK] branch on mode
                loss = criterion(student_outputs.transpose(1,2),targets)  # [IMPORTANT] distillation loss (hard)
            else:
                loss = criterion(student_outputs.transpose(1, 2), targets.transpose(1, 2))  # [IMPORTANT] distillation loss (soft)

            optimizer.zero_grad()                                   # [HOUSEWORK]
            loss.backward()                                         # [IMPORTANT] backpropagation
            optimizer.step()                                        # [IMPORTANT] weight update

            wandb.log({                                             # [HOUSEWORK] experiment tracking
                "batch_loss": loss.item(),                          # [HOUSEWORK]
                "batch": batch_idx + epoch * len(train_dataloader), # [HOUSEWORK]
                "epoch": epoch,                                     # [HOUSEWORK]
            })

        student_model.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))  # [HOUSEWORK] checkpoint
        tokenizer.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))      # [HOUSEWORK]

    return student_model, optimizer                                 # [HOUSEWORK]


def generate(prompts, model_checkpoint_path, sampling_params, seed=0, prompt_template="{prompt}", revision=None):
    """Generate text from a model using vLLM (fast batched inference).

    This simulates "Bob producing text" — the output that Alice observes
    in the observational setting.
    """
    llm = LLM(model=model_checkpoint_path, seed=seed, revision=revision)  # [HOUSEWORK] load model into vLLM

    prompts = [prompt_template.format(prompt=prompt) for prompt in prompts]  # [HOUSEWORK] apply prompt template
    outputs = llm.generate(prompts, sampling_params)                # [IMPORTANT] generate text — this is what Alice observes in the observational setting

    generated_texts = [output.outputs[0].text for output in outputs]  # [HOUSEWORK] extract text from vLLM output
    return generated_texts                                          # [HOUSEWORK]
