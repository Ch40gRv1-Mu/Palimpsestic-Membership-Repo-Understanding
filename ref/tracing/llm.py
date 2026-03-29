"""
Core LLM utilities: training, evaluation, generation, and distillation.

This module provides the building blocks for the experiments:
  - evaluate_model: compute per-token log-probabilities (the "signal" we test)
  - train_model: train a small LLM from scratch with controlled data ordering
  - distill_model: knowledge distillation (teacher→student)
  - generate: sample text from a trained model
"""

import torch
from transformers import LlamaForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
import random
from vllm import LLM

def put(obj, device):
    """Recursively move tensors/dicts/lists to a device. Used for optimizer state."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: put(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [put(v, device) for v in obj]
    else:
        return obj

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
    if prompts is None:
        prompts = [""] * len(texts)
    else:
        assert len(texts) == len(prompts), "texts and prompts must have the same length"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    train_dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
    batch_iterator = tqdm(train_dataloader)

    predictions = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(batch_iterator):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]                        # shift: predict next token from each position
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1) # convert to log-probabilities
            next_tokens = inputs['input_ids'][:, 1:]                   # the actual next tokens (targets)
            # Gather the logprob of the actual next token at each position
            next_logprobs = torch.gather(logprobs, -1, next_tokens.unsqueeze(-1)).squeeze(-1)

            for i in range(len(batch)):
                # Only keep logprobs for non-padding positions
                predictions.append(next_logprobs[i][:inputs['attention_mask'][i,1:].sum()].cpu())

    if metric is not None:
        # Apply the metric function to aggregate per-token logprobs into a scalar per text
        metrics = []
        for text,prompt,prediction in zip(texts,prompts,predictions):
            metrics.append(
                metric(
                    tokenizer.encode(text,truncation=True,return_tensors="pt").squeeze(0),
                    prediction,
                    tokenizer.encode(prompt,truncation=True,return_tensors="pt").squeeze(0)
                )
            )
    else:
        metrics = None

    return predictions,metrics

def model_exists(save_path, epoch=0):
    """Check if a saved model checkpoint exists at the given path."""
    return os.path.exists(os.path.join(save_path, f'epoch-{epoch}'))

def load_model_and_optimizer(save_path, epoch=0):
    """Load a saved LLaMA model and its optimizer state from a checkpoint."""
    model = LlamaForCausalLM.from_pretrained(os.path.join(save_path, f'epoch-{epoch}'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.load_state_dict(torch.load(os.path.join(save_path, f'epoch-{epoch}', "optimizer.pt")))

    return model, optimizer

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
    if index is None:
        assert shuffle == False, "index must be provided if shuffle is False"
    else:
        random.seed(index)    # seed controls the training order → reproducible

    if model is None:
        model = LlamaForCausalLM(config)      # create a fresh model from config
    if optimizer_params is None:
        optimizer_params = {"lr": 1e-5}
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for k,v in optimizer.state.items():
        optimizer.state[k] = put(v, device)    # move optimizer state to GPU

    # Pre-compute shuffle orders for all epochs
    if shuffle:
        shuffle_orders = [random.sample(list(range(len(texts))), len(texts)) for _ in range(epochs)]
    else:
        shuffle_orders = [list(range(len(texts))) for _ in range(epochs)]  # sequential order

    for epoch in range(epochs):
        if (epoch == 0) or reshuffle:
            shuffle_order = shuffle_orders[epoch]
        shuffled_texts = [texts[i] for i in shuffle_order]

        train_dataloader = DataLoader(shuffled_texts, batch_size=batch_size, shuffle=False)
        batch_iterator = tqdm(train_dataloader)

        # Standard causal LM training loop
        for batch_idx, batch in enumerate(batch_iterator):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs['labels'] = inputs['input_ids'].clone()   # causal LM: predict next token

            outputs = model(**inputs)
            loss = outputs.loss                               # cross-entropy loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx + epoch * len(train_dataloader),
                "epoch": epoch,
            })

        if save_path is not None:
            # Save model, tokenizer, and optimizer state for later resuming
            model.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))
            tokenizer.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))
            torch.save(optimizer.state_dict(), os.path.join(save_path, f'epoch-{epoch}', "optimizer.pt"))

    return model, optimizer, shuffle_orders


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
    student_model = LlamaForCausalLM(config)
    if optimizer_params is None:
        optimizer_params = {"lr": 1e-5}
    optimizer = torch.optim.AdamW(student_model.parameters(), **optimizer_params)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    teacher_model.to(device)

    student_model.train()
    teacher_model.eval()        # teacher is frozen

    random.seed(index)

    for epoch in range(epochs):
        train_dataloader = DataLoader(texts, batch_size=batch_size, shuffle=True)
        batch_iterator = tqdm(train_dataloader)

        for batch_idx, batch in enumerate(batch_iterator):
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get targets from teacher (no gradient needed)
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs).logits
                if hard_targets:
                    targets = torch.argmax(teacher_outputs, dim=-1)           # hard: argmax token
                else:
                    targets = torch.nn.functional.softmax(teacher_outputs / temperature, dim=-1)  # soft: probability distribution

            # Get student predictions
            student_outputs = student_model(**inputs).logits

            # Distillation loss: student vs teacher
            if hard_targets:
                loss = criterion(student_outputs.transpose(1,2),targets)
            else:
                loss = criterion(student_outputs.transpose(1, 2), targets.transpose(1, 2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx + epoch * len(train_dataloader),
                "epoch": epoch,
            })

        student_model.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))
        tokenizer.save_pretrained(os.path.join(save_path, f'epoch-{epoch}'))

    return student_model, optimizer


def generate(prompts, model_checkpoint_path, sampling_params, seed=0, prompt_template="{prompt}", revision=None):
    """Generate text from a model using vLLM (fast batched inference).

    This simulates "Bob producing text" — the output that Alice observes
    in the observational setting.
    """
    llm = LLM(model=model_checkpoint_path, seed=seed, revision=revision)

    prompts = [prompt_template.format(prompt=prompt) for prompt in prompts]
    outputs = llm.generate(prompts, sampling_params)

    generated_texts = [output.outputs[0].text for output in outputs]
    return generated_texts