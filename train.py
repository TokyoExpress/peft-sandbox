from transformers import GPTNeoXForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from datasets import load_dataset
from dataset import ShiftingDataset
from tqdm import tqdm
import numpy as np
import argparse
import torch
import wandb

def get_args():
    parser = argparse.ArgumentParser(description="Build index")
    parser.add_argument(
        "--run_name",
        help="name of training run / model",
    )
    parser.add_argument(
        "-r",
        "--rank",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        type=float,
        default=0.1,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="seed for shuffling dataset",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
    )
    return parser.parse_args()

def initialize_model(rank: int, alpha: int, dropout: float):

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=rank, lora_alpha=alpha, lora_dropout=dropout)

    model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step143000",
    cache_dir="./pythia-70m-deduped/step143000",
    )

    tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step143000",
    cache_dir="./pythia-70m-deduped/step143000",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer

if __name__ == "__main__":
    args = get_args()

    run = wandb.init(
        project='pythia-peft-moonscript',
        name=args.run_name,
        config={
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "seed": args.seed,
            "seq_length": args.seq_length,
            "offset": args.offset,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "train_batch_size": args.train_batch_size,
            "val_batch_size": args.val_batch_size,
            "warmup_steps": args.warmup_steps
        })

    # INIT MODEL / TOKENIZER
    model, tokenizer = initialize_model(args.rank, args.alpha, args.dropout)

    # CREATE DATASET / DATALOADER
    dataset = load_dataset("bigcode/the-stack", data_dir="data/moonscript", split="train").train_test_split(test_size=0.1, seed=args.seed)

    print('\nCreating train set.\n')
    train_dataset = ShiftingDataset(args.seq_length)
    train_dataset.preprocess(dataset['train'], tokenizer)
    train_dataset.regenerate()

    print('\nCreating validation set.\n')
    val_dataset = ShiftingDataset(args.seq_length)
    val_dataset.preprocess(dataset['test'], tokenizer)
    val_dataset.regenerate()

    print('\n' + str(len(train_dataset)) + ' sequences for training, ' + str(len(val_dataset)) + ' for validation.')

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size)

    # INIT OPTIMIZER / LR SCHEDULER
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    print('\nComponents ready. Beginning training.')

    # TRAIN
    model = model.to('cuda')

    steps = 0
    for epoch in range(args.num_epochs):
        print('\nEpoch', epoch+1)
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            loss = model(**batch).loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            wandb.log({"train_loss": loss.detach().float(), "learning_rate": lr_scheduler.get_last_lr()[0]}, step=steps)

            steps += 1

        print('train_loss:', total_loss / len(train_dataloader))

        # EVALUATE
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch = {k: v.to('cuda') for k, v in batch.items()}
                val_loss += (model(**batch).loss.detach().float())

        print('val_loss:', val_loss / len(val_dataloader))

        wandb.log({"val_loss": val_loss / len(val_dataloader)}, step=steps)

        # REGENERATE TRAIN SET
        if int(args.offset) > 0:
            train_dataset.regenerate(args.offset)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    # SAVE
    model.save_pretrained('runs/' + args.run_name)