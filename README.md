# peft-sandbox
Code to train and evaluate language models using HuggingFace PEFT and LoRA

## Instructions

Install requirements first:

```
pip install -r requirements.txt
```

You'll need to login to HuggingFace (to receive access to The Stack) and Weights & Biases (to log metrics).

```
huggingface-cli login
wandb login
```

Now you can kick off a training run with the default hyperparameters using:

```
python train.py --run_name default
```

Below are the adjustable parameters and their default values:

```
--rank: 8                   # LoRA rank
--alpha: 32                 # LoRA alpha
--dropout: 0.1              # LoRA dropout
--seed: 21                  # seed for dataset train/val split
--seq_length: 2048          # sequence length for packing
--offset: 0                 # offset parameter for ShiftingDataset
--lr: 1e-3
--num_epochs: 10
--train_batch_size: 16
--val_batch_size: 16
--warmup_steps: 100
```

After training, you can load the model for inference as follows:

```
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

model_path = "path/to/model/dir"

config = PeftConfig.from_pretrained(model_path)
model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
)
model = PeftModel.from_pretrained(model, model_path)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step143000",
)

text = '''my_func ='''

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=256)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
```

## Shifting Dataset

`dataset.py` contains an implementation of `ShiftingDataset`, which allows the option for an `offset` parameter. After each epoch, the entire training set is shifted by `offset` tokens before being chunked again, allowing the model to see contiguous sequences that might have been cut off between sequences otherwise.

After experimentation, there seemed to be no discernable benefit in model performance from this, so the default value of `offset` was set to 0.

A Weights & Biases project containing runs with different values of `offset` can be found [here](https://wandb.ai/samuelzguo/pythia-peft-moonscript).
