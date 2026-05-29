import json
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

from trl import SFTTrainer

# =====================================================
# CONFIGURAÇÕES
# =====================================================

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

TRAIN_FILE = "../data/WebQSP/train.json"
VALID_FILE = "../data/WebQSP/valid.json"

OUTPUT_DIR = "../Subgraph Generator/outputs/retriever_llama2_lora"

MAX_SEQ_LENGTH = 2048

NUM_EPOCHS = 3

LEARNING_RATE = 2e-4

BATCH_SIZE = 1

GRAD_ACCUM = 8

# =====================================================
# PROMPT DO PAPER
# =====================================================

INSTRUCTION = """ Assume you are a **semantic analysis expert**. You will receive an encyclopedic question, related topic entities, and a
set of several retrieved relationships that need to be filtered (which need to assist in inferring the question). Your task is
to carefully consider the information needed to reason about the question and, based on the semantics of the existing
reasoning path, select the top budget relationships from the set that are most likely to help infer the answer to the question.
Guidelines
1. The format of the input is:
**Question**:
The input question
**Topic entity**:
The related topic entities
**Several retrieved relationships**:
A set of several retrieved relationships separated by semicolons (;)
**Budget**:
The number of the selected relationships that are most likely to infer the result.
2. The number of the selected relationships are no more than {{budget}}. reset counter between < count > and
< /count > to {{budget}}.
3. You are allowed to select {{budget}} relationships (starting budget), keep track of it by counting down within tags
< count > < /count >, STOP GENERATING MORE RELATIONSHIPS when hitting 0.
4. Please provide your count, reasons, scores, and selected relationships in the following XML format.
< count > [starting budget] < /count >
< choice > The relationship you select that is most likely to infer the question. < /choice >
< reason > Provide the reasons for the score you assigned to the relationship 1 for helping infer the questions.
< /reason >
< score > The confidence score 0.0-1.0 to select this relation < /score >
< count > [remaining budget] < /count >
< choice > The 2-th relationship you select that is likely to infer the questions. < /choice >
< reason > Provide the reasons for the score you assigned to the relationship 2 for helping infer the questions.
< /reason >
< score > The confidence score 0.0-1.0 to select this relationship < /score >
...
< count > 1 < /count >
< choice > The {{budget}}-th relationship you select that is likely to infer the questions. < /choice >
< reason > Provide the reasons for the score you assigned to the relationship {{budget}} for helping infer the questions.
< /reason >
< score > The confidence score 0.0-1.0 to select this relationship < /score >
"""

# =====================================================
# DATASET
# =====================================================

def load_json(path):

    with open(path, "r", encoding="utf8") as f:

        return json.load(f)


train_data = load_json(TRAIN_FILE)

valid_data = load_json(VALID_FILE)

print(f"Treino: {len(train_data):,}")
print(f"Validação: {len(valid_data):,}")

# =====================================================
# MONTA PROMPT
# =====================================================

def format_example(example):

    prompt = f"""
Instruction:
{INSTRUCTION}

Input:

**Question**:
{example['question']}

**Topic entity**:
{example['topic_entities']}

**Several retrieved relationships**:
{example['candidate_relations']}

**Budget**:
{example['budget']}

Output:
"""

    text = (
        prompt
        + example["response"]
        + tokenizer.eos_token
    )

    return {
        "text": text
    }

# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# CONVERSÃO
# =====================================================

train_dataset = Dataset.from_list(train_data)

valid_dataset = Dataset.from_list(valid_data)

train_dataset = train_dataset.map(
    format_example
)

valid_dataset = valid_dataset.map(
    format_example
)

# =====================================================
# QUANTIZAÇÃO
# =====================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# =====================================================
# MODELO
# =====================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(
    model
)

# =====================================================
# LORA
# =====================================================

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

model = get_peft_model(
    model,
    peft_config
)

model.print_trainable_parameters()

# =====================================================
# TREINAMENTO
# =====================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LEARNING_RATE,

    warmup_ratio=0.03,

    weight_decay=0.01,

    logging_steps=20,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    fp16=True,

    report_to="none"
)

trainer = SFTTrainer(
    model=model,

    train_dataset=train_dataset,

    eval_dataset=valid_dataset,

    args=training_args,

    processing_class=tokenizer,

)

# =====================================================
# TREINO
# =====================================================

trainer.train()

# =====================================================
# SALVA
# =====================================================

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\nTreinamento finalizado.")
print(f"Modelo salvo em: {OUTPUT_DIR}")