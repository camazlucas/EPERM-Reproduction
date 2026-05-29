import json
import random
from pathlib import Path

# =====================================================
# CONFIGURAÇÕES
# =====================================================

INPUT_FILE = r"D:\KGQAArmazenamento\DATASET EPERM\Modulo 1\retriever_train.jsonl"

TRAIN_OUTPUT = r"D:\KGQAArmazenamento\DATASET EPERM\Modulo 1\train.json"
VALID_OUTPUT = r"D:\KGQAArmazenamento\DATASET EPERM\Modulo 1\test.json"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# =====================================================
# FUNÇÕES
# =====================================================

def build_response(gold_relations):
    """
    Monta a resposta XML.
    """

    budget = len(gold_relations)

    linhas = []

    for i, rel in enumerate(gold_relations):

        restante = budget - i

        linhas.append(
            f"<count>{restante}</count>"
        )

        linhas.append(
            f"<choice>{rel}</choice>"
        )

        linhas.append(
            "<score>1.0</score>"
        )

    return "\n".join(linhas)


def convert_example(item):
    """
    Converte um exemplo do retriever dataset
    para o formato do fine-tuning.
    """

    topic_entities = ";".join(
        item["topic_entities"]
    )

    candidate_relations = ";".join(
        item["candidate_relations"]
    )

    gold_relations = item["gold_relations"]

    return {
        "question": item["question"],
        "topic_entities": topic_entities,
        "candidate_relations": candidate_relations,
        "budget": len(gold_relations),
        "response": build_response(
            gold_relations
        )
    }


# =====================================================
# CARREGA DATASET
# =====================================================

print("Lendo dataset...")

examples = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf8"
) as f:

    for line in f:

        item = json.loads(line)

        examples.append(
            convert_example(item)
        )

print(
    f"Exemplos carregados: "
    f"{len(examples):,}"
)

# =====================================================
# EMBARALHA
# =====================================================

random.seed(RANDOM_SEED)

random.shuffle(examples)

# =====================================================
# SPLIT
# =====================================================

train_size = int(
    len(examples) * TRAIN_RATIO
)

train_examples = examples[:train_size]

valid_examples = examples[train_size:]

# =====================================================
# SALVA
# =====================================================

with open(
    TRAIN_OUTPUT,
    "w",
    encoding="utf8"
) as f:

    json.dump(
        train_examples,
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    VALID_OUTPUT,
    "w",
    encoding="utf8"
) as f:

    json.dump(
        valid_examples,
        f,
        ensure_ascii=False,
        indent=2
    )

# =====================================================
# ESTATÍSTICAS
# =====================================================

print("\n==============================")
print("ESTATÍSTICAS")
print("==============================")

print(
    f"Total: {len(examples):,}"
)

print(
    f"Treino: {len(train_examples):,}"
)

print(
    f"Validação: {len(valid_examples):,}"
)

print(
    f"Arquivo treino: {TRAIN_OUTPUT}"
)

print(
    f"Arquivo validação: {VALID_OUTPUT}"
)

print("\nDataset gerado com sucesso.")