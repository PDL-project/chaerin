"""Finetune the sentence transformer model on a custom dataset."""
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sentence_transformers import losses
from datasets import load_dataset

# model = SentenceTransformer("embedding-data/distilroberta-base-sentence-transformer")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
csv_file_path = "AI2Thor/sentence_transformer/data.csv"
# csv_file_path = "data.csv"
dataset = load_dataset("csv", data_files=csv_file_path)

train_examples = []
train_data = dataset["train"]
n_examples = dataset["train"].num_rows

for i in range(n_examples):
    task = train_data[i]["Task"]
    function = train_data[i]["Function"]
    train_examples.append(InputExample(texts=[task, function]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
num_epochs = 10
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data


model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
)


# def get_closest_feasible_action(model, embeddings, action: str):
#     """To convert actions like RotateLeft to Rotate(Left)"""
#     action_embedding = torch.FloatTensor(model.encode([action]))
#     scores = torch.cosine_similarity(embeddings, action_embedding)
#     max_score, max_idx = torch.max(scores, 0)
#     return all_actions[max_idx]
