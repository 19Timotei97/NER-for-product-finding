import numpy as np
import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
import logging
from custom_model import NERModel
import create_dataset_2_0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info(f"Running on {device}")

PRECISION_RECALL_OFFSET = 0.1
MAX_LEN = 512
MAX_EPOCHS = 50
BATCH_SIZE = 128

# TODO: test DistilRoberta for possible better results
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
# model_orig_backbone = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model_orig_backbone = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-uncased")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Freeze the backbone to make sure only the FC layers are trained
model = NERModel(backbone=model_orig_backbone)

for param in model.backbone.parameters():
    param.requires_grad = False

model.to(device)

train_dataset, val_dataset, neg_pos_ratio = create_dataset_2_0.load_dataset(
    json_path='formatted',
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    split_percent=0.95
)


# Construct training / validation loaders with dynamic batching
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=data_collator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=data_collator
)


criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(
    [PRECISION_RECALL_OFFSET * neg_pos_ratio]).to(device))

optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4)

# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1, end_factor=0.01, total_iters=MAX_EPOCHS)

# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=MAX_EPOCHS, last_epoch=48, eta_min=3e-6
# )

# for early stopping
min_validation_loss = None

# Start the training process
for epoch in range(MAX_EPOCHS):
    training_loss = 0
    model.train()

    for batch_id, data in enumerate(train_loader):

        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    logging.info(
        f"Training @{epoch} : {training_loss * BATCH_SIZE / len(train_loader)}")

    validation_loss = 0
    tps, tns, fps, fns = 0, 0, 0, 0
    model.eval()

    with torch.no_grad():
        for batch_id, data in enumerate(val_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            loss = criterion(outputs, targets)

            outputs = torch.sigmoid(outputs)

            for i in range(outputs.shape[0]):
                if outputs[i] < 0.5 and targets[i] < 0.5:
                    tns += 1

                if outputs[i] < 0.5 < targets[i]:
                    fns += 1

                if outputs[i] > 0.5 > targets[i]:
                    fps += 1

                if outputs[i] > 0.5 and targets[i] > 0.5:
                    tps += 1

            validation_loss += loss.item()

        logging.info(
            f"Validation @{epoch} : {validation_loss * BATCH_SIZE / len(val_loader)}")

        if min_validation_loss is None or validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            logging.info(
                f"Saving new model > checkpoints/{min_validation_loss}.dat")
            torch.save(model.state_dict(),
                       f"checkpoints/{min_validation_loss}.dat")

        logging.info(f'TP:{tps} TN:{tns} FP:{fps} FN:{fns}')

        scheduler.step()

# Or we could do it this way
"""
label_list = [0, 1]
seqeval = evaluate.load("seqeval")


# https://huggingface.co/docs/transformers/tasks/token_classification
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label)]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label)]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
        output_dir='./checkpoints',
        learning_rate=3e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=MAX_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

logging.info("Starting training...")
trainer.train()

"""