import os
from datasets import load_dataset


def load_and_preprocess_data(tokenizer):
    # Load dataset based on env var
    dataset_name = os.getenv("FT_DATASET", "squad")
    if dataset_name == "squad":
        dataset = load_dataset("rajpurkar/squad")
        dataset["train"] = dataset["train"].select(range(1000))
        dataset["validation"] = dataset["validation"].select(range(200))
    else:
        # Add support for other datasets
        dataset = load_dataset(dataset_name)
    print("Original train columns:", dataset["train"].column_names)

    # Preprocessing function
    def preprocess_function(examples):
        question = examples["question"].strip()
        context = examples["context"]
        answers = examples["answers"]
        inputs = tokenizer(
            question,
            context,
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answer = answers
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids()

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if (
            offset_mapping[context_start][0] > end_char
            or offset_mapping[context_end][1] < start_char
        ):
            start_position = 0
            end_position = 0
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset_mapping[idx][0] <= start_char:
                idx += 1
            start_position = idx - 1

            idx = context_end
            while idx >= context_start and offset_mapping[idx][1] >= end_char:
                idx -= 1
            end_position = idx + 1

        inputs["start_positions"] = start_position
        inputs["end_positions"] = end_position
        return inputs

    # Tokenize datasets
    tokenized_squad = dataset.map(
        preprocess_function,
        batched=False,
        remove_columns=["question", "context", "answers", "id", "title"],
    )
    print("Tokenized train columns:", tokenized_squad["train"].column_names)
    return tokenized_squad
