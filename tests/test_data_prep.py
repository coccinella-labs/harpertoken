import os
from transformers import AutoTokenizer
from scripts.data_prep import load_and_preprocess_data


def test_load_and_preprocess_data():
    # Mock env
    os.environ["FT_DATASET"] = "squad"
    tokenizer = AutoTokenizer.from_pretrained("harpertoken/quiz")
    data = load_and_preprocess_data(tokenizer)
    assert "train" in data
    assert "validation" in data
    assert len(data["train"]) > 0
    assert "input_ids" in data["train"][0]
    print("Data prep test passed!")


if __name__ == "__main__":
    test_load_and_preprocess_data()
