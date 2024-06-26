import pytest
import os


def get_data_ratio(data_type):
    # Track the total, legitimate and phishing marked urls
    total = 0
    legitimate = 0
    phishing = 0

    # Count all occurrences and which are legitimate and phishing
    with open(os.path.join("data", "raw", f"{data_type}.txt"), "r") as f:
        for line in f:
            total += 1
            if line.startswith("legitimate"):
                legitimate += 1
            elif line.startswith("phishing"):
                phishing += 1

    # Return the occurrence counts
    return total, legitimate, phishing


def test_train_data_ratio():
    # The legitimate and phishing should have a fair ratio and add up to the total together
    total, legitimate, phishing = get_data_ratio("train")
    assert total == legitimate + phishing
    assert 0.4 < legitimate / total < 0.6
    assert 0.4 < phishing / total < 0.6


def test_val_data_ratio():
    # The legitimate and phishing should have a fair ratio and add up to the total together
    total, legitimate, phishing = get_data_ratio("val")
    assert total == legitimate + phishing
    assert 0.4 < legitimate / total < 0.6
    assert 0.4 < phishing / total < 0.6


def test_test_data_ratio():
    # The legitimate and phishing should have a fair ratio and add up to the total together
    total, legitimate, phishing = get_data_ratio("test")
    assert total == legitimate + phishing
    assert 0.4 < legitimate / total < 0.6
    assert 0.4 < phishing / total < 0.6


if __name__ == "__main__":
    pytest.main()
    pass
