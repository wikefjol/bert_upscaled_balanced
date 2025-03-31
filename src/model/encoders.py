class LabelEncoder:
    def __init__(self, labels):
        # Extract unique labels from the provided array
        unique_labels = set(labels)

        # Create dictionaries for encoding and decoding
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(unique_labels)}

    def encode(self, label):
        # Encode label to its corresponding index
        return self.label_to_index.get(label, None)

    def decode(self, index):
        # Decode index to its corresponding label
        return self.index_to_label.get(index, None)