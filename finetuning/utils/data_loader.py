
import pickle

def load_dataset(dataset_name, datapath="../data/"):
    """Load a preprocessed dataset from a .pkl file."""
    with open(datapath + dataset_name + "_data.pkl", "rb") as f:
        return pickle.load(f)
