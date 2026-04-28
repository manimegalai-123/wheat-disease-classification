import os

def test_files_exist():
    assert os.path.exists("wheat_leaf_model_efficientnet.pth")
    assert os.path.exists("test1.png")
