"""
Test script for MultiClassJaccardIndex from torchmetrics.
"""

import torch
from torchmetrics import JaccardIndex



def test_jaccard_index():
    """Test JaccardIndex with 3 classes."""
    
    num_classes = 3

    preds = torch.tensor([0, 1, 2, 2])
    targets = torch.tensor([0, 1, 2, 3])
    
    print("\n--- Torchmetrics Calculation ---")
    
    jaccard_macro = JaccardIndex(
        task="multiclass", 
        num_classes=num_classes, 
        average="macro",
        ignore_index=num_classes
    )
    iou_macro = jaccard_macro(preds, targets)
    print(f"Torchmetrics mIoU (macro): {iou_macro:.4f}")
    

if __name__ == "__main__":
    test_jaccard_index()
