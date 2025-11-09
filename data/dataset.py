from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple



class Dataset_class:
    """Custom dataset class with stratified splitting."""
    
    def __init__(
        self,
        dataset_name: str,
        dataset: ImageFolder,
        train_split: float = 0.6,
        val_split: float = 0.2,
        test_split: float = 0.2,
        random_state: int = 42,
        num_workers: int = 4,
        drop_last: bool = True
    ):
        """
        Initialize dataset with stratified splits.
        
        Args:
            dataset_name: Name of the dataset
            dataset: PyTorch ImageFolder dataset
            train_split: Training set ratio
            val_split: Validation set ratio
            test_split: Test set ratio
            random_state: Random seed for reproducibility
            num_workers: Number of workers for data loading
            drop_last: Whether to drop last incomplete batch
        """
        self.name = dataset_name
        self.dataset = dataset
        self.num_classes = len(self.dataset.classes)
        self.class_names = self.dataset.classes
        num_examples = len(self.dataset)
        
        self._num_workers = num_workers
        self._drop_last = drop_last
        
        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
        
        # Get targets for stratification
        targets = self.dataset.targets
        
        # Step 1: Split into train and temp (val + test)
        train_indices, temp_indices, _, temp_targets = train_test_split(
            list(range(num_examples)),
            targets,
            test_size=(val_split + test_split),
            stratify=targets,
            random_state=random_state
        )
        
        # Step 2: Split temp into val and test
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_split / (val_split + test_split),
            stratify=temp_targets,
            random_state=random_state
        )
        
        # Store splits
        self._splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        print(f"Dataset: {self.name}")
        print(f"Total samples: {num_examples}")
        print(f"Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")
        print(f"Number of classes: {self.num_classes}")
    
    def get_dataloader(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = False
    ) -> DataLoader:
        """
        Create DataLoader for specified split.
        
        Args:
            split: One of 'train', 'val', 'test'
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader instance
        """
        assert split in self._splits, f"Invalid split: {split}"
        
        indices = self._splits[split]
        subset = Subset(self.dataset, indices)
        
        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            drop_last=self._drop_last,
            pin_memory=True
        )
        
        return dataloader
    
    def get_split_indices(self, split: str):
        """Get indices for a specific split."""
        return self._splits[split]
    
    def get_class_distribution(self, split: str) -> Dict[str, int]:
        """Get class distribution for a split."""
        indices = self._splits[split]
        targets = [self.dataset.targets[i] for i in indices]
        
        distribution = {}
        for class_idx, class_name in enumerate(self.class_names):
            count = targets.count(class_idx)
            distribution[class_name] = count
        
        return distribution


def get_dataloaders(
    dataset_root: str,
    dataset_name: str,
    image_size: Tuple[int, int],
    batch_size: int,
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Dataset_class]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_root: Path to dataset root directory
        dataset_name: Name of the dataset
        image_size: Target image size (height, width)
        batch_size: Batch size for dataloaders
        train_split: Training set ratio
        val_split: Validation set ratio
        test_split: Test set ratio
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_wrapper)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    dataset = ImageFolder(dataset_root, transform=transform)
    
    # Create dataset wrapper with splits
    dataset_wrapper = Dataset_class(
        dataset_name=dataset_name,
        dataset=dataset,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        num_workers=num_workers,
        **kwargs
    )
    
    # Create dataloaders
    train_loader = dataset_wrapper.get_dataloader('train', batch_size, shuffle=True)
    val_loader = dataset_wrapper.get_dataloader('val', batch_size, shuffle=False)
    test_loader = dataset_wrapper.get_dataloader('test', batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset_wrapper