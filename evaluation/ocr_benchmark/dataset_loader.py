# Loads the HuggingFace dataset kiethuynhanh/vnpdf-financial-reports-dataset

from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any
from dataclasses import dataclass

from PIL import Image
from datasets import load_dataset, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from logger import get_logger

logger = get_logger(__name__)

# Dataset config
DATASET_NAME = "kiethuynhanh/vnpdf-financial-reports-dataset"
CACHE_DIR = Path("data/hf_cache")


@dataclass
class VnPdfSample:
    """A single sample from the VnPDF dataset."""
    image: Image.Image
    text: str
    custom_id: str
    page_number: int
    report_id: str
    
    def save_image(self, path: str) -> None:
        """Save the image to a file."""
        self.image.save(path)
    
    @property
    def text_length(self) -> int:
        """Get the length of the ground truth text."""
        return len(self.text)
    
    @property
    def is_table_page(self) -> bool:
        """Heuristic to detect if this is a table-heavy page."""
        # Tables often have many numbers and pipe characters
        text = self.text
        number_count = sum(1 for c in text if c.isdigit())
        pipe_count = text.count('|')
        dash_count = text.count('-')
        
        # Heuristic: if >10% of text is numbers or has table-like patterns
        if len(text) > 0:
            number_ratio = number_count / len(text)
            return number_ratio > 0.1 or pipe_count > 10 or dash_count > 20
        return False


class VnPdfDataset:
    """
    Loader for the VnPDF Financial Reports dataset.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the dataset loader.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._dataset: Optional[Dataset] = None
        
    def load(self) -> 'VnPdfDataset':
        """Load the dataset from HuggingFace."""
        logger.info(f"Loading dataset: {DATASET_NAME}")
        
        try:
            self._dataset = load_dataset(
                DATASET_NAME,
                cache_dir=str(self.cache_dir),
                split="train"
            )
            logger.info(f"Loaded {len(self._dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        return self
    
    @property
    def dataset(self) -> Dataset:
        """Get the underlying HuggingFace dataset, loading if needed."""
        if self._dataset is None:
            self.load()
        return self._dataset
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> VnPdfSample:
        """Get a sample by index."""
        item = self.dataset[idx]
        return VnPdfSample(
            image=item["image"],
            text=item["text"],
            custom_id=item["custom_id"],
            page_number=item["page_number"],
            report_id=item["report_id"],
        )
    
    def __iter__(self) -> Iterator[VnPdfSample]:
        """Iterate over all samples."""
        for i in range(len(self)):
            yield self[i]
    
    def get_samples(self, n: Optional[int] = None, shuffle: bool = False) -> List[VnPdfSample]:
        """
        Get a list of samples.
        """
        ds = self.dataset
        
        if shuffle:
            ds = ds.shuffle(seed=42)
        
        if n is not None:
            ds = ds.select(range(min(n, len(ds))))
        
        return [
            VnPdfSample(
                image=item["image"],
                text=item["text"],
                custom_id=item["custom_id"],
                page_number=item["page_number"],
                report_id=item["report_id"],
            )
            for item in ds
        ]
    
    def get_table_pages(self) -> List[VnPdfSample]:
        """Get all samples that appear to be table-heavy pages."""
        return [s for s in self if s.is_table_page]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        samples = list(self)
        text_lengths = [s.text_length for s in samples]
        table_pages = [s for s in samples if s.is_table_page]
        
        return {
            "total_samples": len(samples),
            "table_pages": len(table_pages),
            "non_table_pages": len(samples) - len(table_pages),
            "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "min_text_length": min(text_lengths) if text_lengths else 0,
            "max_text_length": max(text_lengths) if text_lengths else 0,
            "unique_reports": len(set(s.report_id for s in samples)),
        }


def main():
    """Test the dataset loader."""
    print("Loading VnPDF Financial Reports dataset...")
    
    dataset = VnPdfDataset()
    dataset.load()
    
    print("\nDataset Statistics:")
    stats = dataset.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nFirst 3 samples:")
    for i, sample in enumerate(dataset.get_samples(3)):
        print(f"\n  Sample {i+1}:")
        print(f"    ID: {sample.custom_id}")
        print(f"    Report: {sample.report_id}")
        print(f"    Page: {sample.page_number}")
        print(f"    Text length: {sample.text_length} chars")
        print(f"    Is table page: {sample.is_table_page}")
        print(f"    Text preview: {sample.text[:100]}...")


if __name__ == "__main__":
    main()
