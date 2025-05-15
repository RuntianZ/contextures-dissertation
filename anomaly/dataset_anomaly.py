import numpy as np
from framework.dataset import TabularDataset
from typing import Optional, Tuple, Union


class AnomalyDetectionDataset(TabularDataset):

    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        cat_idx: list,
        target_type: str,
        num_classes: int,
        num_features: Optional[int] = None,
        num_instances: Optional[int] = None,
        cat_dims: Optional[list] = None,
        split_indeces: Optional[list] = None,
        split_source: Optional[str] = None,
        la: Optional[Union[int, float]] = None,
        at_least_one_labeled: Optional[bool] = None,
    ) -> None:
        """
        la: ratio or the number of abnormal data in training set. test/val data must have la=None
        at_least_one_labeled: True if we should include at least one abnormal data in training set. test/val data must have la=None
        """
        super().__init__(name,
                         X,
                         y,
                         cat_idx,
                         target_type,
                         num_classes,
                         num_features,
                         num_instances,
                         cat_dims,
                         split_indeces,
                         split_source)
        assert len(set(list(y))) == 2
        assert target_type == "binary"
        # this is the case of val/test dataset, train dataset will be checked later
        if la is None:
            assert at_least_one_labeled is None, "dataset must have at_least_one_labeled = None"
        self.la = la
        self.at_least_one_labeled = at_least_one_labeled
        self.anomaly_rate = y.sum() / len(y) * 100







