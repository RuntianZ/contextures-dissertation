import ast
import pandas as pd
import numpy as np
from copy import deepcopy
from framework.base import StandaloneModule
from anomaly.dataset_anomaly import AnomalyDetectionDataset
from framework.utils import to_numpy

from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.lscp import LSCP
from pyod.models.mad import MAD
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.rod import ROD
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from pyod.models.iforest import IForest
from pyod.models.xgbod import XGBOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.deep_svdd import DeepSVDD

from anomaly.models.pyod_so_gaal import SO_GAAL



class PYOD_Base(StandaloneModule):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        """Implement this: self.model = ..."""
        raise NotImplementedError

    def init_module(self, dataset: AnomalyDetectionDataset) -> AnomalyDetectionDataset:
        model_params = self._filter_model_params()
        if self.seed >= 0:
            model_params.setdefault("random_state", self.seed)
        self.build_model(dataset, model_params)
        self.loadable_items = ['model']
        return dataset

    def fit(self, dataset: AnomalyDetectionDataset) -> None:
        # FIXME: look at ADBench get valid dataset, they use training data, very wield...
        self.logger.debug(f'Calling {self.model_name}.fit')
        if self.has_loaded:
            self.logger.debug('LOF has been loaded, skip fit')
        else:
            X = to_numpy(dataset.data)
            y = to_numpy(dataset.target)
            self.target_type = dataset.target_type
            if self.model_type == "unsupervised":
                self.model.fit(X)
            else:
                self.model.fit(X, y=y)

    def transform(self, dataset: AnomalyDetectionDataset) -> AnomalyDetectionDataset:
        X = to_numpy(dataset.data)
        dataset.pred = self.model.predict(X)
        dataset.pred_proba = self.model.predict_proba(X)
        dataset.pred_proba = dataset.pred_proba
        if len(dataset.pred_proba.shape) == 2:
            dataset.pred_proba = dataset.pred_proba[:, 1]
        # debug use only
        # print(f"Dataset anomaly rate {round(dataset.anomaly_rate, 2)}")
        # print(f"Prediction anomaly rate {round(dataset.pred.sum() / len(dataset.pred) * 100, 2)}")

        return dataset

    def decision_function(self, dataset: AnomalyDetectionDataset) -> AnomalyDetectionDataset:
        X = to_numpy(dataset.data)
        score = self.model.decision_function(X)
        dataset.outlier_score = score
        return dataset


class PYOD_LOF(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = LOF(**model_params)
        self.model_name = "PYOD_LOF"
        self.model_type = "unsupervised"


class PYOD_IForest(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        self.model = IForest(**model_params)
        self.model_name = "PYOD_IForest"
        self.model_type = "unsupervised"


class PYOD_OCSVM(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        model_params["tol"] = float(model_params["tol"])
        self.model = OCSVM(**model_params)
        self.model_name = "PYOD_OCSVM"
        self.model_type = "unsupervised"


class PYOD_ABOD(PYOD_Base):
    # TODO: for 4_ad_breastw, it throws warnings and an error of nan when `n_nieghbors` is small, set it to 30 fix this
    # TODO: for other datasets, it's okay to set `n_nieghbors` to 10 or less
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        # model_params["method"] = "default"
        self.model = ABOD(**model_params)
        self.model_name = "PYOD_ABOD"
        self.model_type = "unsupervised"

    def fit(self, dataset: AnomalyDetectionDataset) -> None:
        X = to_numpy(dataset.data)
        data1 = [list(r) for r in X.astype(float)]
        inputdf = pd.DataFrame(data=data1)
        self.logger.debug(f'Calling {self.model_name}.fit')
        if self.has_loaded:
            self.logger.debug('LOF has been loaded, skip fit')
        else:
            self.target_type = dataset.target_type
            self.model.fit(inputdf)

    def transform(self, dataset: AnomalyDetectionDataset) -> AnomalyDetectionDataset:
        X = to_numpy(dataset.data)
        data1 = [list(r) for r in X.astype(float)]
        inputdf = pd.DataFrame(data=data1)
        dataset.pred = self.model.predict(inputdf)
        dataset.pred_proba = self.model.predict_proba(inputdf)
        dataset.pred_proba = dataset.pred_proba
        if len(dataset.pred_proba.shape) == 2:
            dataset.pred_proba = dataset.pred_proba[:, 1]
        return dataset


class PYOD_CBLOF(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        self.model = CBLOF(**model_params)
        self.model_name = "PYOD_CBLOF"
        self.model_type = "unsupervised"


class PYOD_COF(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = COF(**model_params)
        self.model_name = "PYOD_COF"
        self.model_type = "unsupervised"


class PYOD_COPOD(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = COPOD(**model_params)
        self.model_name = "PYOD_COPOD"
        self.model_type = "unsupervised"


class PYOD_ECOD(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = ECOD(**model_params)
        self.model_name = "PYOD_ECOD"
        self.model_type = "unsupervised"


class PYOD_FeatureBagging(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        self.model = FeatureBagging(**model_params)
        self.model_name = "PYOD_FeatureBagging"
        self.model_type = "unsupervised"


class PYOD_HBOS(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = HBOS(**model_params)
        self.model_name = "PYOD_HBOS"
        self.model_type = "unsupervised"


class PYOD_KNN(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = KNN(**model_params)
        self.model_name = "PYOD_KNN"
        self.model_type = "unsupervised"


class PYOD_LMDD(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        self.model = LMDD(**model_params)
        self.model_name = "PYOD_LMDD"
        self.model_type = "unsupervised"


class PYOD_LODA(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = LODA(**model_params)
        self.model_name = "PYOD_LODA"
        self.model_type = "unsupervised"


class PYOD_LOCI(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        # TODO: super slow
        model_params.pop("random_state")
        self.model = LOCI(**model_params)
        self.model_name = "PYOD_LOCI"
        self.model_type = "unsupervised"


class PYOD_LSCP(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        # TODO: super slow
        model_params["detector_list"] = [LOF(), LOF()]  # TODO: may be make it more flexible later
        self.model = LSCP(**model_params)
        self.model_name = "PYOD_LSCP"
        self.model_type = "unsupervised"


class PYOD_MAD(PYOD_Base):
    # TODO: MAD algorithm is just for univariate data, can't use in our case, but put it here for future uses...
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = MAD(**model_params)
        self.model_name = "PYOD_MAD"
        self.model_type = "supervised"


class PYOD_MCD(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        self.model = MCD(**model_params)
        self.model_name = "PYOD_MCD"
        self.model_type = "unsupervised"


class PYOD_PCA(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        self.model = PCA(**model_params)
        self.model_name = "PYOD_PCA"
        self.model_type = "unsupervised"


class PYOD_ROD(PYOD_Base):
    # TODO: super slow on large dataset
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = ROD(**model_params)
        self.model_name = "PYOD_ROD"
        self.model_type = "unsupervised"


class PYOD_SOD(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = SOD(**model_params)
        self.model_name = "PYOD_SOD"
        self.model_type = "unsupervised"


class PYOD_SOS(PYOD_Base):
    # TODO: RuntimeWarning: overflow encountered in multiply, RuntimeWarning: All-NaN slice encountered
    # TODO: larger `eps` solve this issue, but very wield...
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        self.model = SOS(**model_params)
        self.model_name = "PYOD_SOS"
        self.model_type = "unsupervised"


class PYOD_AutoEncoder(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        if isinstance(model_params["hidden_neuron_list"], str):
            model_params["hidden_neuron_list"] = ast.literal_eval(model_params["hidden_neuron_list"])
        model_params["lr"] = float(model_params["lr"])
        self.model = AutoEncoder(**model_params)
        self.model_name = "PYOD_AutoEncoder"
        self.model_type = "unsupervised"

    def fit(self, dataset: AnomalyDetectionDataset) -> None:
        new_dataset = deepcopy(dataset)
        idx_n = np.where(new_dataset.y == 0)[0]
        new_dataset.X = new_dataset.X[idx_n]
        new_dataset.y = new_dataset.y[idx_n]
        super().fit(new_dataset)


class PYOD_VAE(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        if isinstance(model_params["encoder_neuron_list"], str):
            model_params["encoder_neuron_list"] = ast.literal_eval(model_params["encoder_neuron_list"])
        if isinstance(model_params["decoder_neuron_list"], str):
            model_params["decoder_neuron_list"] = ast.literal_eval(model_params["decoder_neuron_list"])
        model_params["lr"] = float(model_params["lr"])
        self.model = VAE(**model_params)
        self.model_name = "PYOD_VAE"
        self.model_type = "unsupervised"

    def fit(self, dataset: AnomalyDetectionDataset) -> None:
        new_dataset = deepcopy(dataset)
        idx_n = np.where(new_dataset.y == 0)[0]
        new_dataset.X = new_dataset.X[idx_n]
        new_dataset.y = new_dataset.y[idx_n]
        super().fit(new_dataset)


class PYOD_SO_GAAL(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        model_params["lr_d"] = float(model_params["lr_d"])
        model_params["lr_g"] = float(model_params["lr_g"])
        self.model = SO_GAAL(**model_params)
        self.model_name = "PYOD_SO_GAAL"
        self.model_type = "unsupervised"


class PYOD_MO_GAAL(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        model_params.pop("random_state")
        model_params["lr_d"] = float(model_params["lr_d"])
        model_params["lr_g"] = float(model_params["lr_g"])
        self.model = MO_GAAL(**model_params)
        self.model_name = "PYOD_MO_GAAL"
        self.model_type = "unsupervised"


class PYOD_XGBOD(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        self.model = XGBOD(**model_params)
        self.model_name = "PYOD_XGBOD"
        # FIXME: suppose to be semi-supervised, but `y` is not used
        self.model_type = "semi-unsupervised"


class PYOD_DeepSVDD(PYOD_Base):
    def build_model(self, dataset: AnomalyDetectionDataset, model_params: dict) -> None:
        assert model_params["n_features"] is None
        model_params["n_features"] = dataset.X.shape[1]
        if model_params["hidden_neurons"] is not None and isinstance(model_params["hidden_neurons"], str):
            model_params["hidden_neurons"] = ast.literal_eval(model_params["hidden_neurons"])
        self.model = DeepSVDD(**model_params)
        self.model_name = "PYOD_DeepSVDD"
        self.model_type = "unsupervised"


if __name__ == "__main__":
    print("OK")
