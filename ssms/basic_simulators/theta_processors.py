from typing import Dict, Any
from abc import ABC, abstractmethod
import numpy as np


class AbstractThetaProcessor(ABC):
    @abstractmethod
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass


class DefaultThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        return theta


class DSConflictDriftThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["v"] = np.zeros(theta["a"].shape)
        return theta


class DDMSTThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z_dist"] = model_config["simulator_fixed_params"]["z_dist"]
        theta["v_dist"] = model_config["simulator_fixed_params"]["v_dist"]
        theta["t_dist"] = model_config["simulator_param_mappings"]["t_dist"](
            theta["st"]
        )
        return theta


class DDMRayleighThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z_dist"] = model_config["simulator_fixed_params"]["z_dist"]
        theta["v_dist"] = model_config["simulator_fixed_params"]["v_dist"]
        theta["t_dist"] = model_config["simulator_fixed_params"]["t_dist"]
        theta["t"] = (
            np.ones(theta["a"].shape) * model_config["simulator_fixed_params"]["t"]
        )
        return theta


class DDMTruncnormtThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z_dist"] = model_config["simulator_fixed_params"]["z_dist"]
        theta["v_dist"] = model_config["simulator_fixed_params"]["v_dist"]
        theta["t_dist"] = model_config["simulator_param_mappings"]["t_dist"](
            theta["mt"], theta["st"]
        )
        theta["t"] = (
            np.ones(theta["a"].shape) * model_config["simulator_fixed_params"]["t"]
        )
        return theta


class DDMSDVThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z_dist"] = model_config["simulator_fixed_params"]["z_dist"]
        theta["t_dist"] = model_config["simulator_fixed_params"]["t_dist"]
        theta["v_dist"] = model_config["simulator_param_mappings"]["v_dist"](
            theta["sv"]
        )
        return theta


class FullDDMRVThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z_dist"] = model_config["simulator_param_mappings"]["z_dist"](
            theta["sz"]
        )
        theta["t_dist"] = model_config["simulator_param_mappings"]["t_dist"](
            theta["st"]
        )
        theta["v_dist"] = model_config["simulator_param_mappings"]["v_dist"](
            theta["sv"]
        )
        return theta


class ShrinkSpotThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["v"] = np.tile(np.array([0], dtype=np.float32), theta["a"].shape)
        return theta


class LBA2ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
        theta["z"] = np.expand_dims(theta["A"], axis=1)
        theta["a"] = np.expand_dims(theta["b"], axis=1)
        # TODO: The nact parameter seems redundant, should
        # probably just get rid of it at cssm level.
        theta["nact"] = 2

        if "A" in theta:
            del theta["A"]
        if "b" in theta:
            del theta["b"]
        return theta


class LBA3ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["z"] = np.expand_dims(theta["A"], axis=1)
        theta["a"] = np.expand_dims(theta["b"], axis=1)
        # TODO: The nact parameter seems redundant, should
        # probably just get rid of it at cssm level.
        theta["nact"] = 3
        if "A" in theta:
            del theta["A"]
        if "b" in theta:
            del theta["b"]
        return theta


class LBA3V1ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["z"] = np.expand_dims(theta["z"], axis=1)
        theta["nact"] = 3
        return theta


class LBA3AngleV1ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["z"] = np.expand_dims(theta["z"], axis=1)
        theta["theta"] = np.expand_dims(theta["theta"], axis=1)
        theta["nact"] = 3
        return theta


class RLWMLBA3V1ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["v_RL"] = np.column_stack(
            [theta["v_RL_0"], theta["v_RL_1"], theta["v_RL_2"]]
        )
        theta["v_WM"] = np.column_stack(
            [theta["v_WM_0"], theta["v_WM_1"], theta["v_WM_2"]]
        )
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        theta["z"] = np.expand_dims(theta["z"], axis=1)
        theta["theta"] = np.expand_dims(theta["theta"], axis=1)
        theta["nact"] = 3
        return theta


class Race2ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z"] = np.column_stack([theta["z0"], theta["z1"]])
        theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        return theta


class Race3ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z"] = np.column_stack([theta["z0"], theta["z1"], theta["z2"]])
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        return theta


class RaceNoBias2ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z"] = np.column_stack([theta["z"], theta["z"]])
        theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        return theta


class RaceNoZ2ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z"] = np.tile(
            np.array([0.0] * 2, dtype=np.float32), (theta["n_trials"], 1)
        )
        theta["v"] = np.column_stack([theta["v0"], theta["v1"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        return theta


class RaceNoBias3ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z"] = np.column_stack([theta["z"], theta["z"], theta["z"]])
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        return theta


class RaceNoZ3ThetaProcessor(AbstractThetaProcessor):
    def process_theta(
        self, theta: Dict[str, Any], model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        theta["z"] = np.tile(
            np.array([0.0] * 3, dtype=np.float32), (theta["n_trials"], 1)
        )
        theta["v"] = np.column_stack([theta["v0"], theta["v1"], theta["v2"]])
        theta["t"] = np.expand_dims(theta["t"], axis=1)
        theta["a"] = np.expand_dims(theta["a"], axis=1)
        return theta
