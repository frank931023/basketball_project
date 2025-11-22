"""Player detection and tracking powered by RF-DETR and SAM2."""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import supervision as sv
import torch

from utils import read_stub, save_stub


DEFAULT_CONFIG: Dict[str, Any] = {
	"player_detection_model_id": None,
	"player_detection_model_path": None,
	"player_detection_confidence": 0.25,
	"player_detection_iou_threshold": 0.5,
	"player_class_ids": (0,),
	"number_class_id": 1,
	"sam2_config": "configs/sam2/config.yaml",
	"sam2_checkpoint": "checkpoints/sam2_checkpoint.pth",
	"sam2_repo_path": "../segment-anything-2-real-time",
}


def _prepare_config(raw_config: Optional[object]) -> Dict[str, Any]:
	config: Dict[str, Any] = dict(DEFAULT_CONFIG)

	if raw_config is not None:
		if isinstance(raw_config, dict):
			for key, value in raw_config.items():
				if value is not None:
					config[key] = value
		else:
			for key in DEFAULT_CONFIG:
				if hasattr(raw_config, key):
					value = getattr(raw_config, key)
					if value is not None:
						config[key] = value

	env = os.getenv
	config["player_detection_model_id"] = env(
		"PLAYER_DETECTION_MODEL_ID", config["player_detection_model_id"]
	)
	config["player_detection_model_path"] = env(
		"PLAYER_DETECTION_MODEL_PATH", config["player_detection_model_path"]
	)
	config["sam2_config"] = env("SAM2_CONFIG_PATH", config["sam2_config"])
	config["sam2_checkpoint"] = env("SAM2_CHECKPOINT_PATH", config["sam2_checkpoint"])
	config["sam2_repo_path"] = env("SAM2_REPO_PATH", config["sam2_repo_path"])

	player_class_ids = config.get("player_class_ids")
	if player_class_ids is None:
		config["player_class_ids"] = tuple()
	elif isinstance(player_class_ids, (list, tuple, set)):
		config["player_class_ids"] = tuple(player_class_ids)
	else:
		config["player_class_ids"] = (player_class_ids,)

	return config


class PlayerDetector:
	"""Handles player and jersey number detection using RF-DETR or a YOLO fallback."""

	def __init__(self, config: Optional[object] = None) -> None:
		self.config = _prepare_config(config)
		self.model = None
		self.model_kind: Optional[str] = None
		self._load_detector()

	def _load_detector(self) -> None:
		# Prefer the RF-DETR inference SDK when an ID is supplied.
		if self.config["player_detection_model_id"]:
			try:
				from inference import get_model  # type: ignore

				self.model = get_model(model_id=self.config["player_detection_model_id"])
				self.model_kind = "roboflow-inference"
				return
			except ImportError:
				warnings.warn(
					"The `inference` package is not installed. Falling back to a local YOLO model.",
					stacklevel=2,
				)
			except Exception as exc:  # pragma: no cover - defensive logging
				warnings.warn(f"Failed to initialise RF-DETR model: {exc}", stacklevel=2)

		if self.config["player_detection_model_path"]:
			try:
				from ultralytics import YOLO

				self.model = YOLO(self.config["player_detection_model_path"])
				self.model_kind = "ultralytics"
				return
			except Exception as exc:  # pragma: no cover - defensive logging
				warnings.warn(f"Failed to initialise YOLO fallback model: {exc}", stacklevel=2)

		raise RuntimeError(
			"Unable to initialise a player detection model. "
			"Set `PLAYER_DETECTION_MODEL_ID` for RF-DETR or provide a local model path."
		)

	def detect_players(self, frame: np.ndarray) -> sv.Detections:
		"""Detect players in a single frame and keep only player class IDs."""

		detections = self._run_inference(frame)
		if len(detections) == 0:
			return detections

		mask = np.isin(detections.class_id, list(self.config["player_class_ids"]))
		return detections[mask]

	def detect_numbers(self, frame: np.ndarray) -> sv.Detections:
		"""Detect jersey numbers in a single frame."""

		detections = self._run_inference(frame)
		if len(detections) == 0:
			return detections

		return detections[detections.class_id == self.config["number_class_id"]]

	def detect_all_objects(self, frame: np.ndarray) -> sv.Detections:
		"""Run detection without class filtering."""

		return self._run_inference(frame)

	def _run_inference(self, frame: np.ndarray) -> sv.Detections:
		if self.model_kind == "roboflow-inference":
			result = self.model.infer(
				frame,
				confidence=self.config["player_detection_confidence"],
				iou_threshold=self.config["player_detection_iou_threshold"],
			)[0]
			return sv.Detections.from_inference(result)

		if self.model_kind == "ultralytics":
			predictions = self.model.predict(
				frame,
				conf=self.config["player_detection_confidence"],
				iou=self.config["player_detection_iou_threshold"],
				verbose=False,
			)
			detection = predictions[0]
			return sv.Detections.from_ultralytics(detection)

		raise RuntimeError("Detector is not initialised correctly.")


class PlayerTracker:
	"""Handles player tracking using SAM2 with a ByteTrack fallback."""

	def __init__(
		self,
		model_path: Optional[str] = None,
		config: Optional[object] = None,
	) -> None:
		self.config = _prepare_config(config)
		if model_path:
			self.config["player_detection_model_path"] = model_path

		self.detector = PlayerDetector(self.config)
		self.byte_tracker = sv.ByteTrack()

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.autocast_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
		self.predictor = None
		self._next_tracker_id = 1

		self._initialize_sam2()

	def get_object_tracks(
		self,
		frames: List[np.ndarray],
		read_from_stub: bool = False,
		stub_path: Optional[str] = None,
	) -> List[Dict[int, Dict[str, List[float]]]]:
		cached_tracks = read_stub(read_from_stub, stub_path)
		if cached_tracks is not None and len(cached_tracks) == len(frames):
			return cached_tracks

		if not frames:
			return []

		if self.predictor is None:
			tracks = self._track_with_detection_only(frames)
		else:
			tracks = self._track_with_sam2(frames)

		save_stub(stub_path, tracks)
		return tracks

	def _track_with_detection_only(self, frames: List[np.ndarray]) -> List[Dict[int, Dict[str, List[float]]]]:
		tracks: List[Dict[int, Dict[str, List[float]]]] = []

		for frame in frames:
			detections = self.detector.detect_players(frame)
			tracked = self.byte_tracker.update_with_detections(detections)
			tracks.append(self._detections_to_track_dict(tracked))

		return tracks

	def _track_with_sam2(self, frames: List[np.ndarray]) -> List[Dict[int, Dict[str, List[float]]]]:
		tracks: List[Dict[int, Dict[str, List[float]]]] = []

		initial_detections = self.detector.detect_players(frames[0])
		if len(initial_detections) == 0:
			return self._track_with_detection_only(frames)

		tracker_ids = self.initialize_tracking(frames[0], initial_detections)
		if len(tracker_ids) == 0:
			return self._track_with_detection_only(frames)

		tracks.append(self._detections_to_track_dict(initial_detections))

		for frame_idx in range(1, len(frames)):
			frame = frames[frame_idx]
			tracked_detections = self.track_frame(frame)

			if len(tracked_detections) == 0:
				detections = self.detector.detect_players(frame)
				if len(detections) == 0:
					tracks.append({})
					continue

				tracker_ids = self.initialize_tracking(frame, detections)
				if len(tracker_ids) == 0:
					tracks.append({})
					continue
				tracks.append(self._detections_to_track_dict(detections))
				continue

			tracks.append(self._detections_to_track_dict(tracked_detections))

		return tracks

	def _initialize_sam2(self) -> None:
		try:
			sam2_path = Path(self.config["sam2_repo_path"]).resolve()
			if sam2_path.exists():
				import sys

				if str(sam2_path) not in sys.path:
					sys.path.insert(0, str(sam2_path))

			from sam2.build_sam import build_sam2_video_predictor

			config_path = Path(self.config["sam2_config"])
			checkpoint_path = Path(self.config["sam2_checkpoint"])

			if not config_path.is_absolute() and sam2_path.exists():
				config_path = (sam2_path / config_path).resolve()
			else:
				config_path = config_path.resolve()

			if not checkpoint_path.is_absolute() and sam2_path.exists():
				checkpoint_path = (sam2_path / checkpoint_path).resolve()
			else:
				checkpoint_path = checkpoint_path.resolve()

			if not config_path.exists() or not checkpoint_path.exists():
				missing: List[str] = []
				if not config_path.exists():
					missing.append(str(config_path))
				if not checkpoint_path.exists():
					missing.append(str(checkpoint_path))
				warnings.warn(
					"SAM2 resources are missing. Falling back to detection-only tracking: "
					+ ", ".join(missing),
					stacklevel=2,
				)
				return

			self.predictor = build_sam2_video_predictor(str(config_path), str(checkpoint_path))
		except ImportError:
			warnings.warn("SAM2 is not installed. Falling back to detection-only tracking.", stacklevel=2)
		except Exception as exc:  # pragma: no cover - defensive logging
			warnings.warn(f"SAM2 initialisation failed: {exc}", stacklevel=2)

	def initialize_tracking(self, frame: np.ndarray, detections: sv.Detections) -> List[int]:
		if self.predictor is None or len(detections) == 0:
			return []

		tracker_ids = list(range(self._next_tracker_id, self._next_tracker_id + len(detections)))
		self._next_tracker_id += len(detections)
		detections.tracker_id = tracker_ids

		with torch.inference_mode():
			with self._autocast():
				self.predictor.load_first_frame(frame)
				for xyxy, tracker_id in zip(detections.xyxy, tracker_ids):
					bbox = np.asarray([xyxy])
					self.predictor.add_new_prompt(frame_idx=0, obj_id=int(tracker_id), bbox=bbox)

		return tracker_ids

	def track_frame(self, frame: np.ndarray) -> sv.Detections:
		if self.predictor is None:
			return sv.Detections.empty()

		with torch.inference_mode():
			with self._autocast():
				tracker_ids, mask_logits = self.predictor.track(frame)

		ids = np.array(tracker_ids, dtype=int)
		masks = self._mask_logits_to_bool(mask_logits)
		if masks.size == 0:
			return sv.Detections.empty()

		if masks.ndim == 3 and masks.shape[0] > 0:
			filtered_masks = [
				self._filter_segments_by_distance(mask, distance_threshold=300.0)
				for mask in masks
			]
			masks = np.stack(filtered_masks)

		xyxy = sv.mask_to_xyxy(masks=masks)
		return sv.Detections(xyxy=xyxy, mask=masks, tracker_id=ids)

	def _mask_logits_to_bool(self, mask_logits: torch.Tensor) -> np.ndarray:
		def to_numpy(logit) -> np.ndarray:
			if isinstance(logit, torch.Tensor):
				return logit.detach().cpu().numpy()
			return np.asarray(logit)

		if isinstance(mask_logits, torch.Tensor):
			masks = to_numpy(mask_logits > 0.0)
		else:
			masks = np.asarray([to_numpy(logit > 0.0) for logit in mask_logits])

		if masks.ndim == 4:
			masks = np.squeeze(masks, axis=1)
		if masks.ndim == 2:
			masks = masks[np.newaxis, ...]

		return masks.astype(bool)

	def _filter_segments_by_distance(self, mask: np.ndarray, distance_threshold: float = 300.0) -> np.ndarray:
		mask_uint8 = mask.astype(np.uint8)
		num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

		if num_labels <= 1:
			return mask.copy()

		main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
		main_centroid = centroids[main_label]
		filtered_mask = np.zeros_like(mask, dtype=bool)

		for label in range(1, num_labels):
			centroid = centroids[label]
			dist = np.linalg.norm(centroid - main_centroid)
			if label == main_label or dist <= distance_threshold:
				filtered_mask |= labels == label

		return filtered_mask

	def _detections_to_track_dict(self, detections: sv.Detections) -> Dict[int, Dict[str, List[float]]]:
		frame_tracks: Dict[int, Dict[str, List[float]]] = {}

		for idx, xyxy in enumerate(detections.xyxy):
			ids = detections.tracker_id
			tracker_id = int(ids[idx]) if ids is not None and len(ids) > idx else idx + 1
			frame_tracks[tracker_id] = {"bbox": xyxy.tolist()}

		return frame_tracks

	@contextmanager
	def _autocast(self):
		if self.device != "cuda":
			yield
			return

		with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
			yield

