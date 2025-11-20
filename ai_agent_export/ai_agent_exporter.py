from __future__ import annotations
import json
from typing import Any, Dict, Iterable, List, Optional, Union

Number = Union[int, float]


class AiAgentExporter:
    """Exporter for building per-frame summaries suitable for an AI agent
    and for serializing those summaries to JSON.

    This class mirrors the previous module-level API but provides a class
    wrapper to make usage and dependency injection easier.
    """

    AI_AGENT_FRAME_SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Basketball per-frame packet for AI agent",
        "type": "object",
        "properties": {
            "frame_idx": {"type": "integer"},
            "timestamp": {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "null"}]},
            "tactical_positions": {
                "type": "object",
                "description": "Mapping player_id -> [x,y] in tactical view"
            },
            "ball_pos": {"anyOf": [{"type": "array", "items": [{"type":"number"},{"type":"number"}]}, {"type":"null"}]},
            "possession": {"anyOf": [{"type":"integer"}, {"type":"null"}]},
            "team_ids": {"type": "object", "description": "player_id -> team_id"},
            "speeds": {"type": "object", "description": "player_id -> speed_kmh"},
            "recent_events": {"type": "array", "description": "List of recent events (passes/interceptions)"},
            "court_keypoints": {"anyOf": [{"type":"object"}, {"type":"null"}]},
        },
        "required": ["frame_idx"],
    }

    def __init__(self) -> None:
        pass

    def _to_python(self, val: Any):
        """Recursively convert numpy types and other array-like objects to
        plain Python types suitable for JSON serialization.
        """
        try:
            # numpy arrays / pandas objects often expose tolist
            if hasattr(val, "tolist"):
                return self._to_python(val.tolist())
        except Exception:
            pass

        # ultralytics Keypoints and similar objects often expose `.xy` or `.xyn`
        # attributes or support `.cpu().numpy()`; try those first for conversion.
        try:
            # attribute .xy (supervision / ultralytics Keypoints)
            if hasattr(val, "xy"):
                xy = getattr(val, "xy")
                # xy may be a tensor/array or an object with tolist
                if hasattr(xy, "tolist"):
                    return self._to_python(xy.tolist())
                # try .cpu().numpy()
                if hasattr(xy, "cpu") and hasattr(xy, "numpy"):
                    try:
                        return self._to_python(xy.cpu().numpy().tolist())
                    except Exception:
                        pass
                return self._to_python(xy)
            if hasattr(val, "xyn"):
                xyn = getattr(val, "xyn")
                if hasattr(xyn, "tolist"):
                    return self._to_python(xyn.tolist())
                if hasattr(xyn, "cpu") and hasattr(xyn, "numpy"):
                    try:
                        return self._to_python(xyn.cpu().numpy().tolist())
                    except Exception:
                        pass
                return self._to_python(xyn)
        except Exception:
            pass

        if isinstance(val, dict):
            return {str(k): self._to_python(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [self._to_python(x) for x in val]
        # numpy scalar
        try:
            import numpy as _np

            if isinstance(val, _np.generic):
                return val.item()
        except Exception:
            pass

        # torch tensors or other tensors: try cpu().numpy()
        try:
            # avoid importing torch globally; check for cpu and numpy methods
            if hasattr(val, "cpu") and hasattr(val, "numpy"):
                try:
                    arr = val.cpu().numpy()
                    if hasattr(arr, "tolist"):
                        return self._to_python(arr.tolist())
                except Exception:
                    pass
        except Exception:
            pass

        # fallback: attempt to convert numbers or keep as-is
        return val

    def _get_value_for_frame(self, data: Any, frame_idx: int, default=None):
        """Return the value for a given frame index. Accepts list-like or dict-like
        `data`. If `data` is None, returns `default`.
        """
        if data is None:
            return default
        # dict-like
        if isinstance(data, dict):
            return data.get(frame_idx, default)
        # list/tuple-like: index -> frame
        if isinstance(data, (list, tuple)):
            if 0 <= frame_idx < len(data):
                return data[frame_idx]
            return default
        # single scalar value for all frames
        return data

    def _extract_ball_center_from_frame_ball_tracks(self, frame_ball_tracks: Any):
        """Try to extract a single (x,y) position for the ball from a frame's
        ball_tracks entry. Accepts common shapes: dict with 'bbox', list of bboxes,
        or a (x,y) tuple. Returns None if not found.
        """
        if frame_ball_tracks is None:
            return None
        # If already a point
        if isinstance(frame_ball_tracks, (list, tuple)) and len(frame_ball_tracks) == 2 and all(
            isinstance(x, (int, float)) for x in frame_ball_tracks
        ):
            return [float(frame_ball_tracks[0]), float(frame_ball_tracks[1])]

        # If dict that has a center
        if isinstance(frame_ball_tracks, dict):
            # common patterns: {'bbox': [x1,y1,x2,y2]} or {'center':[x,y]}
            if "center" in frame_ball_tracks:
                c = frame_ball_tracks["center"]
                return [float(c[0]), float(c[1])]
            if "bbox" in frame_ball_tracks:
                x1, y1, x2, y2 = frame_ball_tracks["bbox"]
                return [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]

        # If list of tracks, take first bbox-like entry
        if isinstance(frame_ball_tracks, (list, tuple)) and len(frame_ball_tracks) > 0:
            first = frame_ball_tracks[0]
            return self._extract_ball_center_from_frame_ball_tracks(first)

        return None

    def build_summary_per_frame(
        self,
        frames: Optional[Iterable[int]] = None,
        timestamps: Optional[Iterable[Number]] = None,
        tactical_player_positions: Optional[Union[List, Dict]] = None,
        ball_positions: Optional[Union[List, Dict]] = None,
        ball_tracks: Optional[Union[List, Dict]] = None,
        ball_aquisition: Optional[Union[List, Dict]] = None,
        player_assignment: Optional[Union[List, Dict]] = None,
        player_speeds: Optional[Union[List, Dict]] = None,
        recent_events: Optional[Union[List, Dict]] = None,
        court_keypoints: Optional[Union[List, Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """Build a list of per-frame minimal packets. Inputs may be lists
        (indexed by frame) or dicts keyed by frame index.
        """
        # Determine frame indices
        frame_set = set()
        if frames is not None:
            frame_set.update(list(frames))

        # helper to add indices from inputs
        def _add_indices_from(data: Any):
            if data is None:
                return
            if isinstance(data, dict):
                frame_set.update([int(k) for k in data.keys()])
            elif isinstance(data, (list, tuple)):
                frame_set.update(range(len(data)))

        for d in (
            tactical_player_positions,
            ball_positions,
            ball_tracks,
            ball_aquisition,
            player_assignment,
            player_speeds,
            recent_events,
            court_keypoints,
        ):
            _add_indices_from(d)

        if len(frame_set) == 0:
            # nothing provided
            return []

        frames_sorted = sorted(frame_set)

        # If timestamps provided as list/dict, normalize
        ts_map = None
        if timestamps is not None:
            if isinstance(timestamps, dict):
                ts_map = timestamps
            elif isinstance(timestamps, (list, tuple)):
                ts_map = {i: timestamps[i] for i in range(len(timestamps))}

        summary: List[Dict[str, Any]] = []
        for f in frames_sorted:
            ts = ts_map.get(f) if ts_map is not None else None

            tactical_pos = self._get_value_for_frame(tactical_player_positions, f, default={})
            # player assignment/team ids
            team_ids = self._get_value_for_frame(player_assignment, f, default={})
            # speeds
            speeds = self._get_value_for_frame(player_speeds, f, default={})
            # ball acquisition (possession)
            possession = self._get_value_for_frame(ball_aquisition, f, default=None)

            # ball position: prefer explicit ball_positions, else try ball_tracks
            bp = self._get_value_for_frame(ball_positions, f, default=None)
            if bp is None:
                bt = self._get_value_for_frame(ball_tracks, f, default=None)
                bp = self._extract_ball_center_from_frame_ball_tracks(bt)

            events = self._get_value_for_frame(recent_events, f, default=[])
            ck = self._get_value_for_frame(court_keypoints, f, default=None)

            frame_packet: Dict[str, Any] = {
                "frame_idx": int(f),
                "timestamp": self._to_python(ts) if ts is not None else None,
                "tactical_positions": self._to_python(tactical_pos) if tactical_pos is not None else {},
                "ball_pos": self._to_python(bp) if bp is not None else None,
                "possession": self._to_python(possession) if possession is not None else None,
                "team_ids": self._to_python(team_ids) if team_ids is not None else {},
                "speeds": self._to_python(speeds) if speeds is not None else {},
                "recent_events": self._to_python(events) if events is not None else [],
                "court_keypoints": self._to_python(ck) if ck is not None else None,
            }

            summary.append(frame_packet)

        return summary

    def export_summary_json(self, path: str, summary: List[Dict[str, Any]], indent: int = 2) -> None:
        """Write the summary list to `path` as JSON. This is a thin wrapper that
        ensures values are JSON-serializable.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._to_python(summary), f, ensure_ascii=False, indent=indent)


if __name__ == "__main__":
    # Quick example that does not run project code — shows how to call exporter.
    exporter = AiAgentExporter()
    example = exporter.build_summary_per_frame(
        frames=range(3),
        timestamps=[0.0, 0.04, 0.08],
        tactical_player_positions=[{1: [1.0, 2.0]}, {1: [1.1, 2.0]}, {1: [1.2, 2.1]}],
        ball_positions=[None, [100, 120], [102, 118]],
        ball_aquisition=[-1, 1, 1],
        player_assignment=[{1: 0}, {1: 0}, {1: 0}],
        player_speeds=[{1: 5.2}, {1: 5.3}, {1: 5.1}],
        recent_events=[[], [{"type": "pass", "from": 1, "to": 2}], []],
    )
    print("Example summary (3 frames):", json.dumps(example, indent=2))
