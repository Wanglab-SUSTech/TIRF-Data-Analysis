import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from detection import EventRecord


logger = logging.getLogger(__name__)


@dataclass
class ROIState:
    auto_events: List[EventRecord] = field(default_factory=list)
    manual_events: List[EventRecord] = field(default_factory=list)
    hidden_event_keys: Set[Tuple[int, int]] = field(default_factory=set)
    override: Dict[str, Any] = field(default_factory=dict)
    undo_stack: List[dict] = field(default_factory=list)


class AppState:
    def __init__(self):
        self.current_id_index = 0
        self.full_id_list = []
        self.filtered_id_list = []

        self.id_classifications: Dict[Any, int] = {}
        self.roi_states: Dict[Any, ROIState] = {}

        self.selected_event_keys: Set[Tuple[int, int]] = set()
        self.last_selected_event_key: Optional[Tuple[int, int]] = None

        self.navigation_filter_mode = "all"
        self.navigation_filter_class = None

        self.classifications = [1, 2, 3]

        self.classification_undo_stack: List[dict] = []
        self.classification_redo_stack: List[dict] = []

        self._dirty = False

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self):
        self._dirty = True

    def mark_clean(self):
        self._dirty = False

    def clear_event_selection(self):
        self.selected_event_keys.clear()
        self.last_selected_event_key = None

    def reset_for_new_data(self, id_list):
        self.current_id_index = 0
        self.full_id_list = list(id_list)
        self.filtered_id_list = list(id_list)

        self.id_classifications = {}
        self.roi_states = {}
        self.clear_event_selection()

        self.navigation_filter_mode = "all"
        self.navigation_filter_class = None
        self.classifications = [1, 2, 3]

        self.classification_undo_stack = []
        self.classification_redo_stack = []
        self.mark_clean()

    def ensure_roi_state(self, roi_id):
        if roi_id not in self.roi_states:
            self.roi_states[roi_id] = ROIState()
        return self.roi_states[roi_id]

    def get_current_id(self):
        if not self.filtered_id_list:
            return None
        if self.current_id_index < 0:
            self.current_id_index = 0
        if self.current_id_index >= len(self.filtered_id_list):
            self.current_id_index = len(self.filtered_id_list) - 1
        return self.filtered_id_list[self.current_id_index]

    def set_auto_events(self, roi_id, events):
        state = self.ensure_roi_state(roi_id)
        state.auto_events = list(events)

    def get_auto_events(self, roi_id):
        state = self.ensure_roi_state(roi_id)
        return list(state.auto_events)

    def set_manual_events(self, roi_id, events):
        state = self.ensure_roi_state(roi_id)
        state.manual_events = list(events)
        self.mark_dirty()

    def get_manual_events(self, roi_id):
        state = self.ensure_roi_state(roi_id)
        return list(state.manual_events)

    def add_manual_event(self, roi_id, event):
        state = self.ensure_roi_state(roi_id)
        state.manual_events.append(event)
        self.mark_dirty()

    def remove_manual_event_by_key(self, roi_id, event_key):
        state = self.ensure_roi_state(roi_id)
        before = len(state.manual_events)
        state.manual_events = [ev for ev in state.manual_events if ev.event_key != event_key]
        if len(state.manual_events) != before:
            self.mark_dirty()

    def replace_event_by_key(self, roi_id, old_event_key, new_event):
        state = self.ensure_roi_state(roi_id)

        replaced = False

        for index, ev in enumerate(state.manual_events):
            if ev.event_key == old_event_key:
                state.manual_events[index] = new_event
                replaced = True
                break

        if replaced:
            self.mark_dirty()
            return True

        for index, ev in enumerate(state.auto_events):
            if ev.event_key == old_event_key:
                state.auto_events[index] = new_event
                self.mark_dirty()
                return True

        return False

    def get_all_events(self, roi_id):
        state = self.ensure_roi_state(roi_id)
        return list(state.auto_events) + list(state.manual_events)

    def get_display_events(self, roi_id):
        state = self.ensure_roi_state(roi_id)
        events = list(state.auto_events) + list(state.manual_events)
        visible = [ev for ev in events if ev.event_key not in state.hidden_event_keys]
        visible.sort(key=lambda event: (event.start_idx, event.end_idx))
        return visible

    def hide_event_keys(self, roi_id, event_keys):
        state = self.ensure_roi_state(roi_id)
        before = len(state.hidden_event_keys)
        state.hidden_event_keys.update(event_keys)
        if len(state.hidden_event_keys) != before:
            self.mark_dirty()

    def unhide_event_keys(self, roi_id, event_keys):
        state = self.ensure_roi_state(roi_id)
        before = len(state.hidden_event_keys)
        for event_key in set(event_keys):
            state.hidden_event_keys.discard(event_key)
        if len(state.hidden_event_keys) != before:
            self.mark_dirty()

    def restore_hidden_events(self, roi_id):
        state = self.ensure_roi_state(roi_id)
        if state.hidden_event_keys:
            state.hidden_event_keys.clear()
            self.mark_dirty()

    def get_override(self, roi_id):
        state = self.ensure_roi_state(roi_id)
        return state.override

    def set_override(self, roi_id, override_dict):
        state = self.ensure_roi_state(roi_id)
        new_override = dict(override_dict)
        if state.override != new_override:
            state.override = new_override
            self.mark_dirty()

    def clear_override(self, roi_id):
        state = self.ensure_roi_state(roi_id)
        if state.override:
            state.override = {}
            self.mark_dirty()

    def push_undo(self, roi_id, action: dict):
        state = self.ensure_roi_state(roi_id)
        state.undo_stack.append(action)

    def pop_undo(self, roi_id):
        state = self.ensure_roi_state(roi_id)
        if not state.undo_stack:
            return None
        return state.undo_stack.pop()

    def clear_undo(self, roi_id):
        state = self.ensure_roi_state(roi_id)
        state.undo_stack.clear()

    def apply_navigation_filter(self):
        if self.navigation_filter_mode == "all":
            self.filtered_id_list = list(self.full_id_list)
        elif self.navigation_filter_mode == "unclassified":
            self.filtered_id_list = [
                roi_id for roi_id in self.full_id_list
                if roi_id not in self.id_classifications
            ]
        elif self.navigation_filter_mode == "class":
            target_class = self.navigation_filter_class
            self.filtered_id_list = [
                roi_id for roi_id in self.full_id_list
                if self.id_classifications.get(roi_id) == target_class
            ]
        else:
            self.filtered_id_list = list(self.full_id_list)

        if not self.filtered_id_list:
            self.current_id_index = 0
        else:
            self.current_id_index = min(self.current_id_index, len(self.filtered_id_list) - 1)

    def _set_classification_value(self, roi_id, class_num, mark_dirty=True):
        previous = self.id_classifications.get(roi_id)
        if class_num is None:
            if roi_id in self.id_classifications:
                del self.id_classifications[roi_id]
                if mark_dirty:
                    self.mark_dirty()
        else:
            if previous != class_num:
                self.id_classifications[roi_id] = class_num
                if mark_dirty:
                    self.mark_dirty()

    def classify(self, roi_id, class_num):
        self._set_classification_value(roi_id, class_num, mark_dirty=True)

    def unclassify(self, roi_id):
        self._set_classification_value(roi_id, None, mark_dirty=True)

    def _normalize_classification_changes(self, changes: Dict[Any, Dict[str, Optional[int]]]):
        normalized = {}
        for roi_id, change in (changes or {}).items():
            before = change.get("before")
            after = change.get("after")
            if before != after:
                normalized[roi_id] = {"before": before, "after": after}
        return normalized

    def apply_classification_changes(self, changes, record_history=False, label=""):
        normalized = self._normalize_classification_changes(changes)
        if not normalized:
            return False

        for roi_id, change in normalized.items():
            self._set_classification_value(roi_id, change["after"], mark_dirty=False)

        if record_history:
            self.classification_undo_stack.append({
                "label": label,
                "changes": normalized,
            })
            self.classification_redo_stack.clear()

        self.mark_dirty()
        return True

    def toggle_classification(self, roi_id, class_num):
        current = self.id_classifications.get(roi_id)
        new_class = None if current == class_num else class_num
        self.apply_classification_changes(
            {roi_id: {"before": current, "after": new_class}},
            record_history=True,
            label="toggle_classification"
        )
        return new_class

    def can_undo_classification(self):
        return bool(self.classification_undo_stack)

    def can_redo_classification(self):
        return bool(self.classification_redo_stack)

    def undo_classification(self):
        if not self.classification_undo_stack:
            return False

        action = self.classification_undo_stack.pop()
        for roi_id, change in action["changes"].items():
            self._set_classification_value(roi_id, change["before"], mark_dirty=False)
        self.classification_redo_stack.append(action)
        self.mark_dirty()
        return True

    def redo_classification(self):
        if not self.classification_redo_stack:
            return False

        action = self.classification_redo_stack.pop()
        for roi_id, change in action["changes"].items():
            self._set_classification_value(roi_id, change["after"], mark_dirty=False)
        self.classification_undo_stack.append(action)
        self.mark_dirty()
        return True

    def add_classification(self):
        new_class = max(self.classifications) + 1
        if new_class > 9:
            raise ValueError("A maximum of 9 classes is supported (mapped to number keys 1-9).")
        self.classifications.append(new_class)
        self.mark_dirty()
        return new_class

    def serialize_session_payload(self):
        roi_payload = {}
        for roi_id, roi_state in self.roi_states.items():
            if not roi_state.manual_events and not roi_state.hidden_event_keys and not roi_state.override:
                continue

            roi_payload[str(roi_id)] = {
                "manual_events": [event.to_dict() for event in roi_state.manual_events],
                "hidden_event_keys": [list(event_key) for event_key in sorted(roi_state.hidden_event_keys)],
                "override": dict(roi_state.override),
            }

        return {
            "id_classifications": {
                str(roi_id): class_num
                for roi_id, class_num in self.id_classifications.items()
            },
            "roi_states": roi_payload,
            "navigation_filter_mode": self.navigation_filter_mode,
            "navigation_filter_class": self.navigation_filter_class,
            "classifications": list(self.classifications),
        }

    def restore_session_payload(self, payload):
        id_lookup = {str(roi_id): roi_id for roi_id in self.full_id_list}

        self.id_classifications = {}
        for roi_text, class_num in payload.get("id_classifications", {}).items():
            roi_id = id_lookup.get(str(roi_text))
            if roi_id is not None and class_num is not None:
                self.id_classifications[roi_id] = int(class_num)

        self.roi_states = {}
        for roi_text, roi_payload in payload.get("roi_states", {}).items():
            roi_id = id_lookup.get(str(roi_text))
            if roi_id is None:
                continue

            state = self.ensure_roi_state(roi_id)
            state.manual_events = [
                self._event_record_from_dict(item)
                for item in roi_payload.get("manual_events", [])
            ]
            state.hidden_event_keys = {
                tuple(event_key) for event_key in roi_payload.get("hidden_event_keys", [])
            }
            state.override = dict(roi_payload.get("override", {}))
            state.undo_stack = []

        navigation_mode = payload.get("navigation_filter_mode", "all")
        self.navigation_filter_mode = navigation_mode if navigation_mode in {"all", "unclassified", "class"} else "all"

        navigation_class = payload.get("navigation_filter_class")
        self.navigation_filter_class = int(navigation_class) if navigation_class is not None else None

        raw_classes = payload.get("classifications", [1, 2, 3])
        normalized_classes = []
        for class_num in raw_classes:
            try:
                value = int(class_num)
            except (TypeError, ValueError):
                continue
            if 1 <= value <= 9 and value not in normalized_classes:
                normalized_classes.append(value)
        self.classifications = normalized_classes or [1, 2, 3]

        self.classification_undo_stack = []
        self.classification_redo_stack = []
        self.clear_event_selection()
        self.apply_navigation_filter()
        self.mark_dirty()

    @staticmethod
    def _event_record_from_dict(data):
        payload = dict(data)
        if "event_key" in payload and payload["event_key"] is not None:
            payload["event_key"] = tuple(payload["event_key"])
        return EventRecord(**payload)
