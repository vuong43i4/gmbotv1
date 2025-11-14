"""Reminder scheduler for AI speaker prompts.

This module provides a small reminder service that can parse simple Vietnamese
commands such as "nhắc nhở tôi đi ngủ lúc 1h". Reminders are stored on disk so
that they survive device restarts. When a reminder becomes due, a callback is
invoked so the caller can update a display and trigger speech synthesis.

The service is designed to be embedded into the existing XiaoZhi AI chatbot
software stack, but it can also run standalone for demonstration or unit tests.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable, List, Optional


@dataclasses.dataclass(order=True)
class Reminder:
    """Represents a scheduled reminder."""

    due_time: dt.datetime
    message: str = dataclasses.field(compare=False)
    identifier: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex, compare=False)
    created_at: dt.datetime = dataclasses.field(default_factory=dt.datetime.now, compare=False)

    def to_json(self) -> dict:
        return {
            "due_time": self.due_time.isoformat(),
            "message": self.message,
            "identifier": self.identifier,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "Reminder":
        return cls(
            due_time=dt.datetime.fromisoformat(payload["due_time"]),
            message=payload["message"],
            identifier=payload["identifier"],
            created_at=dt.datetime.fromisoformat(payload["created_at"]),
        )


class ReminderStore:
    """Persists reminders as JSON on disk."""

    def __init__(self, storage_path: Path):
        self._storage_path = storage_path
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load(self) -> List[Reminder]:
        with self._lock:
            if not self._storage_path.exists():
                return []
            data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        return [Reminder.from_json(item) for item in data]

    def save(self, reminders: Iterable[Reminder]) -> None:
        with self._lock:
            serialized = [reminder.to_json() for reminder in reminders]
            self._storage_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")


class ReminderScheduler:
    """Schedules reminders and invokes callbacks when they become due."""

    def __init__(
        self,
        store: ReminderStore,
        display_callback: Callable[[Reminder], None],
        speak_callback: Callable[[Reminder], None],
        poll_resolution: float = 1.0,
    ) -> None:
        self._store = store
        self._display_callback = display_callback
        self._speak_callback = speak_callback
        self._poll_resolution = poll_resolution
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._reminders: List[Reminder] = sorted(self._store.load())
        self._thread = threading.Thread(target=self._run_loop, name="reminder-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join()

    def add_reminder(self, reminder: Reminder) -> Reminder:
        with self._condition:
            self._reminders.append(reminder)
            self._reminders.sort()
            self._store.save(self._reminders)
            self._condition.notify_all()
        return reminder

    def upcoming(self) -> List[Reminder]:
        with self._condition:
            return list(self._reminders)

    def remove_reminder(self, identifier: str) -> Optional[Reminder]:
        with self._condition:
            for idx, reminder in enumerate(self._reminders):
                if reminder.identifier == identifier:
                    removed = self._reminders.pop(idx)
                    self._store.save(self._reminders)
                    self._condition.notify_all()
                    return removed
        return None

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            now = dt.datetime.now()
            due: Optional[Reminder] = None

            with self._condition:
                if self._reminders and self._reminders[0].due_time <= now:
                    due = self._reminders.pop(0)
                    self._store.save(self._reminders)
                else:
                    timeout = self._poll_resolution
                    if self._reminders:
                        delta = (self._reminders[0].due_time - now).total_seconds()
                        timeout = max(0.0, min(delta, self._poll_resolution))
                    self._condition.wait(timeout=timeout)
                    continue

            if due is not None:
                self._display_callback(due)
                self._speak_callback(due)


def parse_vietnamese_reminder(command: str, reference: Optional[dt.datetime] = None) -> Reminder:
    """Parse a simple Vietnamese reminder phrase.

    Parameters
    ----------
    command:
        A string such as "nhắc nhở tôi đi ngủ lúc 1h" or "nhắc nhở tôi uống nước lúc 13 giờ 30".
    reference:
        The reference time used when computing the due datetime. Defaults to the current
        local time.

    Returns
    -------
    Reminder
        A reminder instance with the parsed message and due time.

    Raises
    ------
    ValueError
        If the command cannot be parsed.
    """

    reference = reference or dt.datetime.now()
    normalized = command.lower().strip()

    if "nhắc nhở tôi" not in normalized:
        raise ValueError("Không tìm thấy cụm 'nhắc nhở tôi' trong câu lệnh.")
    colon_match = re.search(r"lúc\s*(\d{1,2})\s*[:]\s*(\d{1,2})", normalized)
    if colon_match:
        hour = int(colon_match.group(1))
        minute = int(colon_match.group(2))
        time_fragment = colon_match.group(0)
    else:
        simple_match = re.search(
            r"lúc\s*(\d{1,2})(?:\s*(?:giờ|h))?(?:\s*(\d{1,2}))?",
            normalized,
        )
        if not simple_match:
            raise ValueError("Không tìm thấy thời gian hợp lệ trong câu lệnh.")
        hour = int(simple_match.group(1))
        minute = int(simple_match.group(2) or 0)
        time_fragment = simple_match.group(0)

    if not 0 <= hour <= 23:
        raise ValueError("Giờ không hợp lệ.")
    if not 0 <= minute <= 59:
        raise ValueError("Phút không hợp lệ.")

    start_index = normalized.index("nhắc nhở tôi") + len("nhắc nhở tôi")
    message_original = command[start_index:]
    message_lower = message_original.lower()
    fragment_lower = time_fragment.lower()
    fragment_pos = message_lower.find(fragment_lower)
    if fragment_pos != -1:
        message_original = (
            message_original[:fragment_pos] + message_original[fragment_pos + len(time_fragment):]
        )

    message = re.sub(r"\blúc\b", "", message_original, flags=re.IGNORECASE).strip(" ,.:-")
    message = re.sub(r"\s{2,}", " ", message).strip()
    if message:
        message = message[0].upper() + message[1:]

    due = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due <= reference:
        due += dt.timedelta(days=1)

    if not message:
        message = "Nhắc nhở"

    return Reminder(due_time=due, message=message)


def default_display(reminder: Reminder) -> None:
    print(f"[DISPLAY] {reminder.message} - đến giờ rồi!")


def default_speak(reminder: Reminder) -> None:
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(reminder.message)
        engine.runAndWait()
    except Exception as exc:  # pragma: no cover - fallback path when pyttsx3 missing
        print(f"[SPEAK] {reminder.message} (không thể phát giọng nói: {exc})")


def demo() -> None:
    """Demonstrate scheduling a reminder 10 seconds in the future."""

    store = ReminderStore(Path("./data/reminders.json"))
    scheduler = ReminderScheduler(store, default_display, default_speak, poll_resolution=0.5)
    try:
        reminder = Reminder(due_time=dt.datetime.now() + dt.timedelta(seconds=10), message="Đi ngủ thôi!")
        scheduler.add_reminder(reminder)
        print("Đã thêm nhắc nhở. Chờ trong 10 giây để nghe thử...")
        time.sleep(12)
    finally:
        scheduler.stop()


if __name__ == "__main__":
    demo()
