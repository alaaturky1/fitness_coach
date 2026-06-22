from __future__ import annotations

from app.core.models import Language
from app.feedback.llm import FeedbackContext, get_llm_feedback_client


# Stored as unicode escape sequences to avoid editor/encoding issues on Windows.
ISSUE_TO_TEXT_ESCAPED: dict[str, dict[str, str]] = {
    "visibility_low": {
        "en": "Make sure your full body is visible to the camera.",
        "ar": "\\u062e\\u0644\\u064a \\u062c\\u0633\\u0645\\u0643 \\u0643\\u0644\\u0647 \\u0648\\u0627\\u0636\\u062d \\u0642\\u062f\\u0627\\u0645 \\u0627\\u0644\\u0643\\u0627\\u0645\\u064a\\u0631\\u0627.",
    },
    "excessive_forward_lean": {
        "en": "Chest up, keep your torso more upright.",
        "ar": "\\u0627\\u0631\\u0641\\u0639 \\u0635\\u062f\\u0631\\u0643\\u060c \\u0648\\u062e\\u0644\\u064a \\u0638\\u0647\\u0631\\u0643 \\u0645\\u0633\\u062a\\u0642\\u064a\\u0645 \\u0623\\u0643\\u062a\\u0631.",
    },
    "knee_valgus_left": {
        "en": "Track your left knee over your toes.",
        "ar": "\\u062e\\u0644\\u064a \\u0631\\u0643\\u0628\\u062a\\u0643 \\u0627\\u0644\\u0634\\u0645\\u0627\\u0644 \\u0641\\u064a \\u0627\\u062a\\u062c\\u0627\\u0647 \\u0635\\u0648\\u0627\\u0628\\u0639 \\u0631\\u062c\\u0644\\u0643.",
    },
    "knee_valgus_right": {
        "en": "Track your right knee over your toes.",
        "ar": "\\u062e\\u0644\\u064a \\u0631\\u0643\\u0628\\u062a\\u0643 \\u0627\\u0644\\u064a\\u0645\\u064a\\u0646 \\u0641\\u064a \\u0627\\u062a\\u062c\\u0627\\u0647 \\u0635\\u0648\\u0627\\u0628\\u0639 \\u0631\\u062c\\u0644\\u0643.",
    },
    "hips_sagging": {
        "en": "Tighten your core—don’t let your hips drop.",
        "ar": "\\u0634\\u062f \\u0628\\u0637\\u0646\\u0643. \\u0645\\u0627 \\u062a\\u062e\\u0644\\u064a\\u0634 \\u0627\\u0644\\u062d\\u0648\\u0636 \\u064a\\u0647\\u0628\\u0637.",
    },
    "hips_off_line": {
        "en": "Keep your body in a straight line.",
        "ar": "\\u062e\\u0644\\u064a \\u062c\\u0633\\u0645\\u0643 \\u0639\\u0644\\u0649 \\u062e\\u0637 \\u0645\\u0633\\u062a\\u0642\\u064a\\u0645.",
    },
    "shallow_depth": {
        "en": "Go a bit deeper while staying controlled.",
        "ar": "\\u0627\\u0646\\u0632\\u0644 \\u0623\\u0639\\u0645\\u0642 \\u0634\\u0648\\u064a\\u0629\\u060c \\u0648\\u0627\\u0637\\u0644\\u0639 \\u0628\\u062a\\u062d\\u0643\\u0645.",
    },
    "unknown_exercise": {
        "en": "Tell me the exercise type to coach you better.",
        "ar": "\\u0627\\u062e\\u062a\\u0627\\u0631 \\u0646\\u0648\\u0639 \\u0627\\u0644\\u062a\\u0645\\u0631\\u064a\\u0646\\u060c \\u0648\\u0623\\u0646\\u0627 \\u0623\\u062f\\u064a\\u0643 \\u062a\\u0648\\u062c\\u064a\\u0647 \\u0623\\u062f\\u0642.",
    },
    "pose_detection_failed": {
        "en": "Hold still and make your body clear in the frame.",
        "ar": "\\u0627\\u062b\\u0628\\u062a \\u0644\\u062d\\u0638\\u0629. \\u0648\\u062e\\u0644\\u064a \\u062c\\u0633\\u0645\\u0643 \\u0648\\u0627\\u0636\\u062d \\u0641\\u064a \\u0627\\u0644\\u0635\\u0648\\u0631\\u0629.",
    },
    "pose_detection_error": {
        "en": "Reset your position so I can read your movement clearly.",
        "ar": "\\u0638\\u0628\\u0637 \\u0645\\u0643\\u0627\\u0646\\u0643 \\u062a\\u0627\\u0646\\u064a \\u0639\\u0634\\u0627\\u0646 \\u0623\\u0642\\u0631\\u0623 \\u0627\\u0644\\u062d\\u0631\\u0643\\u0629 \\u0628\\u0648\\u0636\\u0648\\u062d.",
    },
    "pose_detector_unavailable": {
        "en": "Pose detector is offline on the server.",
        "ar": "\\u0642\\u0631\\u0627\\u0621\\u0629 \\u0627\\u0644\\u062d\\u0631\\u0643\\u0629 \\u0645\\u0634 \\u0645\\u062a\\u0627\\u062d\\u0629 \\u062f\\u0644\\u0648\\u0642\\u062a\\u064a.",
    },
    "pose_image_decode_failed": {
        "en": "The camera frame could not be read. Restart the camera and try again.",
        "ar": "\\u0635\\u0648\\u0631\\u0629 \\u0627\\u0644\\u0643\\u0627\\u0645\\u064a\\u0631\\u0627 \\u0645\\u0634 \\u0648\\u0627\\u0636\\u062d\\u0629. \\u0627\\u0641\\u062a\\u062d \\u0627\\u0644\\u0643\\u0627\\u0645\\u064a\\u0631\\u0627 \\u062a\\u0627\\u0646\\u064a \\u0648\\u062c\\u0631\\u0628.",
    },
    "pose_not_detected": {
        "en": "Step back so your full body is visible.",
        "ar": "\\u0627\\u0631\\u062c\\u0639 \\u062e\\u0637\\u0648\\u0629 \\u0644\\u0648\\u0631\\u0627 \\u0639\\u0634\\u0627\\u0646 \\u062c\\u0633\\u0645\\u0643 \\u0643\\u0644\\u0647 \\u064a\\u0628\\u0627\\u0646.",
    },
    "lower_body_not_visible": {
        "en": "Step back until your hips, knees, and ankles are visible.",
        "ar": "\\u0627\\u0631\\u062c\\u0639 \\u0644\\u0648\\u0631\\u0627 \\u0644\\u062d\\u062f \\u0645\\u0627 \\u0627\\u0644\\u062d\\u0648\\u0636 \\u0648\\u0627\\u0644\\u0631\\u0643\\u0628 \\u0648\\u0627\\u0644\\u0643\\u0627\\u062d\\u0644 \\u064a\\u0628\\u0627\\u0646\\u0648\\u0627.",
    },
    "hips_not_visible": {
        "en": "Move the camera back until your hips are visible.",
        "ar": "\\u0627\\u0628\\u0639\\u062f \\u0627\\u0644\\u0645\\u0648\\u0628\\u0627\\u064a\\u0644 \\u0644\\u062d\\u062f \\u0645\\u0627 \\u0627\\u0644\\u062d\\u0648\\u0636 \\u064a\\u0628\\u0627\\u0646.",
    },
    "shoulders_not_visible": {
        "en": "Keep your shoulders inside the frame.",
        "ar": "\\u062e\\u0644\\u064a \\u0643\\u062a\\u0627\\u0641\\u0643 \\u062c\\u0648\\u0647 \\u0627\\u0644\\u0635\\u0648\\u0631\\u0629.",
    },
    "landmarks_low_confidence": {
        "en": "Improve lighting and keep your body steady in the frame.",
        "ar": "\\u0632\\u0648\\u062f \\u0627\\u0644\\u0625\\u0636\\u0627\\u0621\\u0629\\u060c \\u0648\\u0627\\u062b\\u0628\\u062a \\u062c\\u0633\\u0645\\u0643 \\u062c\\u0648\\u0647 \\u0627\\u0644\\u0635\\u0648\\u0631\\u0629.",
    },
}


DEFAULT_POSITIVE_ESCAPED = {
    "en": "Good rep. Keep it controlled.",
    "ar": "\\u062a\\u0645\\u0627\\u0645. \\u0643\\u0645\\u0644 \\u0628\\u0646\\u0641\\u0633 \\u0627\\u0644\\u062a\\u062d\\u0643\\u0645.",
}

COMBINED_ISSUE_TEXT_ESCAPED: dict[tuple[str, str], dict[str, str]] = {
    ("shallow_depth", "excessive_forward_lean"): {
        "en": "Go deeper with your hips while keeping your chest up.",
        "ar": "\\u0627\\u0646\\u0632\\u0644 \\u0623\\u0639\\u0645\\u0642\\u060c \\u0648\\u0627\\u0631\\u0641\\u0639 \\u0635\\u062f\\u0631\\u0643.",
    },
    ("shallow_depth", "hips_sagging"): {
        "en": "Reach full depth and keep your core tight.",
        "ar": "\\u0648\\u0635\\u0644 \\u0644\\u0644\\u0639\\u0645\\u0642 \\u0627\\u0644\\u0643\\u0627\\u0645\\u0644\\u060c \\u0648\\u0634\\u062f \\u0628\\u0637\\u0646\\u0643.",
    },
}

ISSUE_ALIASES: dict[str, str] = {
    "knee_valgus_left": "knee_valgus",
    "knee_valgus_right": "knee_valgus",
}

ALIAS_TEXT_ESCAPED: dict[str, dict[str, str]] = {
    "knee_valgus": {
        "en": "Keep your knees tracking over your toes.",
        "ar": "\\u062e\\u0644\\u064a \\u0631\\u0643\\u0628\\u062a\\u0643 \\u0641\\u064a \\u0627\\u062a\\u062c\\u0627\\u0647 \\u0635\\u0648\\u0627\\u0628\\u0639 \\u0631\\u062c\\u0644\\u0643.",
    }
}


def _unescape(s: str) -> str:
    # NOTE: s is ASCII; decode unicode escapes to real Unicode.
    return s.encode("utf-8").decode("unicode_escape")


def _msg(lang: str, table: dict[str, str]) -> str:
    return _unescape(table.get(lang, table["en"]))


def _normalize_issues(issues: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        key = ISSUE_ALIASES.get(issue, issue)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def rule_based_feedback(language: Language, issues: list[str]) -> str:
    lang = language.value
    normalized = _normalize_issues(issues)

    if len(normalized) >= 2:
        combo_key = tuple(normalized[:2])
        combo = COMBINED_ISSUE_TEXT_ESCAPED.get(combo_key)
        if combo:
            return _msg(lang, combo)

    for issue in normalized:
        alias_msg = ALIAS_TEXT_ESCAPED.get(issue)
        if alias_msg:
            return _msg(lang, alias_msg)
        txt = ISSUE_TO_TEXT_ESCAPED.get(issue)
        if txt:
            return _msg(lang, txt)
    return _msg(lang, DEFAULT_POSITIVE_ESCAPED)


def pick_feedback(
    language: Language,
    issues: list[str],
    *,
    level: str = "beginner",
    exercise: str | None = None,
    score: float | None = None,
    priority: str = "low",
    paused: bool = False,
) -> str:
    normalized = _normalize_issues(issues)
    fallback = rule_based_feedback(language, normalized)
    context = FeedbackContext(
        language=language,
        level=level,
        exercise=exercise,
        issues=tuple(normalized),
        fallback=fallback,
        score=score,
        priority=priority,
        paused=paused,
    )
    return get_llm_feedback_client().generate(context)
