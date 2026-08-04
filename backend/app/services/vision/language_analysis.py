from __future__ import annotations

from collections import defaultdict

from app.core.config import settings
from app.models.vision import OCRWord, VisionLanguageAnalysis

SCRIPT_NAMES = {
    "Latin": "English or Latin-script text",
    "Devanagari": "Devanagari-script text",
    "Bengali": "Bengali-script text",
    "Arabic": "Arabic-script text",
    "Cyrillic": "Cyrillic-script text",
    "CJK": "CJK text",
    "Greek": "Greek-script text",
    "Tamil": "Tamil-script text",
    "Telugu": "Telugu-script text",
    "Kannada": "Kannada-script text",
    "Malayalam": "Malayalam-script text",
}


class VisionLanguageAnalysisService:
    def analyze(self, words: list[OCRWord]) -> VisionLanguageAnalysis:
        retained: list[OCRWord] = []
        script_counts: dict[str, int] = defaultdict(int)
        confidence_weight = 0.0
        confidence_characters = 0

        for word in words:
            if word.confidence < settings.ocr_low_confidence_threshold:
                continue
            letters = [character for character in word.text if character.isalpha()]
            if len(letters) < 2:
                continue
            token_counts: dict[str, int] = defaultdict(int)
            for character in letters:
                script = self._script(character)
                if script:
                    token_counts[script] += 1
            valid_count = sum(token_counts.values())
            if valid_count < 2:
                continue
            retained.append(word)
            for script, count in token_counts.items():
                script_counts[script] += count
            confidence_weight += word.confidence * valid_count
            confidence_characters += valid_count

        valid_characters = sum(script_counts.values())
        confidence = (
            round(confidence_weight / confidence_characters, 4)
            if confidence_characters
            else None
        )
        groups = sorted(script_counts.items(), key=lambda item: item[1], reverse=True)
        detected = [
            {
                "language": SCRIPT_NAMES.get(script, script),
                "script": script,
                "character_count": count,
                "ratio": round(count / valid_characters, 4) if valid_characters else 0.0,
                "confidence": confidence,
            }
            for script, count in groups
        ]
        secondary_characters = groups[1][1] if len(groups) > 1 else 0
        secondary_ratio = (
            secondary_characters / valid_characters if valid_characters else 0.0
        )
        mixed, reason = self._decision(
            valid_characters,
            len(groups),
            secondary_characters,
            secondary_ratio,
            confidence,
        )
        return VisionLanguageAnalysis(
            detected_languages=detected,
            valid_character_count=valid_characters,
            secondary_language_ratio=round(secondary_ratio, 4),
            secondary_language_characters=secondary_characters,
            confidence=confidence,
            sample_text=self._sample(retained),
            mixed_language_routing=mixed,
            decision_reason=reason,
        )

    def _decision(
        self,
        valid_characters: int,
        language_groups: int,
        secondary_characters: int,
        secondary_ratio: float,
        confidence: float | None,
    ) -> tuple[bool, str]:
        if valid_characters < settings.language_min_valid_characters:
            return False, "insufficient_clean_text"
        if language_groups < 2:
            return False, "single_valid_language_group"
        if secondary_characters < settings.language_secondary_min_characters:
            return False, "secondary_language_below_character_minimum"
        if secondary_ratio < settings.language_secondary_ratio:
            return False, "secondary_language_below_ratio_threshold"
        if confidence is None or confidence < settings.language_confidence_threshold:
            return False, "language_confidence_below_threshold"
        return True, "mixed_language_thresholds_met"

    def _sample(self, words: list[OCRWord]) -> str:
        lines: dict[tuple[int, str], list[OCRWord]] = defaultdict(list)
        for word in words:
            lines[(word.page, word.line_id)].append(word)
        text = "\n".join(
            " ".join(item.text for item in sorted(line, key=lambda item: item.reading_order))
            for _, line in sorted(lines.items(), key=lambda item: (item[0][0], item[1][0].reading_order))
        )
        return text[:320]

    def _script(self, character: str) -> str | None:
        code = ord(character)
        if 0x0041 <= code <= 0x024F or 0x1E00 <= code <= 0x1EFF:
            return "Latin"
        if 0x0900 <= code <= 0x097F:
            return "Devanagari"
        if 0x0980 <= code <= 0x09FF:
            return "Bengali"
        if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
            return "Arabic"
        if 0x0400 <= code <= 0x052F:
            return "Cyrillic"
        if 0x0370 <= code <= 0x03FF:
            return "Greek"
        if 0x0B80 <= code <= 0x0BFF:
            return "Tamil"
        if 0x0C00 <= code <= 0x0C7F:
            return "Telugu"
        if 0x0C80 <= code <= 0x0CFF:
            return "Kannada"
        if 0x0D00 <= code <= 0x0D7F:
            return "Malayalam"
        if 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF:
            return "CJK"
        return None


vision_language_analysis_service = VisionLanguageAnalysisService()
