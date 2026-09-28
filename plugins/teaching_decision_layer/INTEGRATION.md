# Hema Teacher integration contract

This plugin is not a global hook. The teacher profile calls it only at explicit teaching decision hinges.

## Image / worksheet path

1. Use the existing vision path first:
   - directly inspect a readable image;
   - if needed, one targeted crop / OCR / grounding check.
2. Keep printed question recognition separate from student-mark interpretation.
3. Build de-identified structured evidence. Good fields include:
   - target_region;
   - region coordinates or role;
   - recognized text/symbol;
   - OCR confidence;
   - printed-font match;
   - stroke irregularity;
   - ink-color relation;
   - whether the mark exists in a known clean source;
   - overlap with printed content;
   - whether answer-region layout is matched.
4. Call teaching_perception_judge only for the checks needed.
5. If a perception check is gray, or ocr_uncertain=yes, do one targeted visual recheck. Do not silently guess.
6. Deterministic source comparison wins over Jev when exact evidence exists.

## Pedagogy path

Call teaching_pedagogy_judge only at a learning hinge:
- a correct answer without enough reasoning evidence;
- repeated same-concept error;
- before advancing when mastery is unclear;
- after a Feynman-style explanation or variant problem;
- session closeout / objective review;
- interaction is becoming entertaining without enough target practice.

Do not call for:
- casual chat;
- an obviously correct/incorrect deterministic answer;
- every short learner turn;
- every generated teaching response.

## Decision authority

- Perception YES >= 0.90; NO <= 0.10; otherwise targeted visual recheck.
- Pedagogy YES >= 0.80; NO <= 0.20; otherwise the Hema Teacher main model reviews.
- Tool/provider failure does not block normal teaching; it returns defer/fallback.
- High-impact family, health, safety, psychological, punishment, or major school-placement decisions remain outside this plugin's authority.
- A Jev judgment is evidence for teaching strategy, not a learner label or permanent diagnosis.

## Privacy

Never send:
- real learner name;
- school/contact/chat/sender IDs;
- credentials;
- raw image bytes;
- full raw transcript.

Only send the minimum de-identified structured evidence required for the current judgment.
