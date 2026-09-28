# Teaching Decision Layer v1

Profile-scoped education judgment plugin.

Principle: vision sees -> Teaching Decision Layer judges -> teacher model teaches.

It does not read images itself and does not persist raw learner evidence.

Perception checks:
- is_student_handwriting
- is_original_content
- is_teacher_annotation
- ocr_uncertain
- answer_region_detected
- answer_complete

Perception thresholds: YES >= 0.90, NO <= 0.10; gray requires targeted visual recheck.

Pedagogy checks:
- understanding_demonstrated
- guessing_likely
- needs_followup_question
- ready_to_advance
- learning_objective_met
- response_too_entertaining

Pedagogy thresholds: YES >= 0.80, NO <= 0.20; gray defers to the teacher model.

Trigger only at decision hinges:
- ambiguous image/worksheet evidence;
- after an answer when true understanding is unclear;
- before advancing after weak or ambiguous evidence;
- repeated errors;
- session objective/engagement review.

Clear deterministic facts remain authoritative.
