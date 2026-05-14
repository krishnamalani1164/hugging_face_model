# =============================================================================
# Toxic Comment Detector
# =============================================================================
# Uses the Hugging Face model `unitary/toxic-bert` to classify user comments
# as TOXIC or CLEAN and display a confidence score for each prediction.
#
# How to run:
#   pip install -r requirements.txt
#   python app.py
# =============================================================================

from transformers import pipeline

# ---------------------------------------------------------------------------
# SECTION 1: Load the Model
# ---------------------------------------------------------------------------
# `pipeline` is a high-level Hugging Face API that bundles tokenisation,
# model inference, and post-processing into a single callable object.
#
# model="unitary/toxic-bert" — a BERT model fine-tuned on the Jigsaw
# toxicity dataset; it outputs a label (toxic / non-toxic) and a score.
#
# device=-1 forces CPU-only inference (no GPU required).
# ---------------------------------------------------------------------------

print("Loading the toxicity detection model — this may take a moment on first run...")
print("(The model weights will be downloaded and cached automatically)\n")

classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    device=-1          # -1 = CPU; change to 0 for GPU if available
)

print("Model loaded successfully!\n")
print("=" * 70)

# ---------------------------------------------------------------------------
# SECTION 2: Read Comments from File
# ---------------------------------------------------------------------------
# Each line in sample_comments.txt is treated as one independent comment.
# Empty lines are skipped so formatting in the file doesn't cause issues.
# ---------------------------------------------------------------------------

COMMENTS_FILE = "sample_comments.txt"

try:
    with open(COMMENTS_FILE, "r", encoding="utf-8") as f:
        # Strip whitespace and skip blank lines
        comments = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"ERROR: Could not find '{COMMENTS_FILE}'.")
    print("Please make sure 'sample_comments.txt' is in the same directory as app.py.")
    exit(1)

print(f"Loaded {len(comments)} comment(s) from '{COMMENTS_FILE}'\n")
print("=" * 70)

# ---------------------------------------------------------------------------
# SECTION 3: Classify Each Comment
# ---------------------------------------------------------------------------
# We loop over every comment, run it through the classifier, and interpret
# the result based on the confidence score:
#   - score > 0.5  → TOXIC  🔴  (model is confident the comment is toxic)
#   - score <= 0.5 → CLEAN  🟢  (model is confident the comment is clean)
#
# The model returns a `score` in [0, 1]. We use 0.5 as the decision
# threshold and convert to a percentage for display.
# ---------------------------------------------------------------------------

toxic_count = 0   # Running total of toxic comments
clean_count = 0   # Running total of clean comments

print("\nClassifying comments...\n")

for index, comment in enumerate(comments, start=1):

    # Run the model — returns a list with one dict, e.g.:
    # [{'label': 'toxic', 'score': 0.9876}]
    result = classifier(comment)[0]

    raw_score = result["score"]          # confidence between 0 and 1

    # NOTE: unitary/toxic-bert always returns label="toxic".
    # The *score* tells us how toxic the comment really is:
    #   score > 0.5  → the model is confident the comment IS toxic
    #   score <= 0.5 → the model is confident the comment is NOT toxic
    # For clean comments we flip the score so it reads as
    # "confidence that this is clean" (e.g. 0.1% toxic → 99.9% clean).

    if raw_score > 0.5:
        display_label = "TOXIC  🔴"
        confidence = raw_score * 100     # e.g. 98.4%
        toxic_count += 1
    else:
        display_label = "CLEAN  🟢"
        confidence = (1 - raw_score) * 100  # e.g. 0.1% toxic → 99.9% clean
        clean_count += 1

    # Pretty-print the result for this comment
    print(f"Comment #{index}")
    print(f"  Text       : {comment}")
    print(f"  Result     : {display_label}")
    print(f"  Confidence : {confidence:.1f}%")
    print("-" * 70)

# ---------------------------------------------------------------------------
# SECTION 4: Print Final Summary
# ---------------------------------------------------------------------------
# After processing all comments, we show an overall breakdown so the user
# can see at a glance how many were flagged as toxic.
# ---------------------------------------------------------------------------

total = len(comments)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Total comments analysed : {total}")
print(f"  Toxic comments          : {toxic_count}  🔴")
print(f"  Clean comments          : {clean_count}  🟢")
print(f"\n  Result: {toxic_count} toxic / {clean_count} clean out of {total} comments")
print("=" * 70)
