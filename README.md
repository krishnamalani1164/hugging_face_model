# Toxic Comment Detector 🔍

A Python command-line tool that uses a pre-trained BERT model from Hugging Face to classify text comments as **toxic** or **clean**, displaying a confidence score for each prediction.

---

## What It Does

The script reads a plain-text file of comments (`sample_comments.txt`, one comment per line), runs each one through a toxicity detection model, and prints:

- Whether each comment is **TOXIC 🔴** or **CLEAN 🟢**
- A **confidence percentage** showing how sure the model is
- A **final summary** with the overall breakdown (e.g., *5 toxic / 5 clean out of 10 comments*)

This could be used as a starting point for content moderation tools, comment filtering systems, or social media monitoring dashboards.

---

## Why I Chose This Idea

I wanted to pick a project that is both **practical** and **easy to demonstrate**. Online toxicity is a real, growing problem — platforms like Reddit, YouTube, and Twitter all deal with toxic comments at scale. Building a detector felt like a meaningful use case where NLP can have direct impact.

I also chose this because text classification is one of the cleanest ways to show how powerful pre-trained models are. With just a few lines of code using Hugging Face's `pipeline` API, you get a fully functional classifier without needing to train anything from scratch. That tradeoff — leveraging existing models vs. building from zero — felt like the right technical decision for a time-limited assessment.

---

## Which Hugging Face Model I Used

**Model:** [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert)

I chose this model for several reasons:

1. **Trained on the right data** — it's fine-tuned on the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset, which is one of the most widely-used benchmarks for toxicity detection
2. **BERT architecture** — BERT reads text bidirectionally (left-to-right *and* right-to-left), which gives it better contextual understanding than older models
3. **Runs on CPU** — the model is ~438 MB, which is large enough to be accurate but small enough to run without a GPU
4. **Simple output format** — returns a toxicity score between 0 and 1, which I threshold at 0.5 to decide TOXIC vs. CLEAN

I considered lighter models like `distilbert-base-uncased` but they aren't fine-tuned for toxicity out of the box. `unitary/toxic-bert` gave me the best balance of accuracy and simplicity.

---

## How to Install Dependencies

**Prerequisites:** Python 3.9 or higher

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/toxic-comment-detector.git
cd toxic-comment-detector

# Install dependencies
pip install -r requirements.txt
```

The dependencies are:

| Package        | Version | Purpose                          |
|----------------|---------|----------------------------------|
| `transformers` | 4.40.2  | Hugging Face pipeline & model    |
| `tokenizers`   | 0.19.1  | Fast tokenisation backend        |
| `torch`        | 2.2.2   | PyTorch for model inference (CPU)|
| `numpy`        | 1.26.4  | Numerical operations             |

> **Note:** On the first run, the model weights (~438 MB) will be automatically downloaded from the Hugging Face Hub and cached locally at `~/.cache/huggingface/`. Subsequent runs load instantly from cache.

---

## How to Run

```bash
python app.py
```

Make sure `sample_comments.txt` is in the same directory as `app.py`. You can edit this file to test your own comments — just put one comment per line.

---

## Example Output

```
Loading the toxicity detection model — this may take a moment on first run...
(The model weights will be downloaded and cached automatically)

Model loaded successfully!

======================================================================
Loaded 10 comment(s) from 'sample_comments.txt'

======================================================================

Classifying comments...

Comment #1
  Text       : You are so stupid, I can't believe anyone would listen to you.
  Result     : TOXIC  🔴
  Confidence : 98.4%
----------------------------------------------------------------------
Comment #2
  Text       : I really enjoyed the presentation today, great job!
  Result     : CLEAN  🟢
  Confidence : 99.9%
----------------------------------------------------------------------
Comment #3
  Text       : Go kill yourself, nobody wants you here.
  Result     : TOXIC  🔴
  Confidence : 94.3%
----------------------------------------------------------------------
Comment #4
  Text       : This is a wonderful community, everyone is so supportive!
  Result     : CLEAN  🟢
  Confidence : 99.9%
----------------------------------------------------------------------
Comment #5
  Text       : You're a complete idiot who knows nothing about this topic.
  Result     : TOXIC  🔴
  Confidence : 98.3%
----------------------------------------------------------------------
Comment #6
  Text       : Thanks for sharing your experience, that was really helpful.
  Result     : CLEAN  🟢
  Confidence : 99.9%
----------------------------------------------------------------------
Comment #7
  Text       : I hate people like you, you make me sick.
  Result     : TOXIC  🔴
  Confidence : 97.6%
----------------------------------------------------------------------
Comment #8
  Text       : The sunset photos you posted are absolutely beautiful!
  Result     : CLEAN  🟢
  Confidence : 99.9%
----------------------------------------------------------------------
Comment #9
  Text       : Shut up you worthless piece of garbage, your opinion means nothing.
  Result     : TOXIC  🔴
  Confidence : 99.2%
----------------------------------------------------------------------
Comment #10
  Text       : I appreciate all the hard work you put into this project, well done!
  Result     : CLEAN  🟢
  Confidence : 99.9%
----------------------------------------------------------------------

======================================================================
SUMMARY
======================================================================
  Total comments analysed : 10
  Toxic comments          : 5  🔴
  Clean comments          : 5  🟢

  Result: 5 toxic / 5 clean out of 10 comments
======================================================================
```

---

## Assumptions, Limitations, and Future Improvements

### Assumptions

- Input comments are in **English** (the model was trained on English text)
- One comment per line in the input file
- A toxicity score above **0.5** is classified as toxic

### Limitations

- **Binary classification only** — the model outputs toxic vs. non-toxic. It doesn't break toxicity down into sub-categories like hate speech, threats, or obscenity
- **English-only** — non-English comments may produce unreliable results
- **No conversational context** — each comment is classified in isolation, so the model can't understand sarcasm or irony that depends on a thread
- **Single file input** — currently only reads from one hardcoded file

### Future Improvements

- Add **multi-label classification** (e.g., using `unitary/multilabel-toxic-comment-classification`) to detect specific toxicity types
- Build a **web interface** with Gradio or Streamlit for non-technical users
- Add **CSV/JSON export** of results for further analysis
- Support **multiple languages** using a multilingual model like `xlm-roberta-base`
- Let users set a **custom confidence threshold** (e.g., only flag comments above 90%)

---

## Project Structure

```
toxic-comment-detector/
├── app.py                # Main script — loads model, classifies, prints results
├── requirements.txt      # Pinned Python dependencies (CPU-safe)
├── README.md             # Project documentation
├── sample_comments.txt   # 10 sample comments for testing
└── .gitignore            # Prevents committing model weights, venv, .env, etc.
```
