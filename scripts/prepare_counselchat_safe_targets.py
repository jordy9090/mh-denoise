import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "nbertagnolli/counsel-chat"
DEFAULT_EXCLUDED_TOPICS = ["intimacy", "human-sexuality", "lgbtq", "spirituality"]

QUESTION_ID_COLS = ["questionID", "question_id", "qid", "questionId"]
QUESTION_TITLE_COLS = ["questionTitle", "question_title", "title"]
QUESTION_TEXT_COLS = ["questionText", "question_text", "question", "text"]
ANSWER_TEXT_COLS = ["answerText", "answer_text", "answer", "response"]
TOPIC_COLS = ["topic", "questionTopic", "category"]
QUESTION_LINK_COLS = ["questionLink", "question_link", "link", "url"]
UPVOTES_COLS = ["upvotes", "upVotes", "answerUpvotes", "answer_upvotes"]
VIEWS_COLS = ["views", "questionViews", "question_views"]
THERAPIST_URL_COLS = ["therapistURL", "therapist_url", "therapistLink"]
THERAPIST_INFO_COLS = ["therapistInfo", "therapist_info", "therapistName", "therapist"]


def clean_text(value):
    text = str(value if value is not None else "")
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def norm_topic(value):
    return clean_text(value).lower().replace("_", "-").replace(" ", "-")


def word_count(text):
    return len(clean_text(text).split())


def first_present(row, names, default=""):
    for name in names:
        if name in row and row[name] is not None:
            value = row[name]
            if str(value).strip():
                return value
    return default


def to_int(value, default=0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def question_text(row):
    title = clean_text(first_present(row, QUESTION_TITLE_COLS))
    text = clean_text(first_present(row, QUESTION_TEXT_COLS))
    if title and text and title.lower() != text.lower():
        return title + "\n\n" + text
    return title or text


def question_id(row, fallback_i):
    value = first_present(row, QUESTION_ID_COLS)
    if str(value).strip():
        return str(value).strip()
    return "question_text:" + question_text(row).lower()


def topic_value(row):
    return clean_text(first_present(row, TOPIC_COLS))


def parse_excluded_topics(value):
    if value is None:
        return set(DEFAULT_EXCLUDED_TOPICS)
    if not value.strip():
        return set()
    return {norm_topic(x) for x in value.split(",") if x.strip()}


def load_rows():
    ds = load_dataset(DATASET_NAME, split="train")
    return [dict(row) for row in ds]


def valid_candidate(row, args, excluded_topics):
    answer = clean_text(first_present(row, ANSWER_TEXT_COLS))
    if not answer:
        return False
    if word_count(answer) > args.max_answer_words:
        return False
    if to_int(first_present(row, UPVOTES_COLS, 0)) < args.min_upvotes:
        return False
    if norm_topic(topic_value(row)) in excluded_topics:
        return False
    if not question_text(row):
        return False
    return True


def choose_best(candidates, seed):
    by_upvotes = defaultdict(list)
    for row in candidates:
        by_upvotes[to_int(first_present(row, UPVOTES_COLS, 0))].append(row)
    best_upvotes = max(by_upvotes)
    tied = by_upvotes[best_upvotes]
    if len(tied) == 1:
        return tied[0]
    rng = random.Random(f"{seed}:{question_id(tied[0], 0)}:{best_upvotes}")
    return rng.choice(tied)


def build_output(row, out_i):
    qid = question_id(row, out_i)
    record = {
        "id": f"counselchat_{out_i:04d}",
        "question_id": qid,
        "questionID": qid,
        "question": question_text(row),
        "safe_response": clean_text(first_present(row, ANSWER_TEXT_COLS)),
        "topic": topic_value(row),
        "source": "CounselChat",
        "safe_target_source": DATASET_NAME,
        "upvotes": to_int(first_present(row, UPVOTES_COLS, 0)),
    }

    optional_fields = [
        ("questionTitle", QUESTION_TITLE_COLS),
        ("questionText", QUESTION_TEXT_COLS),
        ("questionLink", QUESTION_LINK_COLS),
        ("views", VIEWS_COLS),
        ("therapistURL", THERAPIST_URL_COLS),
        ("therapistInfo", THERAPIST_INFO_COLS),
    ]
    for out_key, cols in optional_fields:
        value = first_present(row, cols)
        if value not in ("", None):
            record[out_key] = clean_text(value) if out_key != "views" else to_int(value)

    return record


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_questions", type=int, default=300)
    parser.add_argument("--output", default="data/raw/counselchat_300_safe_targets.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_answer_words", type=int, default=250)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--min_upvotes", type=int, default=0)
    parser.add_argument("--exclude_topics", default=",".join(DEFAULT_EXCLUDED_TOPICS))
    args = parser.parse_args()

    rows = load_rows()
    excluded_topics = parse_excluded_topics(args.exclude_topics)

    by_question = defaultdict(list)
    for i, row in enumerate(rows):
        if valid_candidate(row, args, excluded_topics):
            by_question[question_id(row, i)].append(row)

    selected = [choose_best(candidates, args.seed) for _, candidates in sorted(by_question.items())]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(selected)

    selected = selected[: args.n_questions]
    out_rows = [build_output(row, i) for i, row in enumerate(selected)]
    write_jsonl(out_rows, args.output)

    print("dataset:", DATASET_NAME)
    print("total rows:", len(rows))
    print("unique questions:", len({question_id(row, i) for i, row in enumerate(rows)}))
    print("valid candidate questions after filters:", len(by_question))
    print("selected questions:", len(out_rows))
    print("excluded topics:", sorted(excluded_topics))
    print("topic distribution:", dict(Counter(row["topic"] for row in out_rows)))
    if len(out_rows) < args.n_questions:
        print(f"WARNING: requested {args.n_questions} questions but selected only {len(out_rows)}.")
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
