"""ML classifier arm of the ablation: text -> (advantage, algorithm).

Addresses IEEE Access reviewer 1, comment 8 — the deterministic engine "does not
compare its performance with machine learning or LLM-based classifiers... The choice
of the proposed methodology could be better justified through a comparative
analysis" — and comment 3, which asks whether any back-propagation-based learning is
part of the architecture. The classifier here is a feedforward network trained by
backprop, so the answer is concrete rather than rhetorical.

Design constraint that makes the comparison fair: the existing LLM ablation consumes
the RAW PROBLEM DESCRIPTION. So this arm must too. A classifier fed the analyzer's
already-extracted features would be solving a strictly easier problem and the
numbers would not be comparable.

Protocols (mirroring the paper's own split):
  1. Leave-one-out CV over the 15 standard problems -> compare with the framework's
     15/15 and the bare LLM's 87-93%.
  2. Train on all 15 standard, test on the 12 adversarial -> compare with the
     framework's 9/12 and the LLM's 8/12 (Table 2).

The vectorizer is refit inside every fold. Fitting it once over all data would leak
test vocabulary into training and inflate the LOOCV number.

Usage:
    python -m benchmarks.ml_ablation
"""

import argparse
from collections import Counter

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

from benchmarks.adversarial_problems import ADVERSARIAL_PROBLEMS
from benchmarks.benchmark_suite import BENCHMARK_PROBLEMS

HIDDEN_DIM = 16
EPOCHS = 300
LR = 0.05
SEED = 0

# Same 5 reachable (advantage, algorithm) pairs the decision engine can emit.
CLASSES = [
    ("classical_preferred", "none"),
    ("clear_advantage", "grover"),
    ("clear_advantage", "vqe"),
    ("no_current_advantage", "none"),
    ("potential_advantage", "qaoa"),
]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

ADVERSARIAL_CATEGORIES = {
    "HYPE": "Hype bait",
    "DISGUISED": "Disguised quantum",
    "SCALE": "Scale traps",
    "MISDIR": "Misdirection",
}


def category_of(problem_id: str) -> str:
    for prefix, name in ADVERSARIAL_CATEGORIES.items():
        if problem_id.startswith(prefix):
            return name
    return "Other"


def load_dataset() -> tuple[list, list]:
    """Return (standard, adversarial) as lists of (id, text, label_index)."""
    standard = [
        (p.id, p.description, CLASS_INDEX[(p.expected_advantage, p.expected_algorithm)])
        for p in BENCHMARK_PROBLEMS
    ]
    adversarial = [
        (
            p["id"],
            p["description"],
            CLASS_INDEX[(p["expected_advantage"], p["expected_algorithm"])],
        )
        for p in ADVERSARIAL_PROBLEMS
    ]
    return standard, adversarial


class TextMLP(nn.Module):
    """vocab -> hidden (ReLU) -> 5 classes. Trained by backpropagation."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(CLASSES)),
        )

    def forward(self, x):
        return self.net(x)


def fit_predict(train, test, seed: int = SEED) -> list[int]:
    """Fit TF-IDF + MLP on `train`, return predicted class indices for `test`."""
    torch.manual_seed(seed)

    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1)
    x_train = vectorizer.fit_transform([t for _, t, _ in train]).toarray()
    x_test = vectorizer.transform([t for _, t, _ in test]).toarray()

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor([y for _, _, y in train], dtype=torch.long)

    model = TextMLP(xt.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(xt), yt)   # forward
        loss.backward()                 # backpropagation
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32))
    return logits.argmax(dim=1).tolist()


def majority_baseline(train, test) -> list[int]:
    """Predict the most frequent training label for everything.

    Note this scores 0/15 under LOOCV on the standard set, which is a quirk worth
    understanding rather than a bug: the majority class holds only 5 of 15 items, so
    holding out a majority-class example leaves a 4-4 tie with QAOA, and holding out
    anything else guarantees a miss. The informative reference points are the
    majority-class prior and chance, both reported below.
    """
    most_common = Counter(y for _, _, y in train).most_common(1)[0][0]
    return [most_common] * len(test)


def algorithm_of(index: int) -> str:
    """Algorithm name only — the paper reports algorithm and full accuracy separately."""
    return CLASSES[index][1]


def loocv(standard) -> tuple[int, int, list]:
    """Leave-one-out CV. With 15 examples, a single flip moves accuracy ~6.7 points."""
    correct, details = 0, []
    for i in range(len(standard)):
        train = standard[:i] + standard[i + 1:]
        test = [standard[i]]
        pred = fit_predict(train, test, seed=SEED + i)[0]
        ok = pred == standard[i][2]
        correct += ok
        details.append((standard[i][0], CLASSES[pred], CLASSES[standard[i][2]], ok))
    algo_correct = sum(a[1][1] == a[2][1] for a in details)
    return correct, algo_correct, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="per-problem results")
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="repeat the adversarial protocol over N seeds and report the spread",
    )
    args = parser.parse_args()

    standard, adversarial = load_dataset()

    print("=" * 74)
    print("ML ABLATION ARM — TF-IDF + feedforward network (backprop)")
    print("=" * 74)
    print(f"\nTraining pool: {len(standard)} standard problems, {len(CLASSES)} classes")
    print("Label distribution (full pairs):")
    for i, c in enumerate(CLASSES):
        print(f"    {c[0]}/{c[1]:8s} n={sum(1 for _, _, y in standard if y == i)}")
    print()

    # --- Protocol 1: LOOCV on the standard benchmark ---
    print("--- Protocol 1: leave-one-out CV, 15 standard problems ---")
    correct, algo_correct, details = loocv(standard)
    n = len(standard)
    print(f"  MLP, algorithm only:        {algo_correct}/{n} ({algo_correct/n:.1%})")
    print(f"  MLP, algorithm + advantage: {correct}/{n} ({correct/n:.1%})")

    prior = Counter(y for _, _, y in standard).most_common(1)[0][1]
    print(f"  Majority-class prior:       {prior}/{n} ({prior/n:.1%})")
    print(f"  Chance ({len(CLASSES)} classes):            {1/len(CLASSES):.1%}")

    if args.verbose:
        for pid, pred, truth, ok in details:
            if not ok:
                kind = "advantage" if pred[1] == truth[1] else "algorithm"
                print(f"    MISS {pid:28s} pred={pred[0]}/{pred[1]}  true={truth[0]}/{truth[1]}  [{kind}]")

    # --- Protocol 2: train on standard, test on adversarial ---
    print("\n--- Protocol 2: train on 15 standard, test on 12 adversarial ---")
    preds = fit_predict(standard, adversarial)
    adv_correct = sum(p == y for p, (_, _, y) in zip(preds, adversarial))
    adv_algo = sum(algorithm_of(p) == algorithm_of(y) for p, (_, _, y) in zip(preds, adversarial))
    m = len(adversarial)
    print(f"  MLP, algorithm only:        {adv_algo}/{m} ({adv_algo/m:.1%})")
    print(f"  MLP, algorithm + advantage: {adv_correct}/{m} ({adv_correct/m:.1%})")

    by_category: dict[str, list[int]] = {}
    for pred, (pid, _, y) in zip(preds, adversarial):
        by_category.setdefault(category_of(pid), []).append(int(pred == y))

    print(f"\n  {'Category':22s} {'Correct':>10}")
    for name in ADVERSARIAL_CATEGORIES.values():
        hits = by_category.get(name, [])
        if hits:
            print(f"  {name:22s} {sum(hits)}/{len(hits):>8}")

    if args.verbose:
        print()
        for pred, (pid, _, y) in zip(preds, adversarial):
            mark = "ok  " if pred == y else "MISS"
            print(f"    {mark} {pid:32s} pred={CLASSES[pred][0]}/{CLASSES[pred][1]}"
                  f"  true={CLASSES[y][0]}/{CLASSES[y][1]}")

    # --- Three-way comparison ---
    print("\n" + "=" * 74)
    print("THREE-WAY COMPARISON")
    print("=" * 74)
    print(f"  {'Approach':34s} {'Standard':>12} {'Adversarial':>14}")
    print(f"  {'Deterministic framework':34s} {'15/15 (100%)':>12} {'9/12 (75%)':>14}")
    print(f"  {'Bare LLM (Claude Haiku 4.5)':34s} {'13-14/15':>12} {'8/12 (67%)':>14}")
    print(
        f"  {'TF-IDF + MLP (this arm)':34s} "
        f"{f'{correct}/15 ({correct/15:.0%})':>12} "
        f"{f'{adv_correct}/12 ({adv_correct/12:.0%})':>14}"
    )
    print(
        "\n  Framework and LLM figures are from the paper's Table 2 / Section 6.2.\n"
        "  The MLP is trained on 15 examples — the dataset size IS the finding here,\n"
        "  not an incidental limitation."
    )

    if args.seeds > 1:
        print("\n" + "=" * 74)
        print(f"SEED SENSITIVITY — adversarial protocol over {args.seeds} seeds")
        print("=" * 74)
        scores, cat_scores = [], {}
        for seed in range(args.seeds):
            pr = fit_predict(standard, adversarial, seed=seed)
            scores.append(sum(a == y for a, (_, _, y) in zip(pr, adversarial)))
            for a, (pid, _, y) in zip(pr, adversarial):
                cat_scores.setdefault(category_of(pid), []).append(int(a == y))
        mean = sum(scores) / len(scores)
        var = sum((x - mean) ** 2 for x in scores) / len(scores)
        print(f"  correct/12 per seed: {scores}")
        print(f"  mean {mean:.2f}/12 ({mean/12:.1%})  sd {var ** 0.5:.2f}  "
              f"range [{min(scores)}, {max(scores)}]")
        print("\n  Per-category hit rate across all seeds:")
        for name in ADVERSARIAL_CATEGORIES.values():
            hits = cat_scores.get(name, [])
            if hits:
                print(f"    {name:22s} {sum(hits)/len(hits):6.1%}  (n={len(hits)})")


if __name__ == "__main__":
    main()
