"""Train the Stage 2 MLP by distilling the deterministic rule engine.

The rule engine is a labelling oracle: sample a feature vector, ask
`classify_advantage` for the answer, and you have a labelled example for free. That
sidesteps the real data bottleneck — the 27 hand-written benchmark problems are far
too few to train on, and they stay untouched as a held-out evaluation set.

What this does NOT do is improve accuracy. A distilled network can at best match its
teacher. The point is to see where a learned boundary DIFFERS: the rules switch
hard at N=64, while a trained network produces a graded response near the threshold.
Whether that smoothing is better calibrated than the step is the open question, and
`--report-boundary` prints the evidence.

Usage:
    python -m training.train_advantage_mlp
    python -m training.train_advantage_mlp --samples 40000 --epochs 800
"""

import argparse
import random

import torch
from torch import nn

from quantum_agent.decision_engine import (
    GROVER_MIN_SEARCH_SPACE,
    ProblemType,
    classify_advantage,
    extract_features,
)
from quantum_agent.neural.encoding import (
    CLASSES,
    encode_features,
    label_index,
)
from quantum_agent.neural.model import WEIGHTS_PATH, AdvantageMLP

# Fraction of samples drawn tightly around the Grover threshold. Uniform sampling
# over a log-uniform range puts almost no mass near N=64, so the network would never
# see the boundary it most needs to resolve.
BOUNDARY_SAMPLE_FRACTION = 0.25


def _sample_search_space(rng: random.Random) -> int:
    """Draw a search space size, oversampling the N=64 decision boundary."""
    if rng.random() < BOUNDARY_SAMPLE_FRACTION:
        return rng.randint(
            max(1, GROVER_MIN_SEARCH_SPACE - 32), GROVER_MIN_SEARCH_SPACE + 32
        )
    # log-uniform across the full operating range
    return int(2 ** rng.uniform(0.0, 20.0))


def build_dataset(n_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random problems and label them with the rule engine."""
    rng = random.Random(seed)
    xs: list[list[float]] = []
    ys: list[int] = []

    for _ in range(n_samples):
        features = extract_features(
            problem_type=rng.choice(list(ProblemType)).value,
            search_space_size=_sample_search_space(rng),
            has_oracle=rng.random() < 0.5,
            has_structure=rng.random() < 0.5,
            num_variables=rng.choice([0, 0, rng.randint(1, 200)]),
        )
        advantage, algorithm, _ = classify_advantage(features)
        xs.append(encode_features(features))
        ys.append(label_index(advantage, algorithm))

    return (
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.long),
    )


def train(
    n_samples: int = 20_000,
    epochs: int = 500,
    hidden_dim: int = 16,
    lr: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[AdvantageMLP, float]:
    """Train the network and return it with its held-out accuracy."""
    torch.manual_seed(seed)

    x, y = build_dataset(n_samples, seed)
    x_val, y_val = build_dataset(max(2000, n_samples // 5), seed + 1)

    model = AdvantageMLP(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)   # forward
        loss.backward()               # backpropagation
        optimizer.step()              # gradient descent update

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            with torch.no_grad():
                acc = (model(x_val).argmax(dim=1) == y_val).float().mean().item()
            print(f"  epoch {epoch + 1:4d}/{epochs}  loss={loss.item():.4f}  val_acc={acc:.4f}")

    model.eval()
    with torch.no_grad():
        val_acc = (model(x_val).argmax(dim=1) == y_val).float().mean().item()
    return model, val_acc


def report_boundary(model: AdvantageMLP) -> None:
    """Contrast the rules' hard threshold with the network's graded response."""
    print("\n--- Grover threshold: rules step vs. network confidence ---")
    print(f"{'N':>8}  {'rules':>20}  {'network':>20}  {'p(win)':>7}")
    for n in (16, 32, 48, 56, 60, 62, 64, 66, 70, 80, 128, 1024):
        features = extract_features("unstructured_search", n, has_oracle=True)
        r_adv, r_alg, _ = classify_advantage(features)

        with torch.no_grad():
            probs = torch.softmax(
                model(torch.tensor([encode_features(features)], dtype=torch.float32)),
                dim=1,
            )[0]
        idx = int(probs.argmax().item())
        n_adv, n_alg = CLASSES[idx]

        flag = "" if (r_adv, r_alg) == (n_adv, n_alg) else "   <-- differs"
        print(
            f"{n:>8}  {r_alg.value:>20}  {n_alg.value:>20}  "
            f"{probs[idx].item():>7.3f}{flag}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-boundary", action="store_true")
    parser.add_argument("--out", type=str, default=str(WEIGHTS_PATH))
    args = parser.parse_args()

    print(f"Distilling the rule engine ({args.samples} samples, {args.epochs} epochs)...")
    model, val_acc = train(
        n_samples=args.samples,
        epochs=args.epochs,
        hidden_dim=args.hidden,
        lr=args.lr,
        seed=args.seed,
    )
    print(f"\nHeld-out accuracy vs. rules: {val_acc:.4f}")

    if args.report_boundary:
        report_boundary(model)

    torch.save(model.state_dict(), args.out)
    print(f"\nSaved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
