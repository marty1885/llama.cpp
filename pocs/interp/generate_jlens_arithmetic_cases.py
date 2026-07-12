#!/usr/bin/env python3
"""Generate predeclared single-step arithmetic J-lens cases."""

from pathlib import Path


def main() -> None:
    rows: list[tuple[str, str, str]] = []
    answer_control: list[tuple[str, str, str]] = []
    for a in range(2, 10):
        for b in range(2, 10):
            product = a * b
            rows.append((
                f"product-{a}-{b}",
                f"Compute {a} times {b}, then add 1. Give only the final result:",
                f" {product}",
            ))
            if len(answer_control) < 16:
                answer_control.append((
                    f"answer-product-{a}-{b}",
                    f"Compute {a} times {b}, then add 1. Give only the final result:",
                    f" {product + 1}",
                ))
    for a in range(2, 10):
        for b in range(2, 10):
            total = a + b
            rows.append((
                f"sum-{a}-{b}",
                f"Compute ({a} plus {b}) times 2. Give only the final result:",
                f" {total}",
            ))

    output = Path(__file__).with_name("jlens_arithmetic_cases.tsv")
    output.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    control = Path(__file__).with_name("jlens_arithmetic_answer_control.tsv")
    control.write_text("".join("\t".join(row) + "\n" for row in answer_control), encoding="utf-8")


if __name__ == "__main__":
    main()
