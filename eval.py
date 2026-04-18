"""
Pocket-Agent Local Evaluation
===============================
Runs the model against the public test set and scores results.

Usage:
    python eval.py
"""
import json
import os
import sys
import time
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_tool_call(response: str) -> dict | None:
    """Extract a tool call from the model's response."""
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def score_example(prediction: str, expected: dict) -> float:
    """Score a single example using the grading rubric."""
    pred_call = parse_tool_call(prediction)

    # Case 1: Should refuse
    if expected.get("should_refuse", False):
        if pred_call is not None:
            return -0.5  # Emitted tool call when should have refused
        return 1.0  # Correctly refused

    # Case 2: Should emit a tool call
    if pred_call is None:
        return 0.0  # Failed to emit a tool call

    # Check tool name
    if pred_call.get("tool") != expected["tool"]:
        return 0.0  # Wrong tool

    # Check args
    expected_args = expected.get("args", {})
    pred_args = pred_call.get("args", {})

    all_correct = True
    for key, exp_val in expected_args.items():
        pred_val = pred_args.get(key)
        if pred_val is None:
            all_correct = False
            continue

        # Numeric comparison with plus/minus 1% tolerance
        if isinstance(exp_val, (int, float)) and isinstance(pred_val, (int, float)):
            if exp_val == 0:
                if pred_val != 0:
                    all_correct = False
            elif abs(pred_val - exp_val) / abs(exp_val) > 0.01:
                all_correct = False
        elif str(pred_val).lower() != str(exp_val).lower():
            all_correct = False

    return 1.0 if all_correct else 0.5


def main():
    test_path = os.path.join(os.path.dirname(__file__), "starter", "public_test.jsonl")

    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        sys.exit(1)

    # Load test data
    test_data = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line.strip()))

    print(f"Loaded {len(test_data)} test examples")
    print("=" * 70)

    # Import inference module
    try:
        import inference
    except Exception as e:
        print(f"Failed to import inference module: {e}")
        print("Make sure the model file exists at models/pocket-agent.gguf")
        sys.exit(1)

    # Run evaluation
    results = []
    slice_scores = {}
    total_score = 0
    latencies = []

    for i, example in enumerate(test_data):
        prompt = example["prompt"]
        history = example.get("history", [])
        expected = example["expected"]
        sl = example.get("slice", "?")

        # Time the inference
        start = time.time()
        try:
            prediction = inference.run(prompt, history)
        except Exception as e:
            prediction = f"ERROR: {e}"
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        # Score
        score = score_example(prediction, expected)
        total_score += score

        # Track by slice
        if sl not in slice_scores:
            slice_scores[sl] = {"total": 0, "count": 0, "scores": []}
        slice_scores[sl]["total"] += score
        slice_scores[sl]["count"] += 1
        slice_scores[sl]["scores"].append(score)

        # Print result
        status = "PASS" if score >= 1.0 else "PARTIAL" if score > 0 else "FAIL" if score == 0 else "PENALTY"
        icon = {"PASS": "+", "PARTIAL": "~", "FAIL": "x", "PENALTY": "!"}[status]
        print(f"  [{icon}] [{sl}] Score={score:+.1f} Latency={latency_ms:.0f}ms")
        print(f"      Prompt: {prompt[:80]}...")
        print(f"      Response: {prediction[:100]}...")
        print()

        results.append({
            "prompt": prompt,
            "prediction": prediction,
            "expected": expected,
            "score": score,
            "latency_ms": latency_ms,
            "slice": sl,
        })

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    mean_score = total_score / max(len(test_data), 1)
    mean_latency = sum(latencies) / max(len(latencies), 1)
    max_latency = max(latencies) if latencies else 0

    print(f"\nTotal Score: {total_score:.1f} / {len(test_data)}")
    print(f"Mean Score:  {mean_score:.3f}")
    print(f"Mean Latency: {mean_latency:.0f} ms")
    print(f"Max Latency:  {max_latency:.0f} ms")

    print("\nPer-Slice Breakdown:")
    for sl in sorted(slice_scores.keys()):
        s = slice_scores[sl]
        sl_mean = s["total"] / max(s["count"], 1)
        print(f"  Slice {sl}: {s['total']:.1f}/{s['count']} (mean={sl_mean:.3f}) {s['scores']}")

    # Gate checks
    print("\nGATE CHECKS:")
    model_path = os.path.join(os.path.dirname(__file__), "models", "pocket-agent.gguf")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        gate_pass = size_mb <= 500
        print(f"  Model size: {size_mb:.1f} MB {'PASS' if gate_pass else 'FAIL'} (gate: <=500 MB)")
        if size_mb <= 250:
            print(f"  Bonus: <=250 MB ACHIEVED!")
    else:
        print(f"  Model file not found at {model_path}")

    latency_gate = mean_latency <= 200
    print(f"  Mean latency: {mean_latency:.0f} ms {'PASS' if latency_gate else 'FAIL'} (gate: <=200 ms)")

    # Check for network imports
    inference_path = os.path.join(os.path.dirname(__file__), "inference.py")
    if os.path.exists(inference_path):
        import ast
        with open(inference_path, "r") as f:
            tree = ast.parse(f.read())
        forbidden = {"requests", "urllib", "http", "socket"}
        found_forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden:
                        found_forbidden.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in forbidden:
                    found_forbidden.append(node.module)

        if found_forbidden:
            print(f"  Network imports: FAIL (found: {found_forbidden})")
        else:
            print(f"  Network imports: PASS (none found)")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "total_score": total_score,
            "mean_score": mean_score,
            "mean_latency_ms": mean_latency,
            "max_latency_ms": max_latency,
            "slice_scores": {k: {"total": v["total"], "count": v["count"]} for k, v in slice_scores.items()},
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
