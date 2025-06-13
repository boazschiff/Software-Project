import subprocess
import os

TESTS = [
    {
        "name": "Test 1",
        "args": ["3", "333", "0", "tests/tests/input_1_db_1.txt", "tests/tests/input_1_db_2.txt"],
        "expected": "tests/tests/output_1.txt"
    },
    {
        "name": "Test 2",
        "args": ["7", "0", "tests/tests/input_2_db_1.txt", "tests/tests/input_2_db_2.txt"],
        "expected": "tests/tests/output_2.txt"
    },
    {
        "name": "Test 3",
        "args": ["15", "750", "0", "tests/tests/input_3_db_1.txt", "tests/tests/input_3_db_2.txt"],
        "expected": "tests/tests/output_3.txt"
    }
]

SCRIPT = "kmeanspp.py"

def run_test(test, idx):
    print(f"\nRunning {test['name']}...")
    cmd = ["python3", SCRIPT] + test["args"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("❌ Script exited with error.")
            print(result.stderr)
            return

        output_lines = result.stdout.strip().splitlines()
        with open(test["expected"], 'r') as f:
            expected_lines = [line.strip() for line in f.readlines()]

        if output_lines == expected_lines:
            print("✅ Passed!")
        else:
            print("❌ Output does not match expected.")
            print("--- Output ---")
            print('\n'.join(output_lines))
            print("--- Expected ---")
            print('\n'.join(expected_lines))

    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    for i, test in enumerate(TESTS, 1):
        run_test(test, i)

if __name__ == "__main__":
    main()
