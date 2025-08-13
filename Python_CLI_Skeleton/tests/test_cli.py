import subprocess, sys, json

def run(args):
    return subprocess.run([sys.executable, "-m", "cli_app", *args], check=True, capture_output=True, text=True)

def test_greet():
    out = run(["greet", "Tester"]).stdout.strip()
    assert out == "Hello, Tester!"

def test_sum():
    out = run(["sum", "1", "2", "3"]).stdout.strip()
    assert out == "6.0"

def test_version_json():
    out = run(["version", "--json"]).stdout.strip()
    data = json.loads(out)
    assert "version" in data
