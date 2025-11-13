import subprocess
import sys

scripts = [
    "online_eval.py",
    "online_eval_diffusion_policy.py",
    "online_eval_bc.py"
]

for script in scripts:
    try:
        result = subprocess.run([sys.executable, script], check=True)
        print(f"✅ {script} executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script} execution failed: {e}")
        break  # Optional: stop subsequent scripts on failure