"""
Main Pipeline Runner
Runs all steps in sequence: features → labels → train → predict → backtest
Usage: python main.py
"""

import subprocess
import sys

STEPS = [
    ('Step 1 — Feature Engineering', 'scripts/step1_features.py'),
    ('Step 2 — Label Construction',  'scripts/step2_labels.py'),
    ('Step 3 — Train Model',         'scripts/step3_train.py'),
    ('Step 4 — Predict',             'scripts/step4_predict.py'),
    ('Step 5 — Backtest',            'scripts/step5_backtest.py'),
]

if __name__ == '__main__':
    for name, script in STEPS:
        print(f"\n{name}\n")
        result = subprocess.run([sys.executable, script], cwd='.')
        if result.returncode != 0:
            print(f"\n❌ {name} failed. Stopping pipeline.")
            sys.exit(1)

    print("\n✅ Pipeline complete")
