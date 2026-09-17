"""One-command offline reproduction and verification of AlphaOT results and figures."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*args):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src')
    print('+', ' '.join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'build')
    p.add_argument('--plots-only', action='store_true', help='Use the validated results already in the output directory')
    args = p.parse_args()
    output = args.output.resolve()
    results = output / 'results'
    if not args.plots_only:
        run('-m', 'alphaot.pipeline', '--output', str(results))
    run('scripts/verify_results.py', '--results', str(results))
    run('scripts/figures.py', '--results', str(results), '--output', str(output / 'figures'))
    print(f'Complete: {output / "figures"}', flush=True)


if __name__ == '__main__':
    main()
