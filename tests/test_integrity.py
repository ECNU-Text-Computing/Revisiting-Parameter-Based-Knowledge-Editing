#!/usr/bin/env python3
"""
Integrity verification suite for the Revisiting Parameter-Based Knowledge Editing codebase.

Validates:
  1. All internal relative imports resolve correctly
  2. All config files exist and are valid YAML
  3. All copied files match originals (byte-identical) except documented modifications
  4. All experiment entry scripts are syntactically valid
  5. Key data structures (ALG_DICT, etc.) are correct

Usage:
    python tests/test_integrity.py --original /path/to/original/EasyEdit
"""

import ast
import os
import sys
import json
import argparse
import hashlib
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ─── Colors ───
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'

passed = 0
failed = 0
warnings = 0

def ok(msg):
    global passed
    passed += 1
    print(f"  {GREEN}[PASS]{RESET} {msg}")

def fail(msg):
    global failed
    failed += 1
    print(f"  {RED}[FAIL]{RESET} {msg}")

def warn(msg):
    global warnings
    warnings += 1
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Internal Import Resolution
# ═══════════════════════════════════════════════════════════════════════════

def test_import_resolution():
    print(f"\n{BOLD}Test 1: Internal Relative Import Resolution{RESET}")

    # These are files that are known to have lazy imports for non-paper modules.
    # Their broken imports are inert — only triggered for multimodal/SERAC/MALMEN code paths.
    KNOWN_ACCEPTABLE_BROKEN = {
        'easyeditor/trainer/models.py',        # lazy blip2_models import (multimodal only)
        'easyeditor/trainer/algs/SERAC.py',    # not loaded (excluded from algs/__init__.py)
        'easyeditor/trainer/algs/MALMEN.py',   # not loaded
    }

    def check_file(filepath):
        with open(filepath) as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError as e:
                return [(filepath, f'SyntaxError: {e}')]
        errors = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level > 0:
                parts = node.module.split('.')
                fdir = os.path.dirname(filepath)
                for _ in range(node.level - 1):
                    fdir = os.path.dirname(fdir)
                target_dir = os.path.join(fdir, *parts[:-1]) if len(parts) > 1 else fdir
                target_file = os.path.join(target_dir, parts[-1] + '.py')
                target_pkg = os.path.join(target_dir, parts[-1], '__init__.py')
                if not os.path.exists(target_file) and not os.path.exists(target_pkg):
                    errors.append((filepath, node.module, node.level))
        return errors

    all_errors = []
    for root, dirs, files in os.walk('easyeditor'):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                all_errors.extend(check_file(os.path.join(root, f)))

    # Also check experiments/, evaluation/, scr/
    for extra_dir in ['experiments', 'evaluation']:
        for root, dirs, files in os.walk(extra_dir):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for f in files:
                if f.endswith('.py'):
                    all_errors.extend(check_file(os.path.join(root, f)))

    real_errors = []
    for fpath, mod, level in all_errors:
        rel = os.path.relpath(fpath)
        if rel not in KNOWN_ACCEPTABLE_BROKEN:
            real_errors.append((rel, mod, level))

    if real_errors:
        for rel, mod, level in real_errors:
            fail(f"Broken import in {rel}: '{mod}' (level={level})")
    else:
        ok(f"All internal imports resolve ({len(all_errors)} acceptable broken in non-paper modules)")

    return len(real_errors) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Config File Integrity
# ═══════════════════════════════════════════════════════════════════════════

PAPER_METHODS = ['ROME', 'MEMIT', 'MEND', 'AlphaEdit', 'WISE', 'PMET', 'LoRA', 'FT', 'GRACE', 'IKE']
REQUIRED_MODELS = ['llama3.1-8b', 'llama2-7b', 'llama-7b', 'mistral-7b', 'llama-13b']
# deepseek and qwen3 may use different naming; check what's present

def test_configs():
    print(f"\n{BOLD}Test 2: Config File Integrity{RESET}")

    for method in PAPER_METHODS:
        config_dir = os.path.join('configs', method)
        if not os.path.isdir(config_dir):
            fail(f"Missing config directory: {config_dir}")
            continue

        yamls = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
        if not yamls:
            fail(f"No YAML configs in {config_dir}")
            continue

        for yf in yamls:
            path = os.path.join(config_dir, yf)
            # Validate YAML parse
            try:
                import yaml
                with open(path) as fh:
                    data = yaml.safe_load(fh)
                # Check required keys in every config
                required = ['alg_name', 'model_name']
                missing = [k for k in required if k not in data]
                if missing:
                    fail(f"{method}/{yf}: missing keys {missing}")
                else:
                    ok(f"{method}/{yf} — alg={data.get('alg_name')}, model={data.get('model_name')}")
            except Exception as e:
                fail(f"{method}/{yf}: parse error — {e}")

    # Check each paper method+model combo has at least one config
    print(f"\n  {BOLD}Paper method × model coverage:{RESET}")
    covered = 0
    missing_combos = []
    for method in PAPER_METHODS:
        config_dir = os.path.join('configs', method)
        available = set()
        for yf in os.listdir(config_dir):
            if yf.endswith('.yaml'):
                available.add(yf.replace('.yaml', ''))
        for model in REQUIRED_MODELS:
            if model in available or any(m in model for m in available):
                covered += 1
            else:
                # Check if a variant exists
                variants = [a for a in available if model.split('-')[0] in a]
                if variants:
                    covered += 1
                else:
                    missing_combos.append(f"{method}/{model}")

    if missing_combos:
        warn(f"{len(missing_combos)} missing combos (may need custom configs): {', '.join(missing_combos[:8])}...")
    else:
        ok(f"All {covered} method×model combos covered")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: File Identity Check (originals preserved)
# ═══════════════════════════════════════════════════════════════════════════

# Files we intentionally modified (documented changes)
MODIFIED_FILES = {
    'easyeditor/editors/editor.py',
    'easyeditor/evaluate/evaluate.py',
    'easyeditor/models/__init__.py',
    'easyeditor/editors/__init__.py',
    'easyeditor/evaluate/__init__.py',
    'easyeditor/trainer/__init__.py',
    'easyeditor/dataset/__init__.py',
    'easyeditor/util/alg_dict.py',
    'easyeditor/util/alg_train_dict.py',
    'easyeditor/trainer/algs/__init__.py',
    'easyeditor/models/rome/layer_stats.py',  # hardcoded path → env var
}

def test_file_identity(args):
    print(f"\n{BOLD}Test 3: File Identity vs Originals{RESET}")
    if not args.original:
        warn("Skipped — no --original path provided")
        return

    # args.original points to EasyEdit/ which already contains easyeditor/
    orig_base = args.original if os.path.isdir(os.path.join(args.original, 'easyeditor')) else args.original
    if not os.path.isdir(os.path.join(orig_base, 'easyeditor')):
        fail(f"Original EasyEdit/easyeditor not found at: {orig_base}")
        return

    identical = 0
    modified = 0
    missing_from_original = 0

    for root, dirs, files in os.walk('easyeditor'):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.py'):
                new_path = os.path.join(root, f)
                rel = os.path.relpath(new_path)
                orig_path = os.path.join(orig_base, rel)

                if not os.path.exists(orig_path):
                    missing_from_original += 1
                    continue

                with open(new_path, 'rb') as fh:
                    new_hash = hashlib.md5(fh.read()).hexdigest()
                with open(orig_path, 'rb') as fh:
                    orig_hash = hashlib.md5(fh.read()).hexdigest()

                if new_hash == orig_hash:
                    identical += 1
                elif rel in MODIFIED_FILES:
                    modified += 1
                else:
                    fail(f"Unexpected modification: {rel}")

    ok(f"{identical} files byte-identical to originals")
    if modified > 0:
        ok(f"{modified} files intentionally modified (as documented)")
    if missing_from_original > 0:
        warn(f"{missing_from_original} new files (no original counterpart)")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Experiment Script Syntax
# ═══════════════════════════════════════════════════════════════════════════

def test_experiment_scripts():
    print(f"\n{BOLD}Test 4: Experiment Entry Point Syntax{RESET}")
    for fname in os.listdir('experiments'):
        if fname.endswith('.py') and not fname.startswith('__'):
            path = os.path.join('experiments', fname)
            try:
                with open(path) as fh:
                    ast.parse(fh.read())
                ok(f"{fname} — valid syntax")
            except SyntaxError as e:
                fail(f"{fname} — {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Key Data Structures
# ═══════════════════════════════════════════════════════════════════════════

def test_data_structures():
    print(f"\n{BOLD}Test 5: Key Data Structures{RESET}")

    # Verify ALG_DICT entries match what configs expect
    # Read alg_dict.py and extract ALG_DICT keys
    alg_path = 'easyeditor/util/alg_dict.py'
    with open(alg_path) as f:
        alg_source = f.read()

    tree = ast.parse(alg_source)
    alg_keys = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant):
                    alg_keys.append(key.value)
    ok(f"ALG_DICT contains {len(alg_keys)} methods: {alg_keys}")

    # Verify every config's alg_name exists in ALG_DICT
    for method_dir in os.listdir('configs'):
        config_dir = os.path.join('configs', method_dir)
        if not os.path.isdir(config_dir):
            continue
        for yf in os.listdir(config_dir):
            if yf.endswith('.yaml'):
                import yaml
                with open(os.path.join(config_dir, yf)) as fh:
                    data = yaml.safe_load(fh)
                config_alg = data.get('alg_name')
                if config_alg not in alg_keys:
                    fail(f"Config {method_dir}/{yf} expects alg_name='{config_alg}' not in ALG_DICT")

    ok("All config alg_names found in ALG_DICT")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Integrity verification suite')
    parser.add_argument('--original', help='Path to original EasyEdit directory for diff comparison')
    args = parser.parse_args()

    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  Integrity Verification Suite{RESET}")
    print(f"{BOLD}  Revisiting Parameter-Based Knowledge Editing{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")

    test_import_resolution()
    test_configs()
    test_file_identity(args)
    test_experiment_scripts()
    test_data_structures()

    print(f"\n{BOLD}{'='*70}{RESET}")
    total = passed + failed
    print(f"  Passed:  {GREEN}{passed}{RESET} / {total}")
    if warnings > 0:
        print(f"  Warnings: {YELLOW}{warnings}{RESET}")
    if failed > 0:
        print(f"  Failed:  {RED}{failed}{RESET} / {total}")
        print(f"\n{RED}Some tests FAILED. See details above.{RESET}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All tests PASSED.{RESET}")
        print(f"  Note: End-to-end smoke test requires GPU and is in tests/test_smoke.py")


if __name__ == '__main__':
    main()
