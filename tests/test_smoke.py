#!/usr/bin/env python3
"""
End-to-end smoke test for knowledge editing experiments.

Runs a minimal edit (1 example, ROME on gpt2-xl if available, else checks
all imports and configurations load correctly in a real Python environment).

Requires: torch, transformers (GPU optional but recommended)

Usage:
    python tests/test_smoke.py                    # Full smoke test
    python tests/test_smoke.py --dry-run          # Import/config check only
    python tests/test_smoke.py --method ROME --model gpt2-xl  # Specific test
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

GREEN = '\033[92m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'


def test_import_chain():
    """Verify the full import chain works in a real Python environment."""
    print(f"\n{BOLD}── Import chain{RESET}")
    try:
        from easyeditor import (
            FTHyperParams, IKEHyperParams, MEMITHyperParams, ROMEHyperParams,
            LoRAHyperParams, MENDHyperParams, GraceHyperParams, WISEHyperParams,
            PMETHyperParams, AlphaEditHyperParams,
        )
        print(f"  {GREEN}[OK]{RESET} All HyperParams classes imported")
    except Exception as e:
        print(f"  {RED}[FAIL]{RESET} HyperParams import: {e}")
        return False

    try:
        from easyeditor import BaseEditor
        print(f"  {GREEN}[OK]{RESET} BaseEditor imported")
    except Exception as e:
        print(f"  {RED}[FAIL]{RESET} BaseEditor import: {e}")
        return False

    try:
        from easyeditor import ZsreDataset, WikiCounterfactDataset, KnowEditDataset
        print(f"  {GREEN}[OK]{RESET} Dataset classes imported")
    except Exception as e:
        print(f"  {RED}[FAIL]{RESET} Dataset import: {e}")
        return False

    try:
        from easyeditor.util.alg_dict import ALG_DICT
        methods = list(ALG_DICT.keys())
        print(f"  {GREEN}[OK]{RESET} ALG_DICT: {methods}")
    except Exception as e:
        print(f"  {RED}[FAIL]{RESET} ALG_DICT: {e}")
        return False

    return True


def test_configs_load():
    """Verify all configs load correctly."""
    print(f"\n{BOLD}── Config loading{RESET}")
    import yaml
    from pathlib import Path

    all_ok = True
    for config_file in sorted(Path('configs').rglob('*.yaml')):
        try:
            with open(config_file) as f:
                data = yaml.safe_load(f)
            alg = data.get('alg_name', 'MISSING')
            model = data.get('model_name', 'MISSING')
            print(f"  {GREEN}[OK]{RESET} {config_file} → {alg} @ {model}")
        except Exception as e:
            print(f"  {RED}[FAIL]{RESET} {config_file}: {e}")
            all_ok = False
    return all_ok


def test_minimal_edit(method='ROME', model='gpt2-xl', config_dir='configs'):
    """Run a minimal single-edit test. Requires GPU or CPU with compatible model."""
    print(f"\n{BOLD}── Minimal edit: {method} on {model}{RESET}")

    config_file = os.path.join(config_dir, method, f'{model}.yaml')
    if not os.path.exists(config_file):
        # Try to find any available config
        config_dir_method = os.path.join(config_dir, method)
        yamls = [f for f in os.listdir(config_dir_method) if f.endswith('.yaml')]
        if yamls:
            config_file = os.path.join(config_dir_method, yamls[0])
            model = yamls[0].replace('.yaml', '')
            print(f"  Using: {config_file}")
        else:
            print(f"  {RED}[FAIL]{RESET} No config found for {method}")
            return False

    from easyeditor import BaseEditor

    # Resolve which hparams class to use
    from easyeditor.util.alg_dict import ALG_DICT
    method_to_cls = {
        'ROME': 'ROMEHyperParams',
        'MEMIT': 'MEMITHyperParams',
        'PMET': 'PMETHyperParams',
        'AlphaEdit': 'AlphaEditHyperParams',
        'FT': 'FTHyperParams',
        'MEND': 'MENDHyperParams',
        'WISE': 'WISEHyperParams',
        'LoRA': 'LoRAHyperParams',
        'GRACE': 'GraceHyperParams',
        'IKE': 'IKEHyperParams',
    }

    try:
        # Import the correct hparams class
        import importlib
        mod_name = f'easyeditor.models.{method.lower()}'
        mod = importlib.import_module(mod_name)
        cls_name = method_to_cls[method]
        hparams_cls = getattr(mod, cls_name)

        hparams = hparams_cls.from_hparams(config_file)
        print(f"  Hparams loaded: alg={hparams.alg_name}, model={hparams.model_name}")

        # Try to instantiate editor (this loads the model — needs memory/GPU)
        editor = BaseEditor.from_hparams(hparams)
        print(f"  {GREEN}[OK]{RESET} Editor instantiated, model loaded")

        # Run a trivial edit
        metrics, edited_model, _ = editor.edit(
            prompts=["Who wrote Romeo and Juliet?"],
            target_new=["William Shakespeare"],
            ground_truth=["William Shakespeare"],
            sequential_edit=False,
            test_generation=False,
        )
        print(f"  {GREEN}[OK]{RESET} Edit completed: {len(metrics)} metric(s)")
        return True

    except Exception as e:
        print(f"  {RED}[FAIL]{RESET} {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    p = argparse.ArgumentParser(description='End-to-end smoke test')
    p.add_argument('--dry-run', action='store_true', help='Only check imports/configs')
    p.add_argument('--method', default='ROME', help='Method to test')
    p.add_argument('--model', default='gpt2-xl', help='Model to test')
    p.add_argument('--gpu-only', action='store_true', help='Skip if no GPU')
    args = p.parse_args()

    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  End-to-End Smoke Test{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Always run import and config checks
    ok = True
    ok &= test_import_chain()
    ok &= test_configs_load()

    if not args.dry_run:
        ok &= test_minimal_edit(args.method, args.model)

    print(f"\n{BOLD}{'='*60}{RESET}")
    if ok:
        print(f"{GREEN}{BOLD}  Smoke test PASSED{RESET}")
    else:
        print(f"{RED}{BOLD}  Smoke test FAILED{RESET}")
        sys.exit(1)


if __name__ == '__main__':
    main()
