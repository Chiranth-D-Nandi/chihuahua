"""
FISSION — R0 Backdoor Attack (BadCLIP-style visual trigger).

Public API (lazy imports so `import attacks.r0_backdoor` works without torch):
    from attacks.r0_backdoor.badclip import BadCLIPAttack, poison_dataset
"""

def __getattr__(name: str):
    if name in {
        "BadCLIPAttack",
        "poison_dataset",
        "evaluate_backdoor",
        "TriggerConfig",
        "train_poisoned_probe",
        "run_b0_against_poisoned_probe",
    }:
        from attacks.r0_backdoor import badclip as _b
        return getattr(_b, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BadCLIPAttack",
    "poison_dataset",
    "evaluate_backdoor",
    "TriggerConfig",
    "train_poisoned_probe",
    "run_b0_against_poisoned_probe",
]
