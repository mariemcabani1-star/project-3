"""python -m sandbox.reset  : restore the test calendar to its seed state."""
from sandbox.store import reset, state_path

if __name__ == "__main__":
    reset()
    print(f"Sandbox reset -> {state_path()}")
