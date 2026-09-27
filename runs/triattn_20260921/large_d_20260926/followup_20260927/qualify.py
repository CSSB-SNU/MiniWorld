"""Run the frozen whole-module PyTorch/SGD fixtures with the new boundary."""
from pathlib import Path
import runpy
import sys
import experiments
import selected_next
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
import candidate
candidate.attach=selected_next.attach
runpy.run_path(str(BASE/'qualify.py'),run_name='__main__')
