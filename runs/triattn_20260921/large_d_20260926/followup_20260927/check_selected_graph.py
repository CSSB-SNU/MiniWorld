from pathlib import Path
import runpy
import sys
import selected_next
BASE=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE))
import selected
selected.attach=selected_next.attach
runpy.run_path(str(BASE/'check_selected_graph.py'),run_name='__main__')
