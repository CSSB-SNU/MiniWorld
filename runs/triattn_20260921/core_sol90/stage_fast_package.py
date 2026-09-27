"""Build the guarded exact candidate under its final serving module name."""
from pathlib import Path
import os,runpy,shutil,sys
HERE=Path(__file__).resolve().parent
variant=sys.argv[1] if len(sys.argv)>1 else 'basefastdispatchsyn'
package=sys.argv[2] if len(sys.argv)>2 else 'fast_package'
src=HERE/variant;dst=HERE/package
dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if not p.is_file() or p.suffix not in ('.py','.cu','.cuh','.h'):continue
 relative=p.relative_to(src)
 if relative.as_posix()=='triattn_m1.py':relative=Path('build_source.py')
 q=dst/relative;q.parent.mkdir(parents=True,exist_ok=True)
 s=p.read_text().replace('ta_sol_'+variant,'ta_core_broadcast').replace('triattn_sol_'+variant,'triattn_broadcast')
 q.write_text(s)
for name in ('__init__.py','build_native.py'):
 shutil.copy2(HERE.parent/'core_tiles/package'/name,dst/name)
os.environ['CUTLASS_PATH']=str(HERE.parents[1]/'anthropic_adoption_20260919/cutlass-4.2')
os.environ['TORCH_EXTENSIONS_DIR']=str(HERE/('fast_build' if package=='fast_package' else package+'_build'))
runpy.run_path(str(dst/'build_native.py'),run_name='__main__')
