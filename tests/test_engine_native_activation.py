"""Deferred activation must preserve the running build and qualified old winners."""
import importlib.util
import json
from pathlib import Path
import shutil
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('activate_native',ROOT/'scripts/activate_native_tuning.py')
activation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(activation)


@pytest.fixture
def installation(tmp_path):
    project=tmp_path/'project'
    package=project/'package'
    for d in ['scripts','patches','libs/team-gm/scripts','libs/team-gm/patches','package/kernels','build']:
        (project/d).mkdir(parents=True)
    (package/'kernels/kernel.py').write_text('math is unchanged\n')
    (package/'marker.py').write_text('base\n')
    installer=(ROOT/'scripts/apply_engine_audit_patches.py').read_text()
    names,node=activation.installer_names(installer)
    lines=installer.splitlines(keepends=True)
    installer=''.join(lines[:node.lineno-1])+'PATCH_NAMES = ()\n'+''.join(lines[node.end_lineno:])
    for d in ['scripts','libs/team-gm/scripts']:
        (project/d/'apply_engine_audit_patches.py').write_text(installer)
    patch='--- a/src/miniworld_engine/marker.py\n+++ b/src/miniworld_engine/marker.py\n@@ -1 +1 @@\n-base\n+new\n'
    for d in ['patches','libs/team-gm/patches']:
        (project/d/activation.PATCH).write_text(patch)
    manifest={'patch_sha256':activation.digest(patch.encode()),
              'kernels':activation.kernel_identity(package),
              'files':{'marker.py':{'before':activation.digest(b'base\n'),'after':activation.digest(b'new\n')}},
              'native_ops':['native'],'before_identity':'old','after_identity':'new'}
    (project/'patches/native-tuning-manifest.json').write_text(json.dumps(manifest))
    native=package/'autotune/data/native/H100.json'
    native.parent.mkdir(parents=True)
    native.write_text(json.dumps({'op_identity':'old','entries':{'shape':[{'ms':1} ]}})+'\n')
    triton=package/'autotune/data/triton/H100.json'
    triton.parent.mkdir(parents=True)
    triton.write_text('Triton cache must stay byte identical\n')
    return project,package,native,triton


def test_running_build_blocks_activation(installation):
    project,package,native,triton=installation
    with pytest.raises(RuntimeError,match='has not exited'):
        activation.activate(project,package,after_build=project/'build')
    assert (package/'marker.py').read_text()=='base\n'
    assert json.loads(native.read_text())['op_identity']=='old'


def test_completed_build_activation_and_idempotent_reinstall(installation):
    project,package,native,triton=installation
    before=triton.read_bytes()
    (project/'build/job_exit.json').write_text('{"exit_code":0}')
    result=activation.activate(project,package,after_build=project/'build')
    assert result['compatible_cache_files']==1
    assert (package/'marker.py').read_text()=='new\n'
    assert triton.read_bytes()==before
    data=json.loads(native.read_text())
    assert data['op_identity']=='new'
    assert data['entries']['shape'][0]['ms']==1
    assert data['provenance']['native_identity_migration']['timing_profile']=='legacy-unattributed'
    assert activation.activate(project,package)['already_active']
    installer=project/'scripts/apply_engine_audit_patches.py'
    names,_=activation.installer_names(installer.read_text())
    spec=importlib.util.spec_from_file_location('applier',installer)
    applier=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(applier)
    assert applier.apply_patches(package,project/'patches',names)==[]
    assert installer.read_bytes()==(project/'libs/team-gm/scripts/apply_engine_audit_patches.py').read_bytes()


def test_kernel_edit_refuses_compatibility_migration(installation):
    project,package,native,_=installation
    (package/'kernels/kernel.py').write_text('different math\n')
    with pytest.raises(RuntimeError,match='kernel/launcher'):
        activation.activate(project,package)
    assert json.loads(native.read_text())['op_identity']=='old'


def test_check_only_preserves_all_files(installation):
    project,package,native,triton=installation
    before={str(p):p.read_bytes() for p in project.rglob('*') if p.is_file()}
    result=activation.activate(project,package,check_only=True)
    assert result['check_only']
    assert {str(p):p.read_bytes() for p in project.rglob('*') if p.is_file() and '__pycache__' not in str(p)}==before


def test_unterminated_native_caches_with_spaces(installation):
    project, package, native, _ = installation
    raw = native.read_text().rstrip('\n')
    native.write_text(raw)
    second = native.with_name('NVIDIA H100 80GB HBM3 (sm90).json')
    second.write_text(raw)
    result = activation.activate(project, package)
    assert result['compatible_cache_files'] == 2
    for path in (native, second):
        assert json.loads(path.read_text())['op_identity'] == 'new'
