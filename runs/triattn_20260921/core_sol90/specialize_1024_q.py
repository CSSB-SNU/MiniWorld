"""Extend the installed exact pipeline to a separately guarded L1024 flag."""
def transform(name, s):
    if name == 'build_source.py':
        old = 'return [0, 1024, 1073741824]'
        assert s.count(old) == 1
        return s.replace(old, 'return [0, 1024, 536870912, 1073741824]')
    if name == 'm1_binding.cu':
        old = '        hot=table().find(1073741824);'
        assert s.count(old) == 1
        return s.replace(old, old + '''
    if(flags==0 && q.size(1)==1024 && q.size(2)==4 && q.size(3)==1024 && float(scale)==0x1.6a09e6p-3f)
        hot=table().find(536870912);''')
    if name != 'triattn_m1_sm90.cuh':
        return s
    old = '(kFlags_ & 1073741824) != 0'
    assert s.count(old) == 1
    s = s.replace(old, '(kFlags_ & (1073741824 | 536870912)) != 0')
    old = '    constexpr bool kFast = !T::kSafe && (T::kFlags & 1073741824) != 0;'
    assert s.count(old) == 1
    s = s.replace(old, '''    constexpr bool kFast = !T::kSafe && (T::kFlags & (1073741824 | 536870912)) != 0;
    constexpr int kShapeN = (T::kFlags & 536870912) ? 1024 : 768;''')
    assert s.count(': 768)') == 10
    assert s.count('params.n_ktiles : 6)') == 2
    s = s.replace(': 768)', ': kShapeN)')
    s = s.replace('params.n_ktiles : 6)', 'params.n_ktiles : kShapeN / 128)')
    return s
