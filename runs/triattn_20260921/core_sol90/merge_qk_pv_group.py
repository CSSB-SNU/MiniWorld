"""Keep QK/E/PV issue order, commit QK+PV together after E, wait one group.

One body group contains QK(k+2) and PV(k-1), so wait1 at body k retires
QK(k) and PV(k-3), exactly the old two-commit schedule's wait2 frontier.
Prologue and terminal drain paths keep their original commits.
"""
def transform(name,s):
 if name!='triattn_m1_sm90.cuh':return s
 a=s.index('    auto issue_qk =');b=s.index('    auto issue_pv =',a)
 helper=s[a:b].replace('auto issue_qk =','auto issue_qk_nocommit =')
 helper=helper.replace('        warpgroup_commit_batch();','        // commit deferred until this body\'s PV issue:')
 s=s[:b]+helper+s[b:]
 a=s.index('    auto body =');b=s.index('    // drained state:',a)
 block=s[a:b]
 block=block.replace('warpgroup_wait<kQK1 ? 1 : 2>();','warpgroup_wait<kFast || kQK1 ? 1 : 2>();')
 old='issue_qk(accC[bn], tK, Int<h2>{}, Int<c2>{}, Int<(t2 & 1) * R>{});'
 assert block.count(old)==1
 block=block.replace(old,'if constexpr(kFast) { '+old.replace('issue_qk(','issue_qk_nocommit(')+' } else { '+old+' }')
 return s[:a]+block+s[b:]
