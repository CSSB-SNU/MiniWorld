from clamp.models.pretrained import PretrainedCLAMP
import torch

smiles_dict = {
    "ALA": "C[C@H](N)C(=O)O",
    "ARG": "NC(=N)NCCC[C@H](N)C(=O)O",
    "ASN": "NC(=O)C[C@H](N)C(=O)O",
    "ASP": "OC(=O)C[C@H](N)C(=O)O",
    "CYS": "C([C@@H](N)C(=O)O)S",
    "GLN": "NC(=O)CC[C@H](N)C(=O)O",
    "GLU": "OC(=O)CC[C@H](N)C(=O)O",
    "GLY": "NCC(=O)O",
    "HIS": "c1c(nc[nH]1)C[C@H](N)C(=O)O",
    "ILE": "CC[C@H](C)[C@H](N)C(=O)O",
    "LEU": "CC(C)C[C@H](N)C(=O)O",
    "LYS": "NCCCC[C@H](N)C(=O)O",
    "MET": "CSCC[C@H](N)C(=O)O",
    "PHE": "c1ccc(cc1)C[C@H](N)C(=O)O",
    "PRO": "OC(=O)[C@@H]1CCCN1",
    "SER": "OC[C@H](N)C(=O)O",
    "THR": "C[C@@H](O)[C@H](N)C(=O)O",
    "TRP": "c1ccc2c(c1)c(c[nH]2)C[C@H](N)C(=O)O",
    "TYR": "Oc1ccc(cc1)C[C@H](N)C(=O)O",
    "VAL": "CC(C)[C@H](N)C(=O)O",
    "A": "C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)O)O)O)N",
    "U": "C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)O)O)O",
    "G": "C1=NC2=C(N1[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)O)O)O)NC(=NC2=O)N",
    "C": "C1=CN(C(=O)N=C1N)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)O)O)O",
    "dA": "c1nc(c2c(n1)n(cn2)[C@H]3C[C@@H]([C@H](O3)COP(=O)(O)O)O)N",
    "dT": "CC1=CN(C(=O)NC1=O)[C@H]2C[C@@H]([C@H](O2)COP(=O)(O)O)O",
    "dG": "c1nc2c(n1[C@H]3C[C@@H]([C@H](O3)COP(=O)(O)O)O)nc(nc2O)N",
    "dC": "C1=CN(C(=O)N=C1N)[C@H]2C[C@@H]([C@H](O2)COP(=O)(O)O)O",
}

smiles_list = list(smiles_dict.values())
model = PretrainedCLAMP(device="cpu")
model.eval()

# 1. Generate the initial 28 embeddings (20 AA, 4 RNA, 4 DNA)
emb = model.encode_smiles(smiles_list)

# 2. Slice the embeddings for mean calculations
# Indices: 0-19 (AA), 20-23 (RNA), 24-27 (DNA)
aa_embs = emb[0:20]
rna_embs = emb[20:24]
dna_embs = emb[24:28]

# 3. Calculate "Unknown" embeddings (Average)
unk_aa = torch.mean(aa_embs, dim=0, keepdim=True)
print(unk_aa)
unk_rna = torch.mean(rna_embs, dim=0, keepdim=True)
unk_dna = torch.mean(dna_embs, dim=0, keepdim=True)

# 4. Create the "Gap" token (Zero tensor)
gap_token = torch.zeros((1, emb.shape[1]), device=emb.device)

# 5. Concatenate in the requested order:
# [20 AA] + [Unk AA] + [4 RNA] + [Unk RNA] + [4 DNA] + [Unk DNA] + [Gap]
final_emb = torch.cat([
    aa_embs,   # 0-19
    unk_aa,    # 20
    rna_embs,  # 21-24
    unk_rna,   # 25
    dna_embs,  # 26-29
    unk_dna,   # 30
    gap_token  # 31
], dim=0)

print(f"Original shape: {emb.shape}")
print(f"Final shape: {final_emb.shape}") # Should be (32, embedding_size)
print(final_emb[-1,:])
torch.save(final_emb, 'fp_clamp_embeddings.pt')
