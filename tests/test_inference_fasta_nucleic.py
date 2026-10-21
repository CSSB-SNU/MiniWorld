"""Nucleic fasta bodies: parenthesized canonical CCD codes keep their 1-letter code (FoldBench DNA is written as ``(DC)(DG)...``)."""

from pathlib import Path

from miniworld.data.inference.fasta import EntityType, parse_fasta_file


def _write(tmp_path: Path, kind: str, body: str) -> Path:
    p = tmp_path / "chain_a.fasta"
    p.write_text(f"> t_a | {kind} | Chain:a\n{body}\n")
    return p


def test_dna_parenthesized_canonical_bases(tmp_path):
    cs = parse_fasta_file(_write(tmp_path, "polydeoxyribonucleotide", "(DC)(DG)(DT)(DA)(8NI)"), 0)
    assert cs.entity_type == EntityType.DNA
    assert cs.chemcomp_ids == ["DC", "DG", "DT", "DA", "8NI"]
    assert cs.one_letter_seq == ["C", "G", "T", "A", "X"]


def test_dna_plain_letters_unchanged(tmp_path):
    cs = parse_fasta_file(_write(tmp_path, "polydeoxyribonucleotide", "CGTA"), 0)
    assert cs.chemcomp_ids == ["DC", "DG", "DT", "DA"]
    assert cs.one_letter_seq == ["C", "G", "T", "A"]


def test_rna_parenthesized(tmp_path):
    cs = parse_fasta_file(_write(tmp_path, "polyribonucleotide", "AC(PSU)(G)"), 0)
    assert cs.chemcomp_ids == ["A", "C", "PSU", "G"]
    assert cs.one_letter_seq == ["A", "C", "X", "G"]
