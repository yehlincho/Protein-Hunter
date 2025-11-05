"""
Central place for global constants, chain mappings, and amino acid conversions.
"""

# Map PDB chain ID (A, B, C...) to Boltz-style internal chain index (0, 1, 2...)
# Max 10 chains supported here.
CHAIN_TO_NUMBER = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
}

# Amino acid 3-letter to 1-letter code conversion dict
RESTYPE_3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "MSE": "M",  # Selenomethionine maps to Methionine
}

# RNA sequence type constant
RNA_CHAIN_POLY_TYPE = "polyribonucleotide"

# Cutoff for short RNA sequences in Nhmmer
SHORT_SEQUENCE_CUTOFF = 50

# Hydrophobic amino acids set for scoring
HYDROPHOBIC_AA = set("ACFILMPVWY")