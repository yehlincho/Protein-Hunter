import os
import re
import subprocess
import tempfile
import time
import logging
import string
from typing import List, Tuple

# --- Helper Functions for MSA Processing ---
def run_command(cmd: List[str], cmd_name: str):
    """Run a command and handle errors"""
    logging.info(f"Running {cmd_name}: {' '.join(cmd)}")
    start_time = time.time()
    try:
        completed_process = subprocess.run(
            cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
    except subprocess.CalledProcessError as e:
        logging.exception(f"{cmd_name} failed.\nstdout: {e.stdout}\nstderr: {e.stderr}")
        raise RuntimeError(
            f"{cmd_name} failed\nstdout: {e.stdout}\nstderr: {e.stderr}"
        ) from e

    end_time = time.time()
    logging.info(f"Finished {cmd_name} in {end_time - start_time:.3f} seconds")
    return completed_process


def create_query_fasta_file(sequence: str, path: str, linewidth: int = 80):
    """Creates a fasta file with the sequence"""
    with open(path, "w") as f:
        f.write(">query\n")
        i = 0
        while i < len(sequence):
            f.write(f"{sequence[i : (i + linewidth)]}\n")
            i += linewidth


def parse_fasta(fasta_string: str) -> Tuple[List[str], List[str]]:
    """Parse a FASTA string into sequences and descriptions"""
    sequences = []
    descriptions = []

    lines = fasta_string.strip().split("\n")
    current_seq = ""
    current_desc = ""

    for line in lines:
        if line.startswith(">"):
            if current_seq:  # Save the previous sequence
                sequences.append(current_seq)
                descriptions.append(current_desc)
            current_desc = line[1:].strip()  # Remove the '>' character
            current_seq = ""
        else:
            current_seq += line.strip()

    # Add the last sequence
    if current_seq:
        sequences.append(current_seq)
        descriptions.append(current_desc)

    return sequences, descriptions


def convert_stockholm_to_a3m(stockholm_path: str, max_sequences: int = None) -> str:
    """
    Convert Stockholm format MSA to A3M format (simplified alignment logic).
    Returns an A3M formatted string.
    """
    with open(stockholm_path) as stockholm_file:
        descriptions = {}
        sequences = {}
        reached_max_sequences = False

        # Pass 1: extract sequences
        for line in stockholm_file:
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            
            # Use split to handle multiple spaces/tabs
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue

            seqname, aligned_seq = parts
            
            # Check max sequences before adding
            if max_sequences and len(sequences) >= max_sequences:
                reached_max_sequences = True
                if seqname not in sequences:
                    continue # Skip new sequence if max reached
            
            sequences.setdefault(seqname, "")
            sequences[seqname] += aligned_seq.replace(".", "-") # Replace '.' with '-' for consistency

        if not sequences:
            return ""

        # Pass 2: extract descriptions
        stockholm_file.seek(0)
        for line in stockholm_file:
            line = line.strip()
            if line.startswith("#=GS"):
                columns = line.split(maxsplit=3)
                if len(columns) < 3: continue
                seqname, feature = columns[1:3]
                value = columns[3] if len(columns) == 4 else ""
                if feature != "DE":
                    continue
                if reached_max_sequences and seqname not in sequences:
                    continue
                descriptions[seqname] = value
                if len(descriptions) == len(sequences):
                    break

    # Convert Stockholm to A3M
    a3m_sequences = {}
    
    # The 'query' in A3M is the first sequence in the stockholm file which is the query sequence
    query_name = next(iter(sequences.keys()))
    query_alignment = sequences[query_name]

    for seqname, sto_sequence in sequences.items():
        a3m_seq = ""
        for q_char, s_char in zip(query_alignment, sto_sequence):
            if q_char == '-':
                # Deletion in query, insertion in subject (lowercase)
                if s_char != '-':
                    a3m_seq += s_char.lower()
            else:
                # Match/Mismatch/Deletion
                a3m_seq += s_char
        
        # Remove original query gaps and any '.' characters
        a3m_sequences[seqname] = a3m_seq.replace('-', '')
        
    # Convert to FASTA format
    fasta_chunks = []
    for seqname, seq in a3m_sequences.items():
        # Replace the first sequence name (the HMMER query) with ">query"
        if seqname == query_name:
            fasta_chunks.append(">query")
        else:
            fasta_chunks.append(f">{seqname} {descriptions.get(seqname, '')}")
        fasta_chunks.append(seq)

    return "\n".join(fasta_chunks) + "\n"


# --- HMMER Wrapper Classes ---

class Hmmbuild:
    """Python wrapper for hmmbuild - construct HMM profiles from MSA"""

    def __init__(self, binary_path: str, alphabet: str = None):
        """Initialize Hmmbuild wrapper"""
        self.binary_path = binary_path
        self.alphabet = alphabet

    def build_profile_from_fasta(self, fasta: str) -> str:
        """Build an HMM profile from a FASTA string"""
        # Process FASTA to remove inserted residues (lowercase letters)
        sequences, descriptions = parse_fasta(fasta)
        lines = []
        # A replacement table that removes all lowercase characters (insertions)
        deletion_table = str.maketrans("", "", string.ascii_lowercase)
        
        for seq, desc in zip(sequences, descriptions):
            # Remove inserted residues (lowercase)
            seq = seq.translate(deletion_table)
            lines.append(f">{desc}\n{seq}\n")
        msa = "".join(lines)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_msa_path = os.path.join(tmp_dir, "query.msa")
            output_hmm_path = os.path.join(tmp_dir, "output.hmm")

            with open(input_msa_path, "w") as f:
                f.write(msa)

            # Prepare command
            cmd_flags = ["--informat", "afa"]
            if self.alphabet:
                cmd_flags.append(f"--{self.alphabet}")

            cmd_flags.extend([output_hmm_path, input_msa_path])
            cmd = [self.binary_path, *cmd_flags]

            # Run hmmbuild
            run_command(cmd=cmd, cmd_name="Hmmbuild")

            # Read the output profile
            with open(output_hmm_path) as f:
                hmm = f.read()

            return hmm


class Hmmalign:
    """Python wrapper of the hmmalign binary"""

    def __init__(self, binary_path: str):
        """Initialize Hmmalign wrapper"""
        self.binary_path = binary_path

    def align_sequences_to_profile(self, profile: str, sequences_a3m: str) -> str:
        """Align sequences to a profile and return in A3M format"""

        # Process A3M to remove gaps (essential for HMMER input)
        sequences, descriptions = parse_fasta(sequences_a3m)
        lines = []
        for seq, desc in zip(sequences, descriptions):
            # Remove gaps
            seq = seq.replace("-", "")
            lines.append(f">{desc}\n{seq}\n")
        sequences_no_gaps_a3m = "".join(lines)

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_profile = os.path.join(tmp_dir, "profile.hmm")
            input_seqs = os.path.join(tmp_dir, "sequences.a3m")
            output_a3m_path = os.path.join(tmp_dir, "output.a3m")

            with open(input_profile, "w") as f:
                f.write(profile)

            with open(input_seqs, "w") as f:
                f.write(sequences_no_gaps_a3m)

            # Prepare command
            cmd = [
                self.binary_path,
                "-o",
                output_a3m_path,
                "--outformat",
                "A2M",  # A2M is A3M in the HMMER suite
                input_profile,
                input_seqs,
            ]

            # Run hmmalign
            run_command(cmd=cmd, cmd_name="Hmmalign")

            # Read the aligned output
            with open(output_a3m_path, encoding="utf-8") as f:
                a3m = f.read()

            return a3m


class Nhmmer:
    """Python wrapper of the Nhmmer binary"""

    def __init__(
        self,
        binary_path: str,
        hmmalign_binary_path: str,
        hmmbuild_binary_path: str,
        database_path: str,
        n_cpu: int = 8,
        e_value: float = 1e-3,
        max_sequences: int = 10000,
        alphabet: str = "rna",
        time_limit_minutes: float = None,
    ):
        """Initialize Nhmmer wrapper"""
        self.binary_path = binary_path
        self.hmmalign_binary_path = hmmalign_binary_path
        self.hmmbuild_binary_path = hmmbuild_binary_path
        self.db_path = database_path
        self.e_value = e_value
        self.n_cpu = n_cpu
        self.max_sequences = max_sequences
        self.alphabet = alphabet
        self.time_limit_seconds = (
            time_limit_minutes * 60 if time_limit_minutes else None
        )

    def query(self, target_sequence: str) -> str:
        """Query the database using Nhmmer and return results in A3M format"""
        from tqdm import tqdm
        from boltz_ph.constants import SHORT_SEQUENCE_CUTOFF

        logging.info(f"Querying database with sequence: {target_sequence[:20]}...")

        with tempfile.TemporaryDirectory() as query_tmp_dir:
            input_fasta_path = os.path.join(query_tmp_dir, "query.fasta")
            output_sto_path = os.path.join(query_tmp_dir, "output.sto")

            # Create query FASTA file
            create_query_fasta_file(sequence=target_sequence, path=input_fasta_path)

            # Prepare Nhmmer command
            cmd_flags = [
                "-o", "/dev/null", "--noali", "--cpu", str(self.n_cpu), "-E", str(self.e_value), 
                "-A", output_sto_path
            ]

            # Add alphabet flag
            if self.alphabet:
                cmd_flags.extend([f"--{self.alphabet}"])

            # Special handling for short RNA sequences
            if self.alphabet == "rna" and len(target_sequence) < SHORT_SEQUENCE_CUTOFF:
                cmd_flags.extend(["--F3", str(0.02)])
            else:
                cmd_flags.extend(["--F3", str(1e-5)])

            # Add input and database paths
            cmd_flags.extend([input_fasta_path, self.db_path])

            cmd = [self.binary_path, *cmd_flags]
            start_time = time.time()
            
            # --- Timeout and Execution Logic ---
            try:
                # Use subprocess with timeout
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )

                if self.time_limit_seconds is None:
                    # No time limit: wait indefinitely
                    process.wait()
                else:
                    # Time limit: poll with a progress bar
                    with tqdm(
                        total=self.time_limit_seconds, desc="Search time", unit="sec", leave=False
                    ) as pbar:
                        elapsed = 0
                        while process.poll() is None and elapsed < self.time_limit_seconds:
                            time.sleep(1)
                            elapsed = time.time() - start_time
                            pbar.update(1)
                        
                        if process.poll() is None:
                            print(f"⚠️ Time limit reached ({self.time_limit_seconds} seconds). Terminating search.")
                            process.terminate()
                            process.wait()

                # Get stdout/stderr
                stdout, stderr = process.communicate()
                
                # Check for unexpected termination/failure
                if process.returncode != 0 and not (os.path.exists(output_sto_path) and os.path.getsize(output_sto_path) > 0):
                    print(f"❌ Nhmmer failed with error: {stderr}")
                    return f">query\n{target_sequence}"

            except Exception as e:
                print(f"❌ Error running Nhmmer: {e}")
                return f">query\n{target_sequence}"
            
            # --- Result Processing ---
            if os.path.exists(output_sto_path) and os.path.getsize(output_sto_path) > 0:
                # Build HMM profile from query sequence (needed for hmmalign)
                hmmbuild = Hmmbuild(
                    binary_path=self.hmmbuild_binary_path, alphabet=self.alphabet
                )
                target_sequence_fasta = f">query\n{target_sequence}\n"
                profile = hmmbuild.build_profile_from_fasta(target_sequence_fasta)

                # Convert Stockholm to A3M
                a3m_out = convert_stockholm_to_a3m(
                    output_sto_path, max_sequences=self.max_sequences - 1
                )

                # Align hits to the query profile
                aligner = Hmmalign(binary_path=self.hmmalign_binary_path)
                aligned_a3m = aligner.align_sequences_to_profile(
                    profile=profile, sequences_a3m=a3m_out
                )

                # Return A3M with query sequence first
                return "".join([target_sequence_fasta, aligned_a3m])
            
            # No hits - return only query sequence
            return f">query\n{target_sequence}"


class Msa:
    """Multiple Sequence Alignment container with methods for manipulating it"""

    def __init__(
        self, query_sequence, chain_poly_type, sequences, descriptions, deduplicate=True
    ):
        """Initialize MSA container"""
        if len(sequences) != len(descriptions):
            raise ValueError("The number of sequences and descriptions must match.")

        self.query_sequence = query_sequence
        self.chain_poly_type = chain_poly_type

        if not deduplicate:
            self.sequences = sequences
            self.descriptions = descriptions
        else:
            self.sequences = []
            self.descriptions = []
            # A replacement table that removes all lowercase characters (insertions)
            deletion_table = str.maketrans("", "", string.ascii_lowercase)
            unique_sequences = set()
            for seq, desc in zip(sequences, descriptions):
                # Only compare uppercase part (removes insertions for uniqueness check)
                sequence_no_deletions = seq.translate(deletion_table)
                if sequence_no_deletions not in unique_sequences:
                    unique_sequences.add(sequence_no_deletions)
                    self.sequences.append(seq)
                    self.descriptions.append(desc)

        # Ensure MSA always has at least the query
        if not self.sequences or not self._sequences_are_feature_equivalent(self.sequences[0], query_sequence):
            # Prepend or ensure query sequence is present
            if self.sequences and self._sequences_are_feature_equivalent(self.sequences[0], query_sequence):
                pass
            else:
                self.sequences.insert(0, query_sequence)
                self.descriptions.insert(0, "Original query")

    def _sequences_are_feature_equivalent(self, sequence1, sequence2):
        """Check if two sequences are equivalent (ignoring insertions)"""
        # For RNA, we can simply compare the uppercase versions
        from boltz_ph.constants import RNA_CHAIN_POLY_TYPE
        
        if self.chain_poly_type == RNA_CHAIN_POLY_TYPE:
            seq1_upper = re.sub("[a-z]+", "", sequence1)
            seq2_upper = re.sub("[a-z]+", "", sequence2)
            return seq1_upper == seq2_upper
        return sequence1 == sequence2 # Fallback for other types

    @classmethod
    def from_multiple_msas(cls, msas: List['Msa'], deduplicate: bool = True) -> 'Msa':
        """Initialize MSA from multiple MSAs"""
        if not msas:
            raise ValueError("At least one MSA must be provided.")

        query_sequence = msas[0].query_sequence
        chain_poly_type = msas[0].chain_poly_type
        sequences = []
        descriptions = []

        for msa in msas:
            if msa.query_sequence != query_sequence:
                raise ValueError(
                    f"Query sequences must match: {[m.query_sequence for m in msas]}"
                )
            if msa.chain_poly_type != chain_poly_type:
                raise ValueError(
                    f"Chain poly types must match: {[m.chain_poly_type for m in msas]}"
                )
            # Skip the first sequence (query) if not the first MSA to avoid duplication before deduplication
            start_index = 1 if msa != msas[0] else 0
            sequences.extend(msa.sequences[start_index:])
            descriptions.extend(msa.descriptions[start_index:])
            
        # Ensure query is re-added at the start if it was removed
        if query_sequence not in sequences:
            sequences.insert(0, query_sequence)
            descriptions.insert(0, "Original query")

        return cls(
            query_sequence=query_sequence,
            chain_poly_type=chain_poly_type,
            sequences=sequences,
            descriptions=descriptions,
            deduplicate=deduplicate,
        )

    @classmethod
    def from_a3m(
        cls, query_sequence: str, chain_poly_type: str, a3m: str, max_depth: int = None, deduplicate: bool = True
    ) -> 'Msa':
        """Parse a single A3M and build the Msa object"""
        sequences, descriptions = parse_fasta(a3m)

        if max_depth is not None and 0 < max_depth < len(sequences):
            print(
                f"MSA cropped from depth of {len(sequences)} to {max_depth} to save memory."
            )
            sequences = sequences[:max_depth]
            descriptions = descriptions[:max_depth]

        return cls(
            query_sequence=query_sequence,
            chain_poly_type=chain_poly_type,
            sequences=sequences,
            descriptions=descriptions,
            deduplicate=deduplicate,
        )

    @property
    def depth(self):
        """Return the number of sequences in the MSA"""
        return len(self.sequences)

    def to_a3m(self) -> str:
        """Return the MSA in A3M format"""
        a3m_lines = []
        for desc, seq in zip(self.descriptions, self.sequences):
            a3m_lines.append(f">{desc}")
            a3m_lines.append(seq)
        return "\n".join(a3m_lines) + "\n"