"""
================================================================================
Quantum Figure Caption Extractor - Research Pipeline Step 2
================================================================================

PIPELINE WORKFLOW:
--------------------------------------------------
STEP 1: Download HTML → download_arxiv.py → downloaded_html/
STEP 2: Extract Figures → THIS SCRIPT → Figure_JSON/
STEP 3: Train BERT → Main Notebook → Model Output/

USAGE:
1. Run: python html_figure_extractor.py
2. Requires: downloaded_html/ folder (from Step 1)
3. Output: Figure_JSON/ with JSON files per figure

REQUIREMENTS:
pip install beautifulsoup4 lxml

SAMPLE OUTPUT:
Figure_JSON/
└── 00001_arXiv_2301.12345/
    ├── 2301.12345_fig_1.json
    ├── 2301.12345_fig_2.json
    └── ...

JSON FORMAT:
{
  "arxiv_number": "2301.12345",
  "figure_number": 1,
  "quantum_gates": ["CNOT", "RZ", "H"],
  "quantum_problem": ["QAOA"],
  "raw_caption": "Circuit with CNOT and RZ gates...",
  "description": "First sentence of caption"
}

FEATURES:
✓ Detects 50+ quantum gates (RX, RZZ, Toffoli, iSWAP...)
✓ 40+ algorithms (QAOA, VQE, QML, Grover's...)
✓ Resumable processing with checkpoint
✓ Handles LaTeX/HTML artifacts automatically

================================================================================
"""


import os
import re
import json
import time
from bs4 import BeautifulSoup, NavigableString
from collections import OrderedDict

# ============ CONFIGURATION ============
# All paths are relative - no changes needed
HTML_DIR = "downloaded_html"                    # Input: HTML files from Step 1
OUT_ROOT = "Figure_JSON"                       # Output: Extracted figures as JSON
CHECKPOINT_FILE = "extraction_checkpoint.json" # Progress tracking
PRINT_EVERY = 50                               # Progress update frequency
# ============ END CONFIGURATION ============

os.makedirs(OUT_ROOT, exist_ok=True)


# ============================================
# IMPROVED TEXT NORMALIZATION
# ============================================
def clean_unicode_artifacts(text):
    """
    Remove Unicode artifacts that interfere with pattern matching.
    This is the FIRST step before any other processing.
    """
    if not text:
        return ""
    
    # Remove invisible Unicode characters
    unicode_to_remove = [
        '\u2062',  # Invisible times (⁢)
        '\u2061',  # Function application
        '\u2063',  # Invisible separator
        '\u2064',  # Invisible plus
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # BOM
        '\u00a0',  # Non-breaking space -> regular space
    ]
    
    for char in unicode_to_remove:
        if char == '\u00a0':
            text = text.replace(char, ' ')
        else:
            text = text.replace(char, '')
    
    return text


def normalize_gate_notation(text):
    """
    Normalize gate notation for consistent pattern matching.
    Handles: R z -> Rz, R _ z -> Rz, R(Z) -> RZ, etc.
    """
    # First clean Unicode
    text = clean_unicode_artifacts(text)
    
    # Normalize rotation gates: R z, R_z, R-z, R (z) -> Rz
    # Handle R followed by space/underscore and single letter
    text = re.sub(r'\bR\s*[_\-]?\s*([xyzXYZ])\b', r'R\1', text)
    
    # Handle R ( X ), R(X), R ⁢ ( X ) -> RX
    text = re.sub(r'\bR\s*\(?\s*([XYZ])\s*\)?', r'R\1', text, flags=re.IGNORECASE)
    
    # Handle RXX, RYY, RZZ with spaces: R X X, R_XX, R(XX) -> RXX
    text = re.sub(r'\bR\s*[_\-]?\s*([XYZ])\s*([XYZ])\b', r'R\1\2', text, flags=re.IGNORECASE)
    text = re.sub(r'\bR\s*\(?\s*([XYZ])\s*([XYZ])\s*\)?', r'R\1\2', text, flags=re.IGNORECASE)
    
    # Handle ZZ feature map: Z Z -> ZZ
    text = re.sub(r'\bZ\s+Z\b', 'ZZ', text)
    text = re.sub(r'\bX\s+X\b', 'XX', text)
    text = re.sub(r'\bY\s+Y\b', 'YY', text)
    
    # Normalize CNOT: C N O T, C-NOT -> CNOT
    text = re.sub(r'\bC\s*N\s*O\s*T\b', 'CNOT', text, flags=re.IGNORECASE)
    text = re.sub(r'\bC\s*-\s*NOT\b', 'CNOT', text, flags=re.IGNORECASE)
    
    # Normalize other gates with spaces
    text = re.sub(r'\bS\s*W\s*A\s*P\b', 'SWAP', text, flags=re.IGNORECASE)
    text = re.sub(r'\bi\s*S\s*W\s*A\s*P\b', 'iSWAP', text, flags=re.IGNORECASE)
    
    # Clean multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text


def normalize_for_pattern_matching(text):
    """
    Full normalization pipeline for pattern matching.
    """
    text = clean_unicode_artifacts(text)
    text = normalize_gate_notation(text)
    
    # Normalize controlled gates
    text = re.sub(r'controlled\s*-?\s*NOT', 'controlled-NOT', text, flags=re.IGNORECASE)
    text = re.sub(r'controlled\s*-?\s*([XYZ])', r'controlled-\1', text, flags=re.IGNORECASE)
    text = re.sub(r'controlled\s*-?\s*phase', 'controlled-phase', text, flags=re.IGNORECASE)
    
    return text.strip()


# ============================================
# QUANTUM GATES - IMPROVED Detection Patterns
# ============================================
"""
Comprehensive regex patterns for 50+ quantum gates across:
- Single-qubit: X, Y, Z, H, S, T
- Rotations: RX, RY, RZ, RXX, RYY, RZZ
- Two-qubit: CNOT, CZ, SWAP, iSWAP
- Three-qubit: Toffoli, CCX, CSWAP
- Special: MS (Mølmer-Sørensen), ECR
"""
QUANTUM_GATES_PATTERNS = [
    # === SINGLE-QUBIT PAULI GATES ===
    # X gate - multiple patterns
    ("X", r'\bX\s*[-]?\s*gates?\b'),
    ("X", r'\bX\s*[-]?\s*operators?\b'),
    ("X", r'\bPauli\s*[-]?\s*X\b'),
    ("X", r'\bbit\s*[-]?\s*flip\b'),
    ("X", r'\bNOT\s+gate\b'),
    ("X", r'\bσ[_]?x\b'),
    ("X", r'\bX\s+and\s+[YZ]\s+gates?\b'),  # 'X and Z gates'
    ("X", r'\b[YZ]\s+and\s+X\s+gates?\b'),  # 'Z and X gates'
    ("X", r'\bH\s*,\s*Z\s*,\s*X\b'),  # 'H, Z, X' pattern
    
    # Y gate
    ("Y", r'\bY\s*[-]?\s*gates?\b'),
    ("Y", r'\bY\s*[-]?\s*operators?\b'),
    ("Y", r'\bPauli\s*[-]?\s*Y\b'),
    ("Y", r'\bσ[_]?y\b'),
    
    # Z gate
    ("Z", r'\bZ\s*[-]?\s*gates?\b'),
    ("Z", r'\bZ\s*[-]?\s*operators?\b'),
    ("Z", r'\bPauli\s*[-]?\s*Z\b'),
    ("Z", r'\bphase\s*[-]?\s*flip\b'),
    ("Z", r'\bσ[_]?z\b'),
    ("Z", r'\bX\s+and\s+Z\s+gates?\b'),  # 'X and Z gates'
    ("Z", r'\bZ\s+and\s+X\s+gates?\b'),
    ("Z", r'\bH\s*,\s*Z\s*,\s*X\b'),  # 'H, Z, X' pattern
    
    # === HADAMARD ===
    ("H", r'\bHadamard\b'),
    ("H", r'\bH\s*[-]?\s*gates?\b'),
    ("H", r'\bH\s*,\s*Z\s*,\s*X\b'),  # 'H, Z, X' pattern
    ("H", r'\binitial\s+H(?:adamard)?\s+gates?\b'),
    
    # === PHASE GATES ===
    ("S", r'\bS\s*[-]?\s*gate\b'),
    ("S", r'\bS[†]\b'),
    ("S", r'\bSdg\b'),
    
    ("T", r'\bT\s*[-]?\s*gate\b'),
    ("T", r'\bT[†]\b'),
    ("T", r'\bTdg\b'),
    ("T", r'\bπ/8\s*gate\b'),
    
    # === ROTATION GATES (IMPROVED) ===
    # Matches: Rx, RX, R_x, R x, R(x), Rx(θ), R z, R_z, etc.
    ("RX", r'\bR\s*[_]?\s*[xX]\b'),  # R x, R_x, Rx
    ("RX", r'\bR\s*\(\s*[xX]\s*\)'),
    ("RX", r'\bX\s*[-]?\s*rotation\b'),
    
    ("RY", r'\bR\s*[_]?\s*[yY]\b'),  # R y, R_y, Ry
    ("RY", r'\bR\s*\(\s*[yY]\s*\)'),
    ("RY", r'\bY\s*[-]?\s*rotation\b'),
    
    ("RZ", r'\bR\s*[_]?\s*[zZ]\b'),  # R z, R_z, Rz
    ("RZ", r'\bR\s*\(\s*[zZ]\s*\)'),
    ("RZ", r'\bZ\s*[-]?\s*rotation\b'),
    
    # === TWO-QUBIT GATES ===
    # CNOT variations (including with spaces: C N O T)
    ("CNOT", r'\bCNOT\b'),
    ("CNOT", r'\bC\s*N\s*O\s*T\b'),  # C N O T with spaces
    ("CNOT", r'\bCX\b'),
    ("CNOT", r'\bcontrolled\s*[-]?\s*NOT\b'),
    ("CNOT", r'\bcontrolled\s*[-]?\s*X\b'),
    
    # CZ variations
    ("CZ", r'\bCZ\b'),
    ("CZ", r'\bcontrolled\s*[-]?\s*Z\b'),
    ("CZ", r'\bcontrolled\s*[-]?\s*phase\b'),
    
    # Other controlled gates
    ("CY", r'\bCY\b'),
    ("CY", r'\bcontrolled\s*[-]?\s*Y\b'),
    
    ("CH", r'\bCH\b'),
    ("CH", r'\bcontrolled\s*[-]?\s*Hadamard\b'),
    
    # SWAP gates
    ("SWAP", r'\bSWAP\b'),
    ("SWAP", r'\bswap\s+gate\b'),
    ("iSWAP", r'\biSWAP\b'),
    ("FSWAP", r'\bFSWAP\b'),
    ("FSWAP", r'\bfermionic\s+SWAP\b'),
    
    # === THREE-QUBIT GATES ===
    ("Toffoli", r'\bToffoli\b'),
    ("CCX", r'\bCCX\b'),
    ("CCX", r'\bCCNOT\b'),
    ("CCZ", r'\bCCZ\b'),
    ("Fredkin", r'\bFredkin\b'),
    ("CSWAP", r'\bCSWAP\b'),
    
    # === TWO-QUBIT ROTATION GATES (IMPROVED) ===
    # ZZ feature map, XX interactions, etc.
    ("RXX", r'\bR\s*[_]?\s*XX\b'),
    ("RXX", r'\bXX\s*[-]?\s*rotation\b'),
    ("RXX", r'\bXX\s+gate\b'),
    
    ("RYY", r'\bR\s*[_]?\s*YY\b'),
    ("RYY", r'\bYY\s*[-]?\s*rotation\b'),
    ("RYY", r'\bYY\s+gate\b'),
    
    ("RZZ", r'\bR\s*[_]?\s*ZZ\b'),
    ("RZZ", r'\bZZ\s*[-]?\s*rotation\b'),
    ("RZZ", r'\bZZ\s+gate\b'),
    ("RZZ", r'\bZZ\s+feature\s+map\b'),
    ("RZZ", r'\bIsing\s*[-]?\s*(?:coupling|interaction|gate)\b'),
    
    ("RZX", r'\bR\s*[_]?\s*ZX\b'),
    ("RZX", r'\bZX\s*[-]?\s*rotation\b'),
    
    # === PARAMETERIZED GATES ===
    ("U", r'\bU[123]?\s*[-]?\s*gate\b'),
    ("U", r'\bunitary\s+gate\b'),
    
    # === SPECIAL GATES ===
    ("SX", r'\bSX\b'),
    ("SX", r'\b√X\b'),
    ("SX", r'\bsqrt\s*[-]?\s*X\b'),
    
    ("ECR", r'\bECR\b'),
    ("ECR", r'\becho\s*[-]?\s*cross\s*[-]?\s*resonance\b'),
    
    # Ion trap gates
    ("MS", r'\bMølmer\s*[-]?\s*Sørensen\b'),
    ("MS", r'\bMolmer\s*[-]?\s*Sorensen\b'),
    ("MS", r'\bMS\s+gate\b'),
    
    # === MEASUREMENT ===
    ("Measure", r'\bmeasurement\s+gate\b'),
    ("Reset", r'\breset\s+gate\b'),
]

GATE_PATTERNS = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in QUANTUM_GATES_PATTERNS]


# ============================================
# QUANTUM ALGORITHMS - EXPANDED Detection
# ============================================
QUANTUM_ALGORITHMS_PATTERNS = [
    # === MAJOR ALGORITHMS ===
    ("Shor's algorithm", r"\bShor['']?s?\s*(?:algorithm|factoring)\b"),
    ("Grover's algorithm", r"\bGrover['']?s?\s*(?:algorithm|search)\b"),
    ("Deutsch-Jozsa", r"\bDeutsch\s*[-–]?\s*Jozsa\b"),
    ("Bernstein-Vazirani", r"\bBernstein\s*[-–]?\s*Vazirani\b"),
    ("Simon's algorithm", r"\bSimon['']?s?\s*(?:algorithm|problem)\b"),
    
    # === QFT ===
    ("QFT", r"\bQFT\b"),
    ("QFT", r"\b[Qq]uantum\s+[Ff]ourier\s+[Tt]ransform\b"),
    
    # === PHASE ESTIMATION ===
    ("QPE", r"\bQPE\b"),
    ("QPE", r"\b[Qq]uantum\s+[Pp]hase\s+[Ee]stimation\b"),
    
    # === VARIATIONAL ALGORITHMS ===
    ("VQE", r"\bVQE\b"),
    ("VQE", r"\b[Vv]ariational\s+[Qq]uantum\s+[Ee]igensolver\b"),
    ("QAOA", r"\bQAOA\b"),
    ("QAOA", r"\b[Qq]uantum\s+[Aa]pproximate\s+[Oo]ptimization\b"),
    ("VQA", r"\bVQA\b"),
    ("VQC", r"\bVQC\b"),
    ("VQC", r"\b[Vv]ariational\s+[Qq]uantum\s+[Cc]ircuit\b"),
    
    # === QUANTUM MACHINE LEARNING (EXPANDED) ===
    ("QML", r"\bQML\b"),
    ("QML", r"\b[Qq]uantum\s+[Mm]achine\s+[Ll]earning\b"),
    ("QNN", r"\bQNN\b"),
    ("QNN", r"\b[Qq]uantum\s+[Nn]eural\s+[Nn]etwork\b"),
    ("QCNN", r"\bQCNN\b"),
    ("PQC", r"\bPQC\b"),
    ("PQC", r"\b[Pp]arameteri[sz]ed\s+[Qq]uantum\s+[Cc]ircuit\b"),
    ("QSVM", r"\bQSVM\b"),
    ("QGAN", r"\bQGAN\b"),
    
    # === QUANTUM CLASSIFICATION (NEW) ===
    ("Quantum Classification", r"\b[Qq]uantum\s+[Cc]lassifi(?:er|cation)\b"),
    ("QGC", r"\bQGC\b"),
    ("Quantum Classifier", r"\b[Qq]uantum\s+[Cc]lassifier\b"),
    
    # === LINEAR ALGEBRA ===
    ("HHL", r"\bHHL\b"),
    ("HHL", r"\bHarrow\s*[-]?\s*Hassidim\s*[-]?\s*Lloyd\b"),
    
    # === ERROR CORRECTION ===
    ("QEC", r"\bQEC\b"),
    ("QEC", r"\b[Qq]uantum\s+[Ee]rror\s+[Cc]orrection\b"),
    ("Surface code", r"\b[Ss]urface\s+[Cc]ode\b"),
    ("Stabilizer code", r"\b[Ss]tabilizer\s+[Cc]ode\b"),
    
    # === SIMULATION ===
    ("Hamiltonian simulation", r"\bHamiltonian\s+simulation\b"),
    ("Trotter", r"\bTrotter(?:i[sz]ation)?\b"),
    ("Suzuki-Trotter", r"\bSuzuki\s*[-]?\s*Trotter\b"),
    
    # === COMMUNICATION ===
    ("Quantum Teleportation", r"\b[Qq]uantum\s+[Tt]eleportation\b"),
    ("QKD", r"\bQKD\b"),
    ("BB84", r"\bBB84\b"),
    
    # === QUANTUM STATES ===
    ("Bell state", r"\bBell\s+(?:state|pair)\b"),
    ("GHZ state", r"\bGHZ\s+state\b"),
    ("W state", r"\bW\s+state\b"),
    ("Cluster state", r"\b[Cc]luster\s+state\b"),
    ("Graph state", r"\b[Gg]raph\s+state\b"),
    
    # === OTHER ===
    ("Amplitude amplification", r"\b[Aa]mplitude\s+[Aa]mplification\b"),
    ("Amplitude estimation", r"\b[Aa]mplitude\s+[Ee]stimation\b"),
    ("State preparation", r"\b[Ss]tate\s+[Pp]reparation\b"),
    ("Quantum walk", r"\b[Qq]uantum\s+[Ww]alk\b"),
    
    # === FEATURE MAPS (NEW) ===
    ("ZZ Feature Map", r"\bZZ\s+[Ff]eature\s+[Mm]ap\b"),
    ("Feature Map", r"\b[Ff]eature\s+[Mm]ap\b"),
    ("Quantum Embedding", r"\b[Qq]uantum\s+[Ee]mbedding\b"),
    
    # === ANSATZ (NEW) ===
    ("Hardware-efficient ansatz", r"\b[Hh]ardware\s*[-]?\s*[Ee]fficient\s+[Aa]nsatz\b"),
    ("Ansatz", r"\b[Aa]nsatz\b"),
    ("UCCSD", r"\bUCCSD\b"),
    ("UCC", r"\bUCC\b"),
    
    # === NISQ ===
    ("NISQ", r"\bNISQ\b"),
]

ALGORITHM_PATTERNS = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in QUANTUM_ALGORITHMS_PATTERNS]


# ============================================
# CAPTION CLEANING - IMPROVED
# ============================================
LIGHT_CLEANUP_PATTERNS = [
    # LaTeX artifacts
    (r'\b(superscript|subscript|italic|bold|roman)\b', ''),
    (r'\b(mathrm|mathit|mathbf|mathcal|mathbb|mathsf)\b', ''),
    (r'\b(textrm|textit|textbf|textsf)\b', ''),
    (r'\b(start|end)(POSTSUBSCRIPT|POSTSUPERSCRIPT|RELOP|FLOAT\w+)\b', ''),
    (r'\blimit-from\b', ''),
    (r'\bannotation[- ]?xml\b', ''),
    
    # Empty brackets
    (r'\(\s*\)', ''),
    (r'\[\s*\]', ''),
    (r'\{\s*\}', ''),
    
    # Multiple spaces
    (r'\s{2,}', ' '),
]


def clean_caption_light(text):
    """Light cleaning - preserves meaningful content."""
    if not text:
        return ""
    
    # First: Clean Unicode artifacts
    cleaned = clean_unicode_artifacts(text)
    
    # Apply regex patterns
    for pattern, replacement in LIGHT_CLEANUP_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    
    # Clean subscript artifacts: q subscript 1 -> q1
    cleaned = re.sub(r'(\w)\s+subscript\s+(\d+)', r'\1\2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bq\s*[_]?\s*(\d+)', r'q\1', cleaned, flags=re.IGNORECASE)
    
    # Clean common math artifacts
    cleaned = re.sub(r'\s*[⁢⁣⁤]\s*', '', cleaned)  # Invisible operators
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Remove leading/trailing punctuation
    cleaned = re.sub(r'^[\s,;:\.\-\(\)]+', '', cleaned)
    cleaned = re.sub(r'[\s,;:]+$', '', cleaned)
    
    return cleaned.strip()


def normalize_ws(s):
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", s).strip()


def get_first_sentence(text):
    """Extract the FIRST SENTENCE from text."""
    if not text:
        return ""
    
    text = text.strip()
    
    # Pattern 1: Sentence ending with . ! ? followed by space and capital letter
    match = re.match(r'^(.*?[.!?])\s+[A-Z]', text, re.DOTALL)
    if match:
        result = match.group(1).strip()
        if len(result) > 15:
            return result
    
    # Pattern 2: Sentence ending with . ! ? at end or followed by space
    match = re.match(r'^(.*?[.!?])(?:\s|$)', text, re.DOTALL)
    if match:
        result = match.group(1).strip()
        if len(result) > 15:
            return result
    
    # Pattern 3: Take up to first period
    if '.' in text:
        first_part = text.split('.')[0] + '.'
        if len(first_part) > 15:
            return first_part.strip()
    
    # Fallback
    if len(text) <= 200:
        return text
    
    cut = text[:200].rfind(' ')
    if cut > 100:
        return text[:cut].strip() + "..."
    
    return text[:200] + "..."


# ============================================
# PATTERN EXTRACTION
# ============================================
def extract_quantum_gates(text):
    """Extract quantum gates from caption text."""
    # Normalize text for better pattern matching
    normalized_text = normalize_for_pattern_matching(text)
    
    found_gates = set()
    
    for gate_name, pattern in GATE_PATTERNS:
        if pattern.search(normalized_text):
            found_gates.add(gate_name)
    
    return sorted(list(found_gates)) if found_gates else []


def extract_quantum_algorithms(text):
    """Extract quantum algorithms from caption text."""
    # Normalize text for better pattern matching
    normalized_text = normalize_for_pattern_matching(text)
    
    found_algorithms = set()
    
    for algo_name, pattern in ALGORITHM_PATTERNS:
        if pattern.search(normalized_text):
            found_algorithms.add(algo_name)
    
    return sorted(list(found_algorithms)) if found_algorithms else []


# ============================================
# FIGURE PATTERNS
# ============================================
FIGURE_PREFIX_RE = re.compile(
    r"^\s*(figure|fig\.?)\s*([0-9]+(?:\.[0-9]+)?)\s*[:.\-]?\s*",
    re.IGNORECASE,
)
FIGURE_NUMBER_RE = re.compile(r"\b(Figure|Fig\.?)\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def arxiv_id_from_filename(filename):
    """Extract arXiv ID from filename."""
    base = os.path.basename(filename)
    if base.lower().endswith(".html"):
        base = base[:-5]
    match = re.search(r'(\d{4}\.\d{5})', base)
    return match.group(1) if match else base


def extract_text_from_tag(tag):
    """Extract text from HTML tag."""
    tag_copy = BeautifulSoup(str(tag), 'html.parser')
    for elem in tag_copy.find_all(['annotation', 'annotation-xml']):
        elem.decompose()
    
    texts = []
    for element in tag_copy.descendants:
        if isinstance(element, NavigableString):
            parent = element.parent
            if parent and parent.name in ['annotation', 'script', 'style']:
                continue
            t = str(element).strip()
            if t:
                texts.append(t)
    return ' '.join(texts)


def find_description_position(full_text, description):
    """Find position of DESCRIPTION in full HTML text."""
    if not description:
        return [[-1, -1]]
    
    # Strategy 1: Direct match
    start = full_text.find(description)
    if start != -1:
        return [[start, start + len(description)]]
    
    # Strategy 2: Match first N words
    words = description.split()
    for num_words in [10, 8, 6, 5, 4, 3]:
        if len(words) >= num_words:
            search_phrase = ' '.join(words[:num_words])
            start = full_text.find(search_phrase)
            if start != -1:
                end = start + len(description)
                return [[start, min(end, len(full_text))]]
    
    # Strategy 3: Fuzzy match
    significant_words = [w for w in words if len(w) > 4][:5]
    if significant_words:
        pattern = significant_words[0]
        for word in significant_words[1:]:
            pattern = pattern + r'.{0,50}' + re.escape(word)
        
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            return [[match.start(), match.start() + len(description)]]
    
    return [[-1, -1]]


def process_html_file(html_path, arxiv_id):
    """Process single HTML file and extract figures."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Get full text BEFORE removing annotations
    full_text = normalize_ws(soup.get_text(" ", strip=False))
    
    # Remove annotations for caption extraction
    for elem in soup.find_all(['annotation', 'annotation-xml']):
        elem.decompose()
    
    records = {}
    seen = set()
    
    # Method 1: Find <figure> tags
    for fig in soup.find_all("figure"):
        cap = fig.find("figcaption")
        if not cap:
            continue
        
        raw = extract_text_from_tag(cap)
        if not raw.strip():
            raw = cap.get_text(" ", strip=True)
        
        if raw.lstrip().lower().startswith("table"):
            continue
        
        m = FIGURE_NUMBER_RE.search(raw) or FIGURE_NUMBER_RE.search(cap.get_text(" ", strip=True))
        if not m:
            continue
        
        fig_num = m.group(2)
        if fig_num in seen:
            continue
        seen.add(fig_num)
        
        # Remove figure prefix and clean
        caption_no_prefix = FIGURE_PREFIX_RE.sub("", raw)
        caption_no_prefix = normalize_ws(caption_no_prefix)
        raw_caption = clean_caption_light(caption_no_prefix)
        
        if len(raw_caption) < 10:
            continue
        
        # Get first sentence for description
        description = get_first_sentence(raw_caption)
        
        # Find text positions
        text_positions = find_description_position(full_text, description)
        
        # Extract gates and algorithms (from normalized caption)
        gates = extract_quantum_gates(raw_caption)
        algos = extract_quantum_algorithms(raw_caption)
        
        try:
            fig_int = int(fig_num.split('.')[0]) if '.' in fig_num else int(fig_num)
        except:
            fig_int = 0
        
        key = f"{arxiv_id}_fig_{fig_int}"
        
        records[key] = OrderedDict([
            ("arxiv_number", arxiv_id),
            ("page_number", None),
            ("figure_number", fig_int),
            ("quantum_gates", gates if gates else "not specifically mentioned"),
            ("quantum_problem", algos if algos else "not specifically mentioned"),
            ("raw_caption", raw_caption),
            ("description", description),
            ("text_positions", text_positions),
        ])
    
    # Method 2: Find by caption classes
    for cls in ['ltx_caption', 'caption', 'fig-caption', 'figure-caption']:
        for cap in soup.find_all(class_=cls):
            if cap.find_parent('figure'):
                continue
            
            raw = extract_text_from_tag(cap)
            if not raw.strip():
                raw = cap.get_text(" ", strip=True)
            
            if raw.lstrip().lower().startswith("table"):
                continue
            
            m = FIGURE_NUMBER_RE.search(raw)
            if not m:
                continue
            
            fig_num = m.group(2)
            if fig_num in seen:
                continue
            seen.add(fig_num)
            
            caption_no_prefix = FIGURE_PREFIX_RE.sub("", raw)
            caption_no_prefix = normalize_ws(caption_no_prefix)
            raw_caption = clean_caption_light(caption_no_prefix)
            
            if len(raw_caption) < 10:
                continue
            
            description = get_first_sentence(raw_caption)
            text_positions = find_description_position(full_text, description)
            
            gates = extract_quantum_gates(raw_caption)
            algos = extract_quantum_algorithms(raw_caption)
            
            try:
                fig_int = int(fig_num.split('.')[0]) if '.' in fig_num else int(fig_num)
            except:
                fig_int = 0
            
            key = f"{arxiv_id}_fig_{fig_int}"
            
            if key not in records:
                records[key] = OrderedDict([
                    ("arxiv_number", arxiv_id),
                    ("page_number", None),
                    ("figure_number", fig_int),
                    ("quantum_gates", gates if gates else "not specifically mentioned"),
                    ("quantum_problem", algos if algos else "not specifically mentioned"),
                    ("raw_caption", raw_caption),
                    ("description", description),
                    ("text_positions", text_positions),
                ])
    
    return records


def load_checkpoint():
    """Load checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"processed": [], "total_figures": 0}


def save_checkpoint(checkpoint):
    """Save checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def main():
    print("=" * 70)
    print("  HTML Figure Extractor - IMPROVED VERSION")
    print("=" * 70)
    print(f"\n  Input:  {HTML_DIR}")
    print(f"  Output: {OUT_ROOT}")
    print()
    print("  Improvements:")
    print("    ✓ Better Unicode cleaning (⁢, invisible chars)")
    print("    ✓ Flexible gate detection (R z -> Rz)")
    print("    ✓ ZZ, XX, YY feature maps detected")
    print("    ✓ Expanded algorithm patterns")
    print("    ✓ QML, Ansatz, Feature Map detection")
    print()
    
    if not os.path.exists(HTML_DIR):
        print(f"  ERROR: Folder '{HTML_DIR}' not found!")
        return
    
    html_files = sorted([f for f in os.listdir(HTML_DIR) if f.lower().endswith(".html")])
    total_files = len(html_files)
    
    if total_files == 0:
        print(f"  ERROR: No HTML files found!")
        return
    
    print(f"  HTML files: {total_files}")
    
    checkpoint = load_checkpoint()
    processed_set = set(checkpoint.get("processed", []))
    remaining = [f for f in html_files if f not in processed_set]
    
    print(f"  Already processed: {len(processed_set)}")
    print(f"  Remaining: {len(remaining)}")
    print("-" * 70)
    
    if len(remaining) == 0:
        print("All files processed! Delete checkpoint to reprocess.")
        return
    
    start_time = time.time()
    figures_extracted = checkpoint.get("total_figures", 0)
    
    for i, fname in enumerate(remaining):
        html_path = os.path.join(HTML_DIR, fname)
        arxiv_id = arxiv_id_from_filename(fname)
        
        try:
            records = process_html_file(html_path, arxiv_id)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            checkpoint["processed"].append(fname)
            continue
        
        if records:
            folder_name = os.path.splitext(fname)[0]
            paper_dir = os.path.join(OUT_ROOT, folder_name)
            os.makedirs(paper_dir, exist_ok=True)
            
            for key, rec in records.items():
                json_path = os.path.join(paper_dir, f"{key}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(OrderedDict([(key, rec)]), f, ensure_ascii=False, indent=2)
            
            figures_extracted += len(records)
        
        checkpoint["processed"].append(fname)
        checkpoint["total_figures"] = figures_extracted
        
        if (i + 1) % PRINT_EVERY == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
            
            print(f"[Progress] {len(processed_set) + i + 1}/{total_files} | "
                  f"Figures: {figures_extracted} | "
                  f"Rate: {rate:.1f}/sec | ETA: {eta/60:.1f}m")
            
            save_checkpoint(checkpoint)
    
    save_checkpoint(checkpoint)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"  Files processed: {len(checkpoint['processed'])}")
    print(f"  Figures extracted: {figures_extracted}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"\n  Output: {os.path.abspath(OUT_ROOT)}")
    print("=" * 70)


# ============================================
# TEST FUNCTION
# ============================================
def test_detection():
    """Test the improved detection on your examples."""
    test_captions = [
        "Gate representations of the H, Z, X and CNOT operations respectively.",
        "Quantum circuit for alternating between X and Z gates with 3 qubits.",
        "The quantum circuit for a 3-qubit version of Grover's algorithm.",
        "Circuit diagram for the 4-qubit ZZ feature map. The angles of the R z single gates are not shown.",
        "Circuit representing the unitary evolution made of only CNOT and R z gates.",
        "Hardware-efficient ansatz for preparing the purification. The Ry and Rz are parametrized.",
        "QGC algorithm for quantum classification based on the joint probability density.",
    ]
    
    print("=" * 70)
    print("IMPROVED DETECTION TEST")
    print("=" * 70)
    
    for caption in test_captions:
        # Clean first
        cleaned = clean_caption_light(caption)
        gates = extract_quantum_gates(cleaned)
        algos = extract_quantum_algorithms(cleaned)
        
        print(f"\nCaption: {caption[:60]}...")
        print(f"  Gates: {gates}")
        print(f"  Algorithms: {algos}")


if __name__ == "__main__":
    # Run test first
    test_detection()
    print("\n")
    
    # Then run main
    main()