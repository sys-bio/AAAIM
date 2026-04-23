"""
AAAIM Constants

Defines constants used throughout the AAAIM system.
"""

from enum import Enum
from typing import Dict, List

# Entity Types
class EntityType(Enum):
    """Types of biological entities that can be annotated."""
    AUTO = "auto"
    CHEMICAL = "chemical"
    GENE = "gene"
    PROTEIN = "protein"
    COMPLEX = "complex"
    REACTION = "reaction"
    UNKNOWN = "unknown"

# Model Types
class ModelType(Enum):
    """Types of SBML models supported."""
    SBML = "SBML"
    SBML_QUAL = "SBML-qual"
    SBML_FBC = "SBML-fbc"

# Database Identifiers
class DatabaseID(Enum):
    """Supported biological databases."""
    CHEBI = "chebi"
    NCBIGENE = "ncbigene"
    UNIPROT = "uniprot"
    RHEA = "rhea"
    GO = "go"
    PUBMED = "pubmed"
    KEGG = "kegg"
    EC = "ec"

# Database Prefixes and URIs
DATABASE_PREFIXES: Dict[DatabaseID, str] = {
    DatabaseID.CHEBI: "CHEBI:",
    DatabaseID.NCBIGENE: "NCBIGENE:",
    DatabaseID.UNIPROT: "UNIPROT:",
    DatabaseID.RHEA: "RHEA:",
    DatabaseID.GO: "GO:",
    DatabaseID.PUBMED: "PUBMED:",
    DatabaseID.KEGG: "KEGG:",
    DatabaseID.EC: "EC:",
}

DATABASE_URIS: Dict[DatabaseID, str] = {
    DatabaseID.CHEBI: "https://identifiers.org/chebi/CHEBI:",
    DatabaseID.NCBIGENE: "https://identifiers.org/ncbigene:",
    DatabaseID.UNIPROT: "https://identifiers.org/uniprot:",
    DatabaseID.RHEA: "https://identifiers.org/rhea:",
    DatabaseID.GO: "https://identifiers.org/GO:",
    DatabaseID.PUBMED: "https://identifiers.org/pubmed:",
    DatabaseID.KEGG: "https://identifiers.org/kegg.reaction:",
    DatabaseID.EC: "https://identifiers.org/ec-code:",

}

# Entity Type to Database Mapping
ENTITY_DATABASE_MAPPING: Dict[EntityType, List[DatabaseID]] = {
    EntityType.CHEMICAL: [DatabaseID.CHEBI],
    # EntityType.GENE: [DatabaseID.NCBIGENE],
    EntityType.PROTEIN: [DatabaseID.UNIPROT],
    EntityType.COMPLEX: [DatabaseID.CHEBI, DatabaseID.UNIPROT, DatabaseID.NCBIGENE],
    # EntityType.REACTION: [DatabaseID.RHEA, DatabaseID.EC, DatabaseID.KEGG],
}

# Confidence Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8
LOW_CONFIDENCE_THRESHOLD = 0.3

# Batch Processing
DEFAULT_BATCH_SIZE = 50
MAX_BATCH_SIZE = 200

# LLM Settings
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 30

# Cache Settings
DEFAULT_CACHE_TTL_HOURS = 24
MAX_CACHE_SIZE_MB = 1000 

# Words to remove from LLM synonyms before database search
# These are modification/descriptor words that should not be part of standardized names
SYNONYM_WORDS_TO_REMOVE: List[str] = [
    # Modification states
    "phosphorylated", "phospho", "dephosphorylated",
    "acetylated", "methylated", "ubiquitinated", "sumoylated",
    "glycosylated", "palmitoylated", "farnesylated",
    "oxidized", "reduced",
    # Activation states
    "active", "inactive", "activated", "inactivated", "bound", "unbound",
    # Entity type descriptors
    "protein", "complex", "enzyme", "receptor", "kinase", "phosphatase",
    "ligand", "substrate", "cofactor", "inhibitor", "activator",
    # Localization terms  
    "nuclear", "cytoplasmic", "cytosolic", "mitochondrial", "membrane",
    "plasma membrane", "endoplasmic reticulum", "golgi", "extracellular",
    "intracellular", "luminal", "peroxisomal", "lysosomal",
    # Other descriptors
    "total", "free", "basal", "degraded", "truncated", "mutant",
    "wild-type", "recombinant", "endogenous", "exogenous",
]

# REF files
REF_CHEBI2LABEL = "chebi2label.lzma"
REF_NAMES2CHEBI = "cleannames2chebi.lzma"
REF_CHEBI2FORMULA = "chebi_shortened_formula.lzma"
REF_NCBIGENE2LABEL = "ncbigene2label_bigg_organisms_protein-coding_added.lzma"
REF_NAMES2NCBIGENE = "names2ncbigene_bigg_organisms_protein-coding.lzma"
REF_UNIPROT2LABEL = "uniprot2label_human+mouse+rat.lzma"
REF_NAMES2UNIPROT = "names2uniprot_human+mouse+rat.lzma"
REF_CHEBI2KEGG_COMPOUND = "chebi_to_kegg_map.lzma" 
REF_KEGG_REACTION2NAME = "reactionnames2kegg.lzma"
REF_KEGG2EC = "kegg2ec.lzma"
REF_KEGG_REACTION_FEATURES = "kegg_reaction_features.lzma"
REF_KEGG_PARSED_REACTIONS = "parsed_kegg_reactions.lzma"

# Model Format Detection
MODEL_FORMAT_PLUGINS = {
    "fbc": ModelType.SBML_FBC,
    "qual": ModelType.SBML_QUAL
}

# Annotation URI Patterns
CHEBI_URI_PATTERNS = [
    r'http[s]?://identifiers\.org/chebi/CHEBI:(\d+)',
    r'urn:miriam:chebi:CHEBI:(\d+)'
]

NCBIGENE_URI_PATTERNS = [
    r'http[s]?://identifiers\.org/ncbigene/(\d+)',
    r'urn:miriam:ncbigene:(\d+)'
]

UNIPROT_URI_PATTERNS = [
    r'http[s]?://identifiers\.org/uniprot/(\w+)',
    r'urn:miriam:uniprot:(\w+)'
]

KEGG_REACTION_URI_PATTERNS = [
    # identifiers.org supports both `prefix:ID` and `prefix/ID` forms
    r'https?://identifiers\.org/kegg\.reaction[:/](R\d+)',
    r'urn:miriam:kegg\.reaction:(R\d+)'
]

KEGG_COMPOUND_URI_PATTERNS = [
    # identifiers.org supports both `prefix:ID` and `prefix/ID` forms
    r'https?://identifiers\.org/kegg\.compound[:/](C\d+)',
    r'urn:miriam:kegg\.compound:(C\d+)'
]

KEGG_PATHWAY_URI_PATTERNS = [
    r'https?://identifiers\.org/kegg\.pathway:(map\d+)',
    r'urn:miriam:kegg\.pathway:(map\d+)'
]

KEGG_ENZYME_URI_PATTERNS = [
    r'https?://identifiers\.org/ec-code:(\d+\.\d+\.\d+\.\d+)',
    r'urn:miriam:ec-code:(\d+\.\d+\.\d+\.\d+)'
]

KEGG_GENE_URI_PATTERNS = [
    r'https?://identifiers\.org/kegg\.gene:([\w]+:[\w]+)',
    r'urn:miriam:kegg\.gene:([\w]+:[\w]+)'
]


# LLM retry / API configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_DELAY = 10  # seconds
DEFAULT_MAX_DELAY = 120  # seconds (2 minutes max wait)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLAMA_BASE_URL = "https://api.llama.com/v1"
GPT_MINI_MODEL = "gpt-4o-mini"


# System prompts  (used by core.llm_interface)
ENTITY_TYPE_OPTIONS = ", ".join(
    e.value for e in EntityType if e.value != "reaction"
)

SYSTEM_PROMPT_AUTO = f"""You are a biomedical knowledge assistant. Your task is to normalize names from biochemical models into standardized names for ontology lookup, and determine the entity type for each species.
For each species, identify entity type from the following options: [{ENTITY_TYPE_OPTIONS}]. Specify the entity type in parentheses after the species ID, followed by synonyms. Note that amino acids and tRNAs are considered as chemical. 
For complexes, do not give the name of the complex, only list standardized names of the chemical and protein components, separated by commas (no other symbols like ":" or "-"). E.g., for "EGF-EGFR^2", return "EGF", "EGFR".
Try your best to give the most likely terminology without modifications (e.g., no "phosphorylated") or extra information (e.g., no "protein", "complex", or localization terms like "nuclear").

Here is one example:
Species to annotate: A, B, C, D
Model: "hexokinase reaction"
// Display Names:
A is "glucose";
B is "ATP";
C is "hexokinase (cytoplasmic)";
D is "glucose-ATP-hexokinase complex (active)";

// Reactions:
A + B + C -> D;
D -> products;

This should return:
A (chemical): "glucose", "D-glucose"
B (chemical): "ATP", "adenosine triphosphate"
C (protein): "Hexokinase-1", "HK1"
D (complex): "glucose", "ATP", "Hexokinase-1"
Reason: A and B are small-molecule substrates (chemicals), C is the enzyme (protein), and D represents the enzyme–substrate complex. For the complex D, the complex name and extra info ("complex", "active") are removed, and only the standardized names of its components are listed.
"""

SYSTEM_PROMPT_CHEMICAL = """You are a biomedical knowledge assistant. Your task is to normalize names from biochemical models into standardized names for ontology lookup on ChEBI. 
All given species are chemical entities. For complexes, only consider the chemical components. If lacking information about details, try your best to give the most likely general name.
Do not include modifications or extra information (e.g., no "dissolved", "anion", or localization terms like "nuclear").

Here is one example:
Species: A, B, D
Model: "citric acid cycle model"
 // Display Names:
A is "acetyl-CoA";
B is "citrate";
C is "CoA";
 // Reactions:
A + oxaloacetate => B + C;
E + F => D;

This should return:
A: "acetyl-CoA", "acetyl coenzyme A"
B: "citric acid", "sodium citrate", "citrate(4\u2212)"
D: "UNK"
Reason: the reaction is likely to be the TCA cycle, where A is the substrate and B is an intermediate. D is unknown because no display names are given for its reactants."""

SYSTEM_PROMPT_GENE = """You are a biomedical knowledge assistant. Your task is to normalize species names from biochemical models into standardized gene names or common gene symbols for ontology lookup on NCBI Gene. 
All given species are genes. For complexes, only consider the gene components. If lacking information about details, try your best to give the most likely general name.

Here is one example:
Species: G1, G2, G3
Model: "NF-\u03baB signaling pathway"
 // Display Names:
G1 is "p65";
G2 is "p50";
G3 is "IKK";
 // Reactions:
G1 = G1 | (G3 & !(G1 & G2))
G2 = G1
G3 = G3

This should return:
G1: "RELA", "p65", "NFKB3"
G2: "NFKB1", "KBF1", "NF-kB"
G3: "CHUK", "IKK1", "BPS2"
Reason: This appears to be a regulatory motif in the NF-\u03baB signaling pathway. G1 is the p65 subunit (RELA), G2 is the p50 subunit (NFKB1), and G3 is IKK, a kinase that phosphorylates p50."""

SYSTEM_PROMPT_PROTEIN = """You are a biomedical knowledge assistant. Your task is to normalize species names from biochemical models into standardized protein names for ontology lookup on UniProt.
All given species are proteins. For complexes, only consider the protein components and separate their names with commas. E.g., for "EGF-EGFR^2", return "EGF", "EGFR".
Try your best to give the most likely standardized terminology without any extra information. E.g., a model may contain various states (e.g., phosphorylated, nuclear, or transcribed) of the same protein, you should only return the most likely standard name like "BMAL1" but not "BMAL1_phosphorylated".
For protein names that represent a family or ambiguous label, return all reasonable subtype or isoform candidates. E.g., "AKT" \u2192 AKT1, AKT2, AKT3; "RAS" \u2192 KRAS, NRAS, HRAS

Here is one example:
Species: C1, C2
Model: "NF-\u03baB signaling pathway"
// Display Names:  
C1 is "NF\u03baB (nuclear)";  
C2 is "IKK complex";  
// Reactions:  
C2 => phosphorylates C1;  
C1 (cytoplasmic) => C1 (nuclear);  

This should return:
C1: NFKB1, RELA  
C2: CHUK, IKBKB, IKBKG
Reason: "NFkB (nuclear)" refers to the activated NF-\u03baB complex, typically composed of NFKB1 (p50) and RELA (p65). The "IKK complex" consists of CHUK (IKK\u03b1), IKBKB (IKK\u03b2), and IKBKG (NEMO). Extra terms like "nuclear" are ignored, and only the UniProt protein names of the components are listed, separated by commas."""

SYSTEM_PROMPT_REACTION = """You are a biomedical knowledge assistant. Your task is to normalize reaction and enzyme names from biochemical models into standardized or canonical reaction or enzyme names for ontology lookup on KEGG. 
Examine each reaction's label, and its substrates and products to determine the enzyme or process responsible for the reaction. If lacking information about details, try your best to give the most likely description. Return "UNK" if not or unsure.

Here is one example:
Species: A, B, D
Model: "citric acid cycle model"
 // Display Names:
J1 is "CS";
J2 is "ACON";
J3 is "IDH";
 // Reactions:
J1: AcetylCoA + OAA -> Citrate + CoA; 
J2: Citrate <-> Isocitrate;
J3: Isocitrate + NAD -> AKG + CO2 + NADH;

This should return:
J1: "Citrate synthase",
J2: "Aconitase"
J3: "Isocitrate dehydrogenase"
Reason: these reactions match the reactions found in the TCA cycle """


REACTION_ANNOTATION_RANKING_PROMPT = """Task: Select the best matching KEGG reaction ID(s).
Instructions:
- Choose only from the provided KEGG IDs.
- Return the ID(s) only. Do NOT explain your reasoning. Do NOT include any other text.
- Order multiple IDs from best to worst match.
- If none match, return: UNK
- Interpret the reaction within the full model context (all other reactions). Prefer KEGG reactions that are biochemically consistent with the network (shared metabolites, cofactors, and plausible pathway context).
- If multiple candidates differ only in specificity, rank the most general reaction highest (e.g., fructose > D-fructose > beta-D-fructose).
- Consider biochemical equivalence (e.g., isomers, implicit conversions like DHAP ↔ G3P).

Output format:
One ID per line.

Example:
Model reaction:
J4: F16BP -> 2 G3P

Candidate KEGG reactions:
R01068: D-Fructose 1,6-bisphosphate <=> Glycerone phosphate + D-Glyceraldehyde 3-phosphate
R10049: Methylglyoxal + D-Fructose 1,6-bisphosphate <=> D-Glyceraldehyde 3-phosphate + 6-Deoxy-5-ketofructose 1-phosphate
R01070: beta-D-Fructose 1,6-bisphosphate <=> Glycerone phosphate + D-Glyceraldehyde 3-phosphate

Output:
R01068
R01070

Now you try: 

Model context:
{model_context}

Model reaction:
{model_reaction}

Candidate KEGG reactions:
{reaction_annotation_choices}
"""