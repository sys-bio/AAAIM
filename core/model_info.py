"""
Model Information Extraction for AAAIM

Extracts model information and context for annotation
"""

import re
import libsbml
import antimony
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import logging

from utils.constants import (
    DatabaseID,
    EntityType,
    MODEL_FORMAT_PLUGINS,
    ModelType,
    CHEBI_URI_PATTERNS,
    KEGG_REACTION_URI_PATTERNS,
    KEGG_COMPOUND_URI_PATTERNS,
    NCBIGENE_URI_PATTERNS,
    UNIPROT_URI_PATTERNS,
)

logger = logging.getLogger(__name__)

def detect_model_format(model_file: str) -> Tuple[ModelType, Dict[str, Any]]:
    """
    Detect the format of an SBML model (regular SBML, SBML-fbc, or SBML-qual).
    
    Args:
        model_file: Path to the SBML model file
        
    Returns:
        Tuple of (model_type, format_info) where format_info contains relevant plugin information
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return ModelType.SBML, {}
    
    format_info = {
        "has_fbc": False,
        "has_qual": False,
        "num_species": model.getNumSpecies(),
        "num_reactions": model.getNumReactions()
    }
    
    # Check for FBC plugin
    fbc_plugin = model.getPlugin("fbc")
    if fbc_plugin and fbc_plugin.getNumGeneProducts() > 0:
        format_info["has_fbc"] = True
        format_info["num_gene_products"] = fbc_plugin.getNumGeneProducts()
        return ModelType.SBML_FBC, format_info
    
    # Check for qual plugin
    qual_plugin = model.getPlugin("qual")
    if qual_plugin and qual_plugin.getNumQualitativeSpecies() > 0:
        format_info["has_qual"] = True
        format_info["num_qualitative_species"] = qual_plugin.getNumQualitativeSpecies()
        format_info["num_transitions"] = qual_plugin.getNumTransitions()
        return ModelType.SBML_QUAL, format_info
    
    # Default to regular SBML
    return ModelType.SBML, format_info

def extract_id_from_annotation(annotation_str: str, uri_patterns: List[str], bqbiol_qualifiers: list = None) -> List[str]:
    """
    Helper function to extract IDs from annotation string using both URI patterns.
    If bqbiol_qualifiers are provided, also search within those qualifier blocks for matches.
    Also checks for bqmodel qualifiers (incorrect usage) and extracts from them.
    """
    ids = set()

    # If bqbiol_qualifiers are provided, search within those blocks for matches
    if bqbiol_qualifiers:
        combined_str = ''
        for one_qualifier in bqbiol_qualifiers:
            # Search for both bqbiol and bqmodel qualifiers
            for prefix in ['bqbiol', 'bqmodel']:
                one_match = r'<{}:{}[^>]*?>.*?</{}:{}>'.format(
                    prefix, re.escape(one_qualifier), prefix, re.escape(one_qualifier)
                )
                one_matched = re.findall(one_match, annotation_str, flags=re.DOTALL)
                if len(one_matched) > 0:
                    matched_filt = [s.replace("      ", "") for s in one_matched]
                    one_str = '\n'.join(matched_filt)
                else:
                    one_str = ''
                combined_str = combined_str + one_str

        # Search the combined qualifier string for all URI patterns
        for pattern in uri_patterns:
            matches = re.findall(pattern, combined_str)
            ids.update(matches)
    else:
        # If no bqbiol qualifiers are provided, directly search the annotation string for all URI patterns
        for pattern in uri_patterns:
            matches = re.findall(pattern, annotation_str)
            ids.update(matches)

    return list(ids)

def extract_id_and_qualifier_from_annotation(annotation_str: str, uri_patterns: List[str], bqbiol_qualifiers: list = None) -> Tuple[List[str], List[str]]:
    """
    Helper function to extract IDs and their corresponding qualifiers from annotation string.
    Also checks for bqmodel qualifiers (incorrect usage) and extracts from them.
    Removes duplicate IDs, keeping only the first occurrence and its qualifier.
    
    Args:
        annotation_str: XML annotation string
        uri_patterns: List of URI patterns to search for
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (None for all)
        
    Returns:
        Tuple of (ids, qualifiers) where qualifiers[i] corresponds to ids[i]
        Both lists are deduplicated, keeping only first occurrence
    """
    ids = []
    qualifiers = []
    seen_ids = set()  # Track IDs we've already seen
    
    if bqbiol_qualifiers:
        # Search within specific qualifier blocks for both bqbiol and bqmodel
        for qualifier in bqbiol_qualifiers:
            for prefix in ['bqbiol', 'bqmodel']:
                qualifier_match = r'<{}:{}[^>]*?>.*?</{}:{}>'.format(
                    prefix, re.escape(qualifier), prefix, re.escape(qualifier)
                )
                qualifier_blocks = re.findall(qualifier_match, annotation_str, flags=re.DOTALL)
                
                for block in qualifier_blocks:
                    # Search for URI patterns within this qualifier block
                    for pattern in uri_patterns:
                        matches = re.findall(pattern, block)
                        for match in matches:
                            # Only add if we haven't seen this ID before
                            if match not in seen_ids:
                                ids.append(match)
                                qualifiers.append(qualifier)
                                seen_ids.add(match)
    else:
        # If no specific qualifiers given, search for all qualifiers
        # Find all bqbiol qualifier blocks and extract their content
        full_blocks = re.findall(r'<bqbiol:([^>]+)[^>]*?>(.*?)</bqbiol:\1>', annotation_str, flags=re.DOTALL)
        
        for qualifier_name, block_content in full_blocks:
            # Search for URI patterns within this qualifier block
            for pattern in uri_patterns:
                matches = re.findall(pattern, block_content)
                for match in matches:
                    # Only add if we haven't seen this ID before
                    if match not in seen_ids:
                        ids.append(match)
                        qualifiers.append(qualifier_name)
                        seen_ids.add(match)
        
        # Also search for bqmodel qualifier blocks (incorrect usage but count them)
        model_blocks = re.findall(r'<bqmodel:([^>]+)[^>]*?>(.*?)</bqmodel:\1>', annotation_str, flags=re.DOTALL)
        
        if model_blocks:
            for qualifier_name, block_content in model_blocks:
                # Search for URI patterns within this qualifier block
                for pattern in uri_patterns:
                    matches = re.findall(pattern, block_content)
                    for match in matches:
                        # Only add if we haven't seen this ID before
                        if match not in seen_ids:
                            ids.append(match)
                            qualifiers.append(qualifier_name)
                            seen_ids.add(match)
    
    return ids, qualifiers


def _warn_bqmodel_in_annotation(annotation_str: str, entity_id: str, entity_kind: str) -> None:
    if "bqmodel:" in annotation_str:
        logger.warning(
            f"{entity_kind} '{entity_id}': Found bqmodel qualifier instead of bqbiol - incorrect usage"
        )


def _collect_species_ids_from_list(
    model: libsbml.Model,
    target: Dict[str, List[str]],
    uri_patterns: List[str],
    bqbiol_qualifiers: Optional[list],
    *,
    entity_kind: str = "Species",
) -> None:
    for species in model.getListOfSpecies():
        species_id = species.getId()
        if not species.isSetAnnotation():
            continue
        annotation_str = species.getAnnotation().toXMLString()
        _warn_bqmodel_in_annotation(annotation_str, species_id, entity_kind)
        found = extract_id_from_annotation(annotation_str, uri_patterns, bqbiol_qualifiers)
        if found:
            target[species_id] = list(dict.fromkeys(found))


def _collect_qual_species_ids_simple(
    model: libsbml.Model,
    target: Dict[str, List[str]],
    uri_patterns: List[str],
    bqbiol_qualifiers: Optional[list],
) -> None:
    qual_plugin = model.getPlugin("qual")
    if not qual_plugin:
        return
    for qual_species in qual_plugin.getListOfQualitativeSpecies():
        qid = qual_species.getId()
        if not qual_species.isSetAnnotation():
            continue
        annotation_str = qual_species.getAnnotation().toXMLString()
        _warn_bqmodel_in_annotation(annotation_str, qid, "Qualitative species")
        found = extract_id_from_annotation(annotation_str, uri_patterns, bqbiol_qualifiers)
        if found:
            target[qid] = list(dict.fromkeys(found))


def _collect_fbc_gene_product_ids_simple(
    model: libsbml.Model,
    target: Dict[str, List[str]],
    uri_patterns: List[str],
    bqbiol_qualifiers: Optional[list],
) -> None:
    fbc_plugin = model.getPlugin("fbc")
    if not fbc_plugin:
        return
    for gene_product in fbc_plugin.getListOfGeneProducts():
        gid = gene_product.getId()
        if not gene_product.isSetAnnotation():
            continue
        annotation_str = gene_product.getAnnotation().toXMLString()
        _warn_bqmodel_in_annotation(annotation_str, gid, "Gene")
        found = extract_id_from_annotation(annotation_str, uri_patterns, bqbiol_qualifiers)
        if found:
            target[gid] = list(dict.fromkeys(found))


def _qualifier_map_from_parallel_lists(ids: List[str], qualifier_list: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, ann_id in enumerate(ids):
        out[ann_id] = qualifier_list[i] if i < len(qualifier_list) else "unknown"
    return out


def _record_entity_annotations_with_qualifiers(
    annotations: Dict[str, List[str]],
    qualifier_annotations: Dict[str, Dict[str, str]],
    entity_id: str,
    ids: List[str],
    qualifier_list: List[str],
) -> None:
    if not ids:
        return
    annotations[entity_id] = ids
    qualifier_annotations[entity_id] = _qualifier_map_from_parallel_lists(ids, qualifier_list)


def find_species_with_chebi_annotations(model_file: str, bqbiol_qualifiers: list = None) -> Dict[str, List[str]]:
    """
    Find species with existing ChEBI annotations.

    Args:
        model_file: Path to the SBML model file
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])

    Returns:
        Dictionary mapping species IDs to their ChEBI annotation IDs
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()

    if model is None:
        return {}

    model_type, _ = detect_model_format(model_file)
    chebi_annotations: Dict[str, List[str]] = {}

    if model_type in (ModelType.SBML, ModelType.SBML_FBC):
        _collect_species_ids_from_list(
            model, chebi_annotations, CHEBI_URI_PATTERNS, bqbiol_qualifiers
        )
    elif model_type == ModelType.SBML_QUAL:
        _collect_qual_species_ids_simple(
            model, chebi_annotations, CHEBI_URI_PATTERNS, bqbiol_qualifiers
        )

    return chebi_annotations

def find_species_with_annotations_and_qualifiers(model_file: str, database: str, bqbiol_qualifiers: list = None) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Find species with existing annotations and their corresponding qualifiers for any supported database.

    Args:
        model_file: Path to the SBML model file
        database: Database to search ("chebi", "ncbigene", "uniprot", "kegg").
            For species, "kegg" means KEGG compound IDs (C#####).
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])

    Returns:
        Tuple of (annotations, qualifier_annotations) where:
        - annotations: Dictionary mapping species IDs to their annotation IDs
        - qualifier_annotations: Dictionary mapping species IDs to a dict of {annotation_id: qualifier}
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()

    if model is None:
        return {}, {}

    model_type, _ = detect_model_format(model_file)
    annotations: Dict[str, List[str]] = {}
    qualifier_annotations: Dict[str, Dict[str, str]] = {}

    # Select URI patterns based on database
    if isinstance(database, DatabaseID):
        database = database.value

    if database == DatabaseID.CHEBI.value:
        uri_patterns = CHEBI_URI_PATTERNS
    elif database == DatabaseID.NCBIGENE.value:
        uri_patterns = NCBIGENE_URI_PATTERNS
    elif database == DatabaseID.UNIPROT.value:
        uri_patterns = UNIPROT_URI_PATTERNS
    elif database == DatabaseID.KEGG.value:
        uri_patterns = KEGG_COMPOUND_URI_PATTERNS
    else:
        logger.warning(f"Database {database} not supported")
        return {}, {}

    # Process species annotations (all model types that expose Species)
    for species in model.getListOfSpecies():
        species_id = species.getId()
        if not species.isSetAnnotation():
            continue
        annotation_str = species.getAnnotation().toXMLString()
        _warn_bqmodel_in_annotation(annotation_str, species_id, "Species")
        ids, qualifier_list = extract_id_and_qualifier_from_annotation(
            annotation_str, uri_patterns, bqbiol_qualifiers
        )
        _record_entity_annotations_with_qualifiers(
            annotations, qualifier_annotations, species_id, ids, qualifier_list
        )

    # Process FBC gene products if applicable
    if model_type == ModelType.SBML_FBC and database in [DatabaseID.NCBIGENE.value, DatabaseID.UNIPROT.value]:
        fbc_plugin = model.getPlugin("fbc")
        if fbc_plugin:
            for gene_product in fbc_plugin.getListOfGeneProducts():
                gene_id = gene_product.getId()
                if not gene_product.isSetAnnotation():
                    continue
                annotation_str = gene_product.getAnnotation().toXMLString()
                _warn_bqmodel_in_annotation(annotation_str, gene_id, "Gene")
                ids, qualifier_list = extract_id_and_qualifier_from_annotation(
                    annotation_str, uri_patterns, bqbiol_qualifiers
                )
                _record_entity_annotations_with_qualifiers(
                    annotations, qualifier_annotations, gene_id, ids, qualifier_list
                )

    # Process qualitative species if applicable
    if model_type == ModelType.SBML_QUAL:
        qual_plugin = model.getPlugin("qual")
        if qual_plugin:
            for qual_species in qual_plugin.getListOfQualitativeSpecies():
                qual_id = qual_species.getId()
                if not qual_species.isSetAnnotation():
                    continue
                annotation_str = qual_species.getAnnotation().toXMLString()
                _warn_bqmodel_in_annotation(annotation_str, qual_id, "Qualitative species")
                ids, qualifier_list = extract_id_and_qualifier_from_annotation(
                    annotation_str, uri_patterns, bqbiol_qualifiers
                )
                _record_entity_annotations_with_qualifiers(
                    annotations, qualifier_annotations, qual_id, ids, qualifier_list
                )

    return annotations, qualifier_annotations
    
def find_species_with_ncbigene_annotations(model_file: str, bqbiol_qualifiers: list = None) -> Dict[str, List[str]]:
    """
    Find species with existing NCBI gene annotations.
    
    Args:
        model_file: Path to the SBML model file
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])
        
    Returns:
        Dictionary mapping species IDs to their NCBI gene annotation IDs
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return {}

    model_type, _ = detect_model_format(model_file)
    ncbigene_annotations: Dict[str, List[str]] = {}

    if model_type == ModelType.SBML_FBC:
        _collect_fbc_gene_product_ids_simple(
            model, ncbigene_annotations, NCBIGENE_URI_PATTERNS, bqbiol_qualifiers
        )
    elif model_type == ModelType.SBML_QUAL:
        _collect_qual_species_ids_simple(
            model, ncbigene_annotations, NCBIGENE_URI_PATTERNS, bqbiol_qualifiers
        )
    elif model_type == ModelType.SBML:
        _collect_species_ids_from_list(
            model, ncbigene_annotations, NCBIGENE_URI_PATTERNS, bqbiol_qualifiers
        )

    return ncbigene_annotations

def find_species_with_uniprot_annotations(model_file: str, bqbiol_qualifiers: list = None) -> Dict[str, List[str]]:
    """
    Find species with existing UniProt annotations.
    
    Args:
        model_file: Path to the SBML model file
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])
        
    Returns:
        Dictionary mapping species IDs to their UniProt annotation IDs
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return {}

    model_type, _ = detect_model_format(model_file)
    uniprot_annotations: Dict[str, List[str]] = {}

    if model_type == ModelType.SBML_FBC:
        _collect_fbc_gene_product_ids_simple(
            model, uniprot_annotations, UNIPROT_URI_PATTERNS, bqbiol_qualifiers
        )
    elif model_type == ModelType.SBML_QUAL:
        _collect_qual_species_ids_simple(
            model, uniprot_annotations, UNIPROT_URI_PATTERNS, bqbiol_qualifiers
        )
    elif model_type == ModelType.SBML:
        _collect_species_ids_from_list(
            model, uniprot_annotations, UNIPROT_URI_PATTERNS, bqbiol_qualifiers
        )

    return uniprot_annotations


def find_reactions_with_kegg_annotations(model_file: str, bqbiol_qualifiers: list = None) -> Dict[str, List[str]]:
    """
    Find reactions with existing KEGG annotations.

    Args:
        model_file: Path to the SBML model file
        bqbiol_qualifiers: List of bqbiol qualifiers to extract (e.g. ['is', 'isVersionOf', 'hasPart'])

    Returns:
        Dictionary mapping reaction IDs to their KEGG annotation IDs
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()

    if model is None:
        return {}

    model_type, format_info = detect_model_format(model_file)
    kegg_annotations = {}

    if model_type in [ModelType.SBML, ModelType.SBML_FBC]:
        for reaction in model.getListOfReactions():
            reaction_id = reaction.getId()

            if reaction.isSetAnnotation():
                annotation = reaction.getAnnotation()
                annotation_str = annotation.toXMLString()
                kegg_ids = extract_id_from_annotation(annotation_str, KEGG_REACTION_URI_PATTERNS, bqbiol_qualifiers)
                if kegg_ids:
                    kegg_annotations[reaction_id] = kegg_ids

    elif model_type == ModelType.SBML_QUAL:
        # QUAL models typically use Transitions, not Reactions
        qual_plugin = model.getPlugin("qual")
        if qual_plugin and hasattr(qual_plugin, "getListOfTransitions"):
            for transition in qual_plugin.getListOfTransitions():
                transition_id = transition.getId()
                if transition.isSetAnnotation():
                    annotation = transition.getAnnotation()
                    annotation_str = annotation.toXMLString()
                    kegg_ids = extract_id_from_annotation(annotation_str, KEGG_REACTION_URI_PATTERNS, bqbiol_qualifiers)
                    if kegg_ids:
                        kegg_annotations[transition_id] = kegg_ids

    return kegg_annotations, {} # empty qualifier annotations


def get_species_display_names(
    model_file: str,
    entity_type: str | EntityType = EntityType.CHEMICAL,
) -> Dict[str, str]:
    """
    Get the display names for all species in the model.
    Supports regular species, FBC gene products, and qual qualitative species.
    
    Args:
        model_file: Path to the SBML model file
        entity_type: Type of entity ("chemical" for species, "gene" for gene products)
        
    Returns:
        Dictionary mapping species/gene IDs to their display names
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        return {}
    
    model_type, format_info = detect_model_format(model_file)
    
    if isinstance(entity_type, str):
        try:
            entity_type = EntityType(entity_type)
        except ValueError:
            entity_type = EntityType.CHEMICAL

    if entity_type in (EntityType.GENE, EntityType.PROTEIN):
        names = {}
        
        if model_type == ModelType.SBML_FBC:
            # Use FBC plugin for gene products
            fbc_plugin = model.getPlugin("fbc")
            if fbc_plugin:
                for gene_product in fbc_plugin.getListOfGeneProducts():
                    gene_id = gene_product.getIdAttribute()
                    
                    # Try to get name in order of preference: name > label > id
                    if gene_product.isSetName() and gene_product.getName():
                        gene_name = gene_product.getName()
                    elif gene_product.isSetLabel() and gene_product.getLabel():
                        gene_name = gene_product.getLabel()
                    else:
                        gene_name = gene_id
                    
                    names[gene_id] = gene_name
        
        elif model_type == ModelType.SBML_QUAL:
            # Use qual plugin for qualitative species
            qual_plugin = model.getPlugin("qual")
            if qual_plugin:
                for qual_species in qual_plugin.getListOfQualitativeSpecies():
                    qual_id = qual_species.getId()
                    
                    # Try to get name in order of preference: name > id
                    if qual_species.isSetName() and qual_species.getName():
                        qual_name = qual_species.getName()
                    else:
                        qual_name = qual_id
                    
                    names[qual_id] = qual_name
        else:
            names = {val.getId(): val.getName() for val in model.getListOfSpecies()}
        return names
    else:
        # Use regular species for chemical entities
        names = {val.getId(): val.getName() for val in model.getListOfSpecies()}
        return names

def get_all_species_ids(model_file: str, entity_type: str = "chemical") -> List[str]:
    """
    Get all species IDs from an SBML model.
    Supports regular species, FBC gene products, and qual qualitative species.
    
    Args:
        model_file: Path to SBML model file
        entity_type: Type of entity ("chemical" for species, "gene" for gene products)
        
    Returns:
        List of species/gene IDs
    """
    display_names = get_species_display_names(model_file, entity_type)
    return list(display_names.keys())

def get_reaction_display_names(model_file: str) -> Dict[str, str]:
    """
    Get the display names for all reactions in the model.
    Supports regular SBML reactions and SBML-qual transitions.
    
    Args:
        model_file: Path to the SBML model file

    Returns:
        Dictionary mapping reaction or transition IDs to their display names
    """
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()

    if model is None:
        return {}

    model_type, format_info = detect_model_format(model_file)
    names = {}

    if model_type in [ModelType.SBML, ModelType.SBML_FBC]:
        # Regular SBML or FBC: use reactions
        for reaction in model.getListOfReactions():
            reaction_id = reaction.getId()

            # Prefer name over ID
            if reaction.isSetName() and reaction.getName():
                reaction_name = reaction.getName()
            else:
                reaction_name = reaction_id

            names[reaction_id] = reaction_name

    elif model_type == ModelType.SBML_QUAL:
        # SBML-qual: use transitions
        qual_plugin = model.getPlugin("qual")
        if qual_plugin and hasattr(qual_plugin, "getListOfTransitions"):
            for transition in qual_plugin.getListOfTransitions():
                transition_id = transition.getId()

                if transition.isSetName() and transition.getName():
                    transition_name = transition.getName()
                else:
                    transition_name = transition_id

                names[transition_id] = transition_name

    return names

def get_all_reaction_ids(model_file: str) -> List[str]:
    """
    Get all species IDs from an SBML model.
    Supports regular species, FBC gene products, and qual qualitative species.
    
    Args:
        model_file: Path to SBML model file
        entity_type: Type of entity ("chemical" for species, "gene" for gene products)
        
    Returns:
        List of species/gene IDs
    """
    display_names = get_reaction_display_names(model_file)
    return list(display_names.keys())

def extract_qual_transitions(model_file: str, species_ids: List[str]) -> List[str]:
    """
    Extract boolean transitions from SBML-qual models.
    Self loops are ignored.
    
    Args:
        model_file: Path to the SBML-qual model file
        species_ids: List of species IDs to filter transitions for
        
    Returns:
        List of transition strings in the format "target = rule"
    """
    
    # Open the SBML file and get the qualitative_model model
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()  # Check whether a model exists
    qualitative_model = model.getPlugin("qual")  # Get the qualitative_model part of the model.
    if qualitative_model is None:  # Check whether a Qual model exists.
        logger.warning("Error loading SBML file: no Qual plugin found")
        return None

    # STEP: read the formulas from transitions and convert them to bnet format.
    transitions = []
    for transition in qualitative_model.getListOfTransitions():  # Scan all the transitions.
        # Get the output variable
        output = transition.getListOfOutputs()
        if len(output) > 1:  # check whether there is a single output
            logger.warning(f"Multiple outputs assigned. List of outputs: {output}")
            return None
        else:
            target = output[0].getQualitativeSpecies()

        # Get the formula
        logic_terms = transition.getListOfFunctionTerms()

        if len(logic_terms) == 0:  # Empty transition in SBML file, skip it.
            continue

        if len(logic_terms) > 1:  # check whether there exists a single formula only, error otherwise
            logger.warning(f"Multiple logic terms present. Number of terms: {len(logic_terms)}")
            return None
        else:  # Get the SBML QUAL formula
            formula = libsbml.formulaToL3String(logic_terms[0].getMath())

        # Convert the SBML QUAL formula into bnet syntax before parsing it.
        rule = re.sub(r'\|\|', '|', formula)  # convert || to |
        rule = re.sub(r'&&', '&', rule)  # convert && to &
        # convert (<var> == 1) or <var> == 1 to <var>
        rule = re.sub(r'\(?(\w+)\s*==\s*1\)?', r'\1', rule)
        # convert (<var> == 0) or <var> == 0 to ! <var>
        rule = re.sub(r'\(?(\w+)\s*==\s*0\)?', r'!\1', rule)

        # Remove self loops: skip if left_side and right_side are the same (ignoring whitespace)
        if target.strip() == rule.strip():
            # print(f"Skipping self loop: {target.strip()} = {rule.strip()}")
            continue
        
        # Check if this transition involves any of our target species
        if target in species_ids or any(species_id in rule for species_id in species_ids):
            transitions.append(f"{target} = {rule}")
    
    return transitions


def _parse_antimony_reaction_matches(model_file: str) -> List[Tuple[str, str, str]]:
    """Parse antimony export into (left_side, arrow, right_side) tuples for each reaction."""
    antimony.clearPreviousLoads()
    sbml_model = antimony.loadSBMLFile(model_file)
    if sbml_model == -1:
        logger.error(f"Error loading SBML file with antimony: {antimony.getLastError()}")
        return []

    antimony_string = antimony.getAntimonyString()
    reaction_pattern = re.compile(r"// Reactions:.*?(?=//|$)", re.DOTALL)
    reactions_section = reaction_pattern.search(antimony_string)
    
    reaction_matches = []
    if reactions_section:
        reactions_text = reactions_section.group(0).replace("// Reactions:", "").strip()
        reaction_pattern = re.compile(r"([^;]+)(=>|->)([^;]+);", re.MULTILINE)
        reaction_matches = reaction_pattern.findall(reactions_text)

    # If no matches found with '=>', try with '=' instead
    if not reaction_matches:
        reaction_pattern = re.compile(r"// Rate Rules:.*?(?=//|$)", re.DOTALL)
        reactions_section = reaction_pattern.search(antimony_string)
        if reactions_section:
            reactions_text = reactions_section.group(0).replace("// Rate Rules:", "").strip()
            reaction_pattern = re.compile(r"([^;]+)(=>|->|=)([^;]+);", re.MULTILINE)
            reaction_matches = reaction_pattern.findall(reactions_text)

    return reaction_matches


def map_reaction_ids_to_stoichiometry_strings(model_file: str) -> Dict[str, str]:
    """
    Map each SBML reaction id to a one-line stoichiometry string ``RID: lhs arrow rhs``
    (same style as KEGG ranking prompts).
    """
    out: Dict[str, str] = {}
    for left_side, arrow, right_side in _parse_antimony_reaction_matches(model_file):
        m = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", left_side.strip(), re.DOTALL)
        if not m:
            continue
        rid, rest = m.group(1), m.group(2).strip()
        out[rid] = f"{rid}: {rest} {arrow} {right_side.strip()}"
    return out


def build_antimony_reaction_index(model_file: str) -> List[Tuple[str, str, Set[str]]]:
    """
    Parse the model once via antimony and build an index of:
        (reaction_id, reaction_string_without_id_prefix, set_of_species_ids_in_reaction)

    This is intended for performance-sensitive loops that repeatedly filter
    reactions by a changing set of target species IDs (e.g., EM iterations).
    """
    index: List[Tuple[str, str, Set[str]]] = []
    for left_side, arrow, right_side in _parse_antimony_reaction_matches(model_file):
        m = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", left_side.strip(), re.DOTALL)
        if not m:
            continue
        reaction_id = m.group(1).strip()
        left_side_body = m.group(2).strip()

        left_side_cleaned = re.sub(r"^[A-Za-z0-9_]+:\s*", "", left_side.strip())
        # Keep the historical behavior: the returned reaction string excludes the RID prefix.
        reaction_str = f"{left_side_body} {arrow} {right_side.strip()}"

        all_ids_in_reaction = set(
            re.findall(r"\b([A-Za-z0-9_]+)\b", left_side + " " + right_side)
        )
        index.append((reaction_id, reaction_str, all_ids_in_reaction))
    return index


def _species_ids_from_stoichiometry_side(side_str: str) -> Set[str]:
    """Metabolite/species tokens on one side of a reaction equation.

    Canonical parsing for the AAAIM pipeline; reused via
    :func:`reaction_stoichiometry_lhs_rhs_species` from
    :func:`core.database_search._get_kegg_recommendations_rulebased` so that
    "exchange-style" reactions (empty LHS or RHS) are detected consistently.
    """
    out: Set[str] = set()
    side = str(side_str or "").strip()
    if not side:
        return out
    for term in side.split("+"):
        parts = term.strip().split()
        if not parts:
            continue
        if len(parts) == 1:
            met = parts[0]
        else:
            try:
                float(parts[0])
            except ValueError:
                met = term.strip()
            else:
                met = parts[-1]
        met = met.lstrip("$").strip()
        if met:
            out.add(met)
    return out


def reaction_stoichiometry_lhs_rhs_species(reaction_str: str) -> Tuple[Set[str], Set[str]]:
    """Return ``(lhs_species_ids, rhs_species_ids)`` parsed from a reaction string."""
    s = str(reaction_str or "")
    if "=>" in s or "->" in s:
        lhs, rhs = re.split(r"=>|->", s, maxsplit=1)
        return (
            _species_ids_from_stoichiometry_side(lhs),
            _species_ids_from_stoichiometry_side(rhs),
        )
    return set(), set()


def exchange_constraint_skipped_reaction_ids(model_file: str) -> Set[str]:
    """SBML reaction ids with an empty LHS or RHS (source/sink/exchange-style).

    When ``include_exchange_reactions`` is False in the rule-based KEGG matcher,
    these reactions get no candidates and are omitted from LLM re-ranking
    (see ``exchange_skipped`` metadata in ``_get_kegg_recommendations_rulebased``).
    """
    out: Set[str] = set()
    for reaction_id, reaction_str, _ in build_antimony_reaction_index(model_file):
        lhs, rhs = reaction_stoichiometry_lhs_rhs_species(reaction_str)
        if not lhs or not rhs:
            out.add(str(reaction_id))
    return out


def filter_reactions_from_antimony_index(
    antimony_index: List[Tuple[str, str, Set[str]]],
    species_ids: List[str],
) -> Tuple[List[str], List[str], Set[str]]:
    """
    Filter a prebuilt antimony reaction index down to reactions involving any
    of the provided *species_ids*.

    Returns:
        (reactions, reaction_ids, related_species)
    """
    species_set = set(species_ids)
    reactions: List[str] = []
    reaction_ids: List[str] = []
    related_species: Set[str] = set(species_ids)

    for reaction_id, reaction_str, ids_in_reaction in antimony_index:
        if ids_in_reaction & species_set:
            reactions.append(reaction_str)
            reaction_ids.append(reaction_id)
            related_species |= ids_in_reaction

    return reactions, reaction_ids, related_species


def map_reaction_ids_to_participant_ids(model_file: str) -> Dict[str, Set[str]]:
    """
    Map each reaction id to the set of species ids appearing in that reaction.

    Uses the cached antimony parsing (via :func:`build_antimony_reaction_index`).
    """
    out: Dict[str, Set[str]] = {}
    for reaction_id, _reaction_str, ids_in_reaction in build_antimony_reaction_index(model_file):
        out[str(reaction_id)] = set(ids_in_reaction)
    return out


def extract_reactions_from_sbml(model_file: str, species_ids: List[str]) -> Tuple[List[str], set]:
    """
    Extract reactions from an SBML model file using antimony.
    Returns both the reactions involving the target species and a set of all related species.
    
    Args:
        model_file: Path to the SBML model file
        species_ids: List of species IDs to filter reactions for
        
    Returns:
        Tuple containing:
        - List of reaction strings
        - Set of all species IDs involved in the filtered reactions
    """
    reactions = []
    related_species = set(species_ids)

    reaction_matches = _parse_antimony_reaction_matches(model_file)

    # Filter reactions to only include those involving our species
    for match in reaction_matches:
        left_side, arrow, right_side = match
        
        # Remove reaction ID prefix (e.g., "J53:" or "R53:" from beginning of left_side)
        # This captures patterns like "J53: " or "R1: " etc. at the start
        left_side_cleaned = re.sub(r'^[A-Za-z0-9_]+:\s*', '', left_side.strip())
        
        reaction_str = f"{left_side_cleaned} {arrow} {right_side.strip()}"
        
        # Check if any of our species IDs are in this reaction
        if any(re.search(r'\b' + re.escape(species_id) + r'\b', left_side + ' ' + right_side) for species_id in species_ids):
            reactions.append(reaction_str)
            
            # Extract all species IDs from this reaction
            all_ids_in_reaction = re.findall(r'\b([A-Za-z0-9_]+)\b', left_side + ' ' + right_side)
            related_species.update(all_ids_in_reaction)
            
    return reactions, related_species

def extract_model_info(model_file: str, species_ids: List[str], entity_type: str = "chemical") -> Dict[str, Any]:
    """
    Extract display names and reactions/transitions for the specified species.
    Supports regular SBML, SBML-fbc, and SBML-qual models.
    
    Args:
        model_file: Path to the SBML model file
        species_ids: List of species IDs to include
        entity_type: Type of entity ("chemical" for species, "gene" for gene products)
        
    Returns:
        Dictionary with model name, model type, display names, and reactions/transitions
    """
    ########## MODEL DETECTION AND BASIC INFO ##########
    reader = libsbml.SBMLReader()
    document = reader.readSBML(model_file)
    model = document.getModel()
    
    if model is None:
        logger.error(f"Error loading SBML file: {model_file}")
        return {}
    
    model_type, format_info = detect_model_format(model_file)
    
    # Extract model name
    model_name = model.getName() if model.isSetName() else model.getId() if model.isSetId() else ""
    
    # Extract model notes
    model_notes = model.getNotesString() if model.isSetNotes() else ""
    if model_notes != "":
        # Remove HTML tags from model notes
        model_notes = re.sub(r'<[^>]*>', '', model_notes)

        # Split by newlines and/or multiple spaces
        lines = re.split(r'\n|\s{2,}', model_notes)
        
        # List of keywords/fragments that indicate boilerplate text
        boilerplate_keywords = [
            'copyright', 'public domain', 'rights', 'CC0', 'dedication', 
            'please refer', 'BioModels Database', 'cite', 'citing',
            'terms of use', 'Li C', 'entitled to use', 'redistribute',
            'commercially', 'restricted way', 'verbatim'
        ]
        
        # Filter out lines containing boilerplate keywords
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if line and not any(keyword.lower() in line.lower() for keyword in boilerplate_keywords):
                filtered_lines.append(line)
        
        # Reassemble the filtered content with proper spacing
        model_notes = '\n'.join(filtered_lines)

    if isinstance(entity_type, str):
        try:
            entity_type = EntityType(entity_type)
        except ValueError:
            entity_type = EntityType.CHEMICAL

    ########## DISPLAY NAMES ##########
    all_display_names = get_species_display_names(model_file, entity_type)
    # filter to only include species_ids
    filtered_display_names = {id: all_display_names.get(id, "") for id in species_ids if id in all_display_names}

    ########## REACTIONS/TRANSITIONS ##########
    reactions = []
    
    if entity_type in (EntityType.GENE, EntityType.PROTEIN) and model_type == ModelType.SBML_QUAL:
        # For SBML-qual gene models, extract boolean transitions
        reactions = extract_qual_transitions(model_file, species_ids)
        
        # Update display names to include all species mentioned in transitions
        related_species = set(species_ids)
        for transition in reactions:
            # Extract species IDs from transition rules
            all_ids_in_transition = re.findall(r'\b([A-Za-z0-9_]+)\b', transition)
            related_species.update(all_ids_in_transition)
        
        # Filter display names to include our target species and all related species
        filtered_display_names = {species_id: all_display_names.get(species_id, "") for species_id in related_species if species_id in all_display_names}
        
    elif entity_type in (EntityType.GENE, EntityType.PROTEIN) and model_type == ModelType.SBML_FBC:
        # For SBML-fbc gene models, reactions are empty (genes don't participate in reactions directly)
        reactions = []
    
    else:
        # For chemical entities or regular SBML models, use antimony
        reactions, related_species = extract_reactions_from_sbml(model_file, species_ids)
    
        # Filter display names to include our target species and all related species
        filtered_display_names = {species_id: all_display_names.get(species_id, "") for species_id in related_species if species_id in all_display_names}
    
    return {
        "model_name": model_name,
        "model_type": model_type,
        "format_info": format_info,
        "display_names": filtered_display_names,
        "reactions": reactions,
        "model_notes": model_notes
    }

def format_prompt(
    model_file: str,
    species_ids: List[str],
    entity_type: str | EntityType = EntityType.CHEMICAL,
    top_k: int = 3,
    context: bool = True,
) -> str:
    """
    Format the information for the LLM prompt.
    Adapts format based on model type (SBML, SBML-fbc, SBML-qual).
    
    Args:
        model_file: Path to the SBML model file
        species_ids: List of species IDs to include in the prompt
        entity_type: Type of entity ("chemical" for species, "gene" for gene products, "protein" for protein products,
                        "reaction" for reactions, "auto" for automatic entity type detection)
        top_k: Number of synonyms to request from LLM (default: 3)
        context: If True, include full model context (model name, reactions, notes). 
                 If False, only include display names. (default: True)
        
    Returns:
        Formatted prompt string
    """
    # Helper function to get entity type options from enum
    def _get_entity_type_options() -> str:
        """Get comma-separated list of entity types from EntityType enum (excluding REACTION)."""
        exclude_types = ['reaction']
        types = [e.value for e in EntityType if e.value not in exclude_types]
        return ', '.join(types)
    if isinstance(entity_type, str):
        try:
            entity_type = EntityType(entity_type)
        except ValueError:
            entity_type = EntityType.CHEMICAL

    model_info = extract_model_info(model_file, species_ids, entity_type)
    if model_info == {}:
        return ""
    
    model_type = model_info.get("model_type", ModelType.SBML)
    entity_type_str = entity_type.value
    
    # Check if display_names are all empty
    display_names = model_info["display_names"]
    has_display_names = any(name and name.strip() for name in display_names.values())
    
    # Check if reactions list is not empty
    has_reactions = len(model_info["reactions"]) > 0
    
    # Check if model_notes are not empty
    has_notes = model_info["model_notes"] and model_info["model_notes"].strip()

    if model_type == ModelType.SBML_QUAL:
        if entity_type == EntityType.AUTO:
            # Auto entity type detection for SBML-qual models
            prompt = f"Now annotate these species:\nSpecies to annotate: {', '.join(species_ids)}\n"
            if context:
                prompt += f'Model: "{model_info["model_name"]}"\n'
            
            if has_display_names:
                prompt += "// Display Names:\n"
                for sid, name in display_names.items():
                    if name and name.strip():
                        prompt += f'{sid}:"{name}"; '
            
            if context and has_reactions:
                prompt += "\n"
                prompt += "// Boolean Transitions (target = rule):\n"
                prompt += '\n'.join(model_info["reactions"]) + "\n"
            
            if context and has_notes:
                prompt += "\n"
                prompt += f'// Notes:\n"{model_info["model_notes"]}"\n'
            
            entity_type_options = _get_entity_type_options()
            prompt += f"\nFor each species, determine its entity type ({entity_type_options}).\n"
            prompt += f"Return up to {top_k} standardized names or common synonyms for each species, ranked by likelihood.\n"
            prompt += f"Specify the entity type in parentheses after each species ID.\n"
            prompt += f"Use the below format, do not include any other text except the synonyms, and give short reasons for all species after 'Reason:' by the end.\n\n"
            prompt += 'SpeciesA (entity_type): "name1", "name2", …\nSpeciesB (entity_type): …\nReason: …'
            return prompt
        
        # SBML-qual models have boolean transitions
        prompt = f"Now annotate these:\n{entity_type_str.title()} to annotate: {', '.join(species_ids)}\n"
        if context:
            prompt += f'Model: "{model_info["model_name"]}"\n'
        
        if has_display_names:
            prompt += "// Display Names:\n"
            for sid, name in display_names.items():
                if name and name.strip():
                    prompt += f'{sid}:"{name}"; '
        
        if context and has_reactions:
            prompt += "\n"
            prompt += "// Boolean Transitions (target = rule):\n"
            prompt += '\n'.join(model_info["reactions"]) + "\n"
        
        if context and has_notes:
            prompt += "\n"
            prompt += f'// Notes:\n"{model_info["model_notes"]}"\n'
        
        prompt += f"\nReturn up to {top_k} standardized names or common synonyms for each {entity_type_str}, ranked by likelihood. Provide components names for complexes, which may exceed the limit of {top_k}.\n"
        prompt += f"Use the below format, do not include any other text except the synonyms, and give short reasons for all {entity_type_str}s after 'Reason:' by the end.\n\n"
        prompt += 'SpeciesA: "name1", "name2", …\nSpeciesB: …\nReason: …'
        return prompt

    elif model_type == ModelType.SBML_FBC:
        if entity_type == EntityType.AUTO:
            # Auto entity type detection for SBML-fbc models
            prompt = f"Now annotate these species:\nSpecies to annotate: {', '.join(species_ids)}\n"
            if context:
                prompt += f'Model: "{model_info["model_name"]}"\n'
            
            if has_display_names:
                prompt += "// Display Names:\n"
                for sid, name in display_names.items():
                    if name and name.strip():
                        prompt += f'{sid}:"{name}"; '
            
            if context and has_reactions:
                prompt += "\n"
                prompt += "// Reactions:\n"
                prompt += '\n'.join(model_info["reactions"]) + "\n"
            
            if context and has_notes:
                prompt += "\n"
                prompt += f'// Notes:\n"{model_info["model_notes"]}"\n'
            
            entity_type_options = _get_entity_type_options()
            prompt += f"\nFor each species, determine its entity type ({entity_type_options}).\n"
            prompt += f"Return up to {top_k} standardized names or common synonyms for each species, ranked by likelihood. Provide components names for complexes, which may exceed the limit of {top_k}.\n"
            prompt += f"Specify the entity type in parentheses after each species ID.\n"
            prompt += f"Use the below format, do not include any other text except the synonyms, and give short reasons for all species after 'Reason:' by the end.\n\n"
            prompt += 'SpeciesA (entity_type): "name1", "name2", …\nSpeciesB (entity_type): …\nReason: …'
            return prompt
        
        elif entity_type in (EntityType.GENE, EntityType.PROTEIN):
            # SBML-fbc models don't have reactions for genes or proteins
            prompt = f"Now annotate these:\n{entity_type_str.title()} to annotate: {', '.join(species_ids)}\n"
            if context:
                prompt += f'Model: "{model_info["model_name"]}"\n'
            
            if has_display_names:
                prompt += "// Display Names:\n"
                for sid, name in display_names.items():
                    if name and name.strip():
                        prompt += f'{sid}:"{name}"; '
            
            if context and has_notes:
                prompt += "\n"
                prompt += f'// Notes:\n"{model_info["model_notes"]}"\n'
            
            prompt += f"\nReturn up to {top_k} standardized names or common synonyms for each {entity_type_str}, ranked by likelihood.\n"
            prompt += f"Use the below format, do not include any other text except the synonyms, and give short reasons for all {entity_type_str}s after 'Reason:' by the end.\n\n"
            prompt += 'SpeciesA: "name1", "name2", …\nSpeciesB: …\nReason: …'
            return prompt
        
        else: # FBC, chemicals, same as SBML
            prompt = f"Now annotate these:\n{entity_type_str.title()} to annotate: {', '.join(species_ids)}\n"
            if context:
                prompt += f'Model: "{model_info["model_name"]}"\n'
            
            if has_display_names:
                prompt += "// Display Names:\n"
                for sid, name in display_names.items():
                    if name and name.strip():
                        prompt += f'{sid}:"{name}"; '
            
            if context and has_reactions:
                prompt += "\n"
                prompt += "// Reactions:\n"
                prompt += '\n'.join(model_info["reactions"]) + "\n"
            
            if context and has_notes:
                prompt += "\n"
                prompt += f'// Notes:\n"{model_info["model_notes"]}"\n'
            
            prompt += f"\nReturn up to {top_k} standardized names or common synonyms for each {entity_type_str}, ranked by likelihood. Provide components names for complexes, which may exceed the limit of {top_k}.\n"
            prompt += f"Use the below format, do not include any other text except the synonyms, and give short reasons for all {entity_type_str}s after 'Reason:' by the end.\n\n"
            prompt += 'SpeciesA: "name1", "name2", …\nSpeciesB: …\nReason: …'
            return prompt             
    
    else:  # SBML
        if entity_type == EntityType.REACTION:
            reaction_display_names = get_reaction_display_names(model_file)
            reaction_ids = get_all_reaction_ids(model_file)
            
            # Check if reaction display names are all empty
            has_reaction_display_names = any(name and name.strip() for name in reaction_display_names.values())

            prompt = "Now annotate these metabolic reactions using KEGG data:\n"
            prompt += f"Reactions to annotate: {', '.join(reaction_ids)}\n"
            if context:
                prompt += f'Model: "{model_info["model_name"]}"\n'
            
            if has_reaction_display_names:
                prompt += "// Display Names:\n"
                for rid, name in reaction_display_names.items():
                    if name and name.strip():
                        prompt += f'{rid}:"{name}"; '
            
            if context and has_reactions:
                prompt += "\n"
                prompt += "// Reactions:\n"
                prompt += '\n'.join(model_info["reactions"]) + "\n"
            
            if context and has_notes:
                prompt += "\n"
                prompt += f'// Notes:\n"{model_info["model_notes"]}"\n'
            
            prompt += f"\nReturn up to {top_k} standardized names or common synonyms for each reaction, ranked by likelihood.\n"
            prompt += f"Use the below format, do not include any other text except the synonyms, and give short reasons for all {entity_type_str} after 'Reason:' by the end.\n\n"
            prompt += 'ReactionA: "name1", "name2", …\nReactionB: …\nReason: …'
            return prompt
        
        elif entity_type == EntityType.AUTO:
            # Auto entity type detection for regular SBML models
            prompt = f"Now annotate these species:\nSpecies to annotate: {', '.join(species_ids)}\n"
            if context:
                prompt += f'Model: "{model_info["model_name"]}"\n'
            
            if has_display_names:
                prompt += "// Display Names:\n"
                for sid, name in display_names.items():
                    if name and name.strip():
                        prompt += f'{sid}:"{name}"; '
            
            if context and has_reactions:
                prompt += "\n"
                prompt += "// Reactions:\n"
                prompt += '\n'.join(model_info["reactions"]) + "\n"
            
            if context and has_notes:
                prompt += "\n"
                prompt += f'// Notes:\n"{model_info["model_notes"]}"\n'
            
            entity_type_options = _get_entity_type_options()
            prompt += f"\nFor each species, determine its entity type ({entity_type_options}).\n"
            prompt += f"Return up to {top_k} standardized names or common synonyms for each species, ranked by likelihood. Provide components names for complexes, which may exceed the limit of {top_k}.\n"
            prompt += f"Specify the entity type in parentheses after each species ID.\n"
            prompt += f"Use the below format, do not include any other text except the synonyms, and give short reasons for all species after 'Reason:' by the end.\n\n"
            prompt += 'SpeciesA (entity_type): "name1", "name2", …\nSpeciesB (entity_type): …\nReason: …'
            return prompt
        
        else:
            prompt = f"Now annotate these:\n{entity_type.title()} to annotate: {', '.join(species_ids)}\n"
            if context:
                prompt += f'Model: "{model_info["model_name"]}"\n'
            
            if has_display_names:
                prompt += "// Display Names:\n"
                for sid, name in display_names.items():
                    if name and name.strip():
                        prompt += f'{sid}:"{name}"; '
            
            if context and has_reactions:
                prompt += "\n"
                prompt += "// Reactions:\n"
                prompt += '\n'.join(model_info["reactions"]) + "\n"
            
            if context and has_notes:
                prompt += "\n"
                prompt += f'// Notes:\n"{model_info["model_notes"]}"\n'
            
            if context:
                prompt += f"\nReturn up to {top_k} standardized names or common synonyms for each {entity_type}, ranked by likelihood. Provide components names for complexes, which may exceed the limit of {top_k}.\n"
                prompt += f"Use the below format, do not include any other text except the synonyms, and give short reasons for all {entity_type}s after 'Reason:' by the end.\n\n"
                prompt += 'SpeciesA: "name1", "name2", …\nSpeciesB: …\nReason: …'
            else:
                prompt += f"\nReturn up to {top_k} standardized names or common synonyms for each {entity_type}, ranked by likelihood. Provide components names for complexes, which may exceed the limit of {top_k}.\n"
                prompt += f"Use the below format, do not include any other text except the synonyms.\n\n"
                prompt += 'SpeciesA: "name1", "name2", …\nSpeciesB: …'
            return prompt 