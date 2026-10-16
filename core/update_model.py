import pandas as pd
import libsbml
import re
from core.model_info import detect_model_format

# XML helpers (adapted from AMAS)
def divide_existing_annotation(inp_str, qualifier):
    """
    Parse the annotation string to extract items in <rdf:Bag> for the specified qualifier.
    Returns dict with 'container' (annotation with empty block) and 'items' (list of <rdf:li> lines).
    """
    qualifier_pattern = rf"(<{qualifier}[^>]*?>\s*<rdf:Bag>.*?</rdf:Bag>\s*</{qualifier}>)"
    qualifier_matches = re.findall(qualifier_pattern, inp_str, re.DOTALL)
    if not qualifier_matches:
        return None
    rdf_li_pattern = r"<rdf:li[^>]*/>"
    items = []
    for block in qualifier_matches:
        items.extend(re.findall(rdf_li_pattern, block))
    match_prefix = re.match(rf"<{qualifier}.*?>", qualifier_matches[0])
    if match_prefix:
        qualifier_opening = match_prefix.group()
    else:
        qualifier_opening = f"<{qualifier}>"
    empty_qualifier_block = (
        f"      {qualifier_opening}\n        <rdf:Bag>\n        </rdf:Bag>\n      </{qualifier}>"
    )
    stripped_annotation = re.sub(qualifier_pattern, "", inp_str, flags=re.DOTALL).strip()
    container = stripped_annotation.replace(
        "</rdf:Description>",
        f"{empty_qualifier_block}\n    </rdf:Description>"
    )
    return {"container": container, "items": items}

def insert_items_back_to_container(container, items, qualifier):
    """
    Insert <rdf:li> items back into the <rdf:Bag> of the specified qualifier in the container.
    """
    bag_pattern = rf"(<{qualifier}[^>]*?>\s*<rdf:Bag>).*?</rdf:Bag>\s*</{qualifier}>"
    match = re.search(bag_pattern, container, re.DOTALL)
    items_str = "\n".join([f"          {item}" for item in items])
    if match:
        qualifier_opening = match.group(1)
        updated_bag = f"{qualifier_opening}\n{items_str}\n        </rdf:Bag>"
        updated_container = re.sub(bag_pattern, updated_bag + f"\n      </{qualifier}>", container, flags=re.DOTALL)
    else:
        qualifier_opening = f"<{qualifier} xmlns:bqbiol=\"http://biomodels.net/biology-qualifiers/\">"
        updated_bag = f"      {qualifier_opening}\n        <rdf:Bag>\n{items_str}\n        </rdf:Bag>\n      </{qualifier}>"
        updated_container = container.replace(
            "</rdf:Description>",
            f"{updated_bag}\n    </rdf:Description>"
        )
    return updated_container

def annotation_resource(annotation):
    """identifiers.org URI for a table value such as UNIPROT:P15056 or CHEBI:15996."""
    prefix, ident = str(annotation).split(':', 1)
    prefix, ident = prefix.strip().lower(), ident.strip()
    if prefix == 'chebi':
        if not ident.upper().startswith('CHEBI:'):
            ident = f'CHEBI:{ident}'
        return f'http://identifiers.org/chebi/{ident}'
    if prefix == 'uniprot':
        return f'http://identifiers.org/uniprot/{ident}'
    if prefix == 'ncbigene':
        return f'http://identifiers.org/ncbigene/{ident}'
    if prefix == 'kegg' and ident[:1].upper() == 'R':
        return f'http://identifiers.org/kegg.reaction/{ident}'
    if prefix == 'kegg' and ident[:1].upper() == 'C':
        return f'http://identifiers.org/kegg.compound/{ident}'
    return f'http://identifiers.org/{prefix}/{ident}'


def _resource_key(uri):
    return uri.strip().replace('https://', 'http://').rstrip('/').lower()


def _item_resource(item):
    match = re.search(r'rdf:resource="([^"]+)"', item)
    return match.group(1) if match else ''


def create_annotation_item(annotation):
    return f'<rdf:li rdf:resource="{annotation_resource(annotation)}"/>'


def _row_qualifier(row, default):
    raw = row.get('qualifier', default)
    if pd.isna(raw) or str(raw).strip().lower() in {'', 'na', 'nan', 'none'}:
        return default
    qualifier = str(raw).strip()
    if qualifier.lower().startswith('bqbiol:'):
        qualifier = qualifier.split(':', 1)[1]
    return qualifier

def clean_items(items):
    # Only deduplicate by full item string
    return sorted(set(items))

# Main update logic
def update_annotation(
    original_model_path, 
    recommendation_table = None, 
    new_model_path = None, 
    qualifier='is',
    add_all=False,
    ):
    """
    Update SBML model annotations based on a recommendation table.
    Only add/delete the specified ontology (e.g., CHEBI), preserving all other annotation terms (e.g., KEGG).

    add_all: add every annotation in the table and ignore the update_annotation column.
    """
    if recommendation_table is None:
        recommendation_table = original_model_path + '_species.csv'
    if new_model_path is None:
        new_model_path = original_model_path + '_updated.xml'
    # Load model
    reader = libsbml.SBMLReader()
    document = reader.readSBML(original_model_path)
    model = document.getModel()
    if model is None:
        raise ValueError(f"Could not load SBML model: {original_model_path}")
    # Load table
    if isinstance(recommendation_table, str):
        df = pd.read_csv(recommendation_table)
    else:
        df = recommendation_table.copy()
    # Skip Reason row if present
    if not df.empty and 'id' in df.columns:
        reason_rows = df[df['id'] == 'Reason:']
        if not reason_rows.empty:
            reason_text = reason_rows.iloc[0].get('display_name', '')
            # if reason_text:
            #     print(f"LLM Reason: {reason_text}")
        df = df[df['id'] != 'Reason:'].copy()
    # Get model type
    model_type, _ = detect_model_format(original_model_path)
    # Group by species
    grouped = df.groupby('id')
    added, removed, updated_species = 0, 0, set()
    for species_id, group in grouped:
        to_add, to_delete = [], []
        for _, row in group.iterrows():
            action = 'add' if add_all else str(row.get('update_annotation', 'ignore')).lower()
            annotation = row.get('annotation', '')
            if pd.isna(annotation) or ':' not in str(annotation):
                continue
            annotation = str(annotation).strip()
            row_qualifier = _row_qualifier(row, qualifier)
            resource_key = _resource_key(annotation_resource(annotation))
            if action == 'add':
                to_add.append((annotation, row_qualifier, resource_key))
            elif action == 'delete':
                to_delete.append(resource_key)
        sbml_elem = None
        if model_type.name.startswith('SBML'):
            sbml_elem = model.getElementBySId(species_id)
        elif model_type.name == 'SBML_FBC':
            sbml_elem = model.getElementBySId(species_id)
            if sbml_elem is None and hasattr(model, 'getPlugin'):
                fbc = model.getPlugin('fbc')
                if fbc:
                    sbml_elem = fbc.getGeneProduct(species_id)
        elif model_type.name == 'SBML_QUAL':
            qual = model.getPlugin('qual')
            if qual:
                sbml_elem = qual.getQualitativeSpecies(species_id)
        if sbml_elem is None:
            continue
        if sbml_elem.isSetAnnotation():
            annotation_str = sbml_elem.getAnnotation().toXMLString()
        else:
            annotation_str = ''
        # Handle existing annotations - preserve all qualifier blocks
        existing_items = []
        if annotation_str:
            # Extract all bqbiol qualifier blocks
            qualifier_blocks = re.findall(r'<bqbiol:([^>]+)[^>]*?>(.*?)</bqbiol:\1>', annotation_str, flags=re.DOTALL)
            for qual_name, block_content in qualifier_blocks:
                # Extract items from this qualifier block
                rdf_li_pattern = r"<rdf:li[^>]*/>"
                items = re.findall(rdf_li_pattern, block_content)
                for item in items:
                    existing_items.append((item, qual_name))
        
        removed_this_species = 0
        if not to_add and not to_delete and not existing_items:
            continue

        # Group existing items by qualifier
        existing_by_qualifier = {}
        for item, qual_name in existing_items:
            existing_by_qualifier.setdefault(qual_name, []).append(item)

        delete_keys = set(to_delete)
        for qual_name, items in list(existing_by_qualifier.items()):
            kept = []
            for item in items:
                if _resource_key(_item_resource(item)) in delete_keys:
                    removed += 1
                    removed_this_species += 1
                else:
                    kept.append(item)
            existing_by_qualifier[qual_name] = kept

        existing_keys = {
            _resource_key(_item_resource(item))
            for items in existing_by_qualifier.values()
            for item in items
        }
        new_by_qualifier = {}
        for annotation, row_qualifier, resource_key in to_add:
            if resource_key in existing_keys:
                continue
            existing_keys.add(resource_key)
            new_by_qualifier.setdefault(row_qualifier, []).append(create_annotation_item(annotation))
            added += 1
        
        # Merge existing and new items by qualifier
        final_by_qualifier = {}
        for qual_name in set(list(existing_by_qualifier.keys()) + list(new_by_qualifier.keys())):
            final_by_qualifier[qual_name] = []
            if qual_name in existing_by_qualifier:
                final_by_qualifier[qual_name].extend(existing_by_qualifier[qual_name])
            if qual_name in new_by_qualifier:
                final_by_qualifier[qual_name].extend(new_by_qualifier[qual_name])
        
        # Build new annotation string. libsbml drops the annotation when bqbiol is undeclared.
        # It also rejects the annotation when the element has no metaid.
        if final_by_qualifier:
            if not sbml_elem.isSetMetaId():
                sbml_elem.setMetaId(str(species_id))
            meta_id = sbml_elem.getMetaId() or species_id
            new_anno = (
                '<annotation>\n'
                '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
                'xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">\n'
                f'    <rdf:Description rdf:about="#{meta_id}">\n'
            )
            
            for qual_name, items in final_by_qualifier.items():
                if items:  # Only add qualifier blocks with items
                    qualifier_tag = f'bqbiol:{qual_name}'
                    new_anno += f'      <{qualifier_tag}>\n        <rdf:Bag>\n'
                    for item in items:
                        new_anno += f'          {item}\n'
                    new_anno += f'        </rdf:Bag>\n      </{qualifier_tag}>\n'
            
            new_anno += '    </rdf:Description>\n  </rdf:RDF>\n</annotation>'
        else:
            new_anno = ''
        sbml_elem.setAnnotation(new_anno)
        if to_add or removed_this_species:
            updated_species.add(species_id)
    writer = libsbml.SBMLWriter()
    writer.writeSBMLToFile(document, new_model_path)
    print(f"Update complete. {len(updated_species)} species updated. {added} annotations added, {removed} removed.")
