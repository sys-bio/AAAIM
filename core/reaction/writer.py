"""Explicit additive reaction writer; recommendations never trigger SBML edits."""
from pathlib import Path
import re

import libsbml
import pandas as pd

from .species_evidence import load_model


def write_reaction_annotations(model_file, recommendations, output_file):
    """Write only explicitly approved, successful selections; preserve all CVTerms.

    Set update_annotation='add' on reviewed rows. Neither confidence nor a
    returned prediction constitutes approval. The original path cannot be used
    as output. Species qualifiers are not accepted by this reaction-only writer.
    """
    if Path(model_file).resolve() == Path(output_file).resolve():
        raise ValueError("Choose a separate output SBML path")
    if Path(output_file).exists():
        raise FileExistsError(output_file)
    frame = pd.read_csv(recommendations).fillna("") if isinstance(recommendations, (str, Path)) else recommendations
    document = load_model(model_file)
    model = document.getModel()
    additions = []
    for row in frame.to_dict("records"):
        if row.get("update_annotation") != "add":
            continue
        annotation = str(row.get("annotation", ""))
        reaction = model.getReaction(str(row["id"]))
        if (reaction is None or row.get("status") != "selected" or row.get("qualifier", "is") != "is"
                or not re.fullmatch(r"KEGG:R\d{5}", annotation)):
            raise ValueError("Only selected, reviewed KEGG reaction identities can be written")
        additions.append((reaction, annotation.split(":")[1]))
    for reaction, identifier in additions:
        uri = "https://identifiers.org/kegg.reaction/" + identifier
        present = any(term.getBiologicalQualifierType() == libsbml.BQB_IS and
                      any(term.getResourceURI(i).rstrip("/").endswith(identifier) for i in range(term.getNumResources()))
                      for term in reaction.getCVTerms() or [])
        if present:
            continue
        if not reaction.isSetMetaId():
            base = "aaaim_reaction_" + reaction.getId()
            meta_id = base
            index = 0
            while model.getElementByMetaId(meta_id) is not None:
                index += 1
                meta_id = base + "_" + str(index)
            reaction.setMetaId(meta_id)
        term = libsbml.CVTerm(libsbml.BIOLOGICAL_QUALIFIER)
        term.setBiologicalQualifierType(libsbml.BQB_IS)
        term.addResource(uri)
        if reaction.addCVTerm(term) != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise ValueError("Could not add reaction CVTerm")
    if not libsbml.writeSBMLToFile(document, str(output_file)):
        raise OSError("Could not save output SBML")
    return str(output_file)
