"""
Feedback Module for AAAIM

Allows users to provide feedback on annotation recommendations and
iteratively refine them through multi-turn LLM conversations.

Main entry point for users is the ``AnnotationResult`` class returned by
``annotate_model`` / ``curate_model``.  It exposes two simple methods:

    result = annotate_model("model.xml", ...)
    result = result.revise("Species X should be glucose-6-phosphate")
    # or interactively:
    result.feedback_loop()
"""

import time
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import logging

from core.model_info import extract_model_info
from core.llm_interface import (
    query_llm_message_with_history,
    parse_llm_response,
)

logger = logging.getLogger(__name__)

FEEDBACK_PROMPT_TEMPLATE = (
    "The user reviewed your previous recommendations and has the following feedback:\n"
    "---\n"
    "{feedback}\n"
    "---\n"
    "Please revise your response for the species mentioned above, "
    "taking the user's feedback into account.  Keep the same output format."
)


# ---------------------------------------------------------------------------
# AnnotationResult – the simple user-facing wrapper
# ---------------------------------------------------------------------------

class AnnotationResult:
    """Bundle returned by ``annotate_model`` / ``curate_model``.

    Supports tuple unpacking for backward compatibility::

        recommendations_df, metrics = annotate_model(...)

    And provides feedback helpers::

        result = annotate_model(...)
        result = result.revise("glucose should map to CHEBI:17234")
        result.feedback_loop()
    """

    def __init__(
        self,
        recommendations_df: pd.DataFrame,
        metrics: Dict[str, Any],
        *,
        model_file: str,
        conversation_history: List[Dict[str, Any]],
        entities_to_evaluate: List[str],
        entity_type: str,
        database: str,
        method: str,
        llm_model: str,
        top_k: int,
        n_return: int = 3,
        tax_id: str = None,
        existing_annotations: Optional[Dict[str, List[str]]] = None,
        qualifier_annotations: Optional[Dict[str, List[str]]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        csv_path: str = None,
        validation: bool = False,
    ):
        self.recommendations_df = recommendations_df
        self.species_recommendations_df = None
        self.reaction_recommendations_df = None
        self.metrics = metrics

        # Internal context (not exposed in metrics)
        self._model_file = model_file
        self._conversation_history = list(conversation_history)
        self._entities_to_evaluate = entities_to_evaluate
        self._entity_type = entity_type
        self._database = database
        self._method = method
        self._llm_model = llm_model
        self._top_k = top_k
        self._n_return = n_return
        self._tax_id = tax_id
        self._existing_annotations = existing_annotations or {}
        self._qualifier_annotations = qualifier_annotations or {}
        self._model_info = model_info
        self._csv_path = csv_path
        self._validation = validation
        self._revision_count = 0
        self._revision_history: List[Dict[str, Any]] = []

    # -- backward-compatible tuple unpacking ----------------------------------
    def __iter__(self):
        """``df, metrics = annotate_model(...)`` still works."""
        return iter((self.recommendations_df, self.metrics))

    # -- simple feedback API --------------------------------------------------
    def revise(self, feedback: str) -> "AnnotationResult":
        """Revise recommendations with a single round of user feedback.

        Args:
            feedback: Free-text description of what to change.

        Returns:
            ``self`` (mutated) with updated ``recommendations_df``.
            A versioned CSV is saved automatically so previous results
            are never overwritten.
        """
        self._revision_count += 1
        print(f"Revising recommendations (v{self._revision_count})...")

        updated_df, revision_metrics, updated_history = _revise_recommendations(
            model_file=self._model_file,
            previous_recommendations_df=self.recommendations_df,
            feedback=feedback,
            conversation_history=self._conversation_history,
            entities_to_evaluate=self._entities_to_evaluate,
            entity_type=self._entity_type,
            database=self._database,
            method=self._method,
            llm_model=self._llm_model,
            top_k=self._top_k,
            n_return=self._n_return,
            tax_id=self._tax_id,
            existing_annotations=self._existing_annotations,
            qualifier_annotations=self._qualifier_annotations,
            model_info=self._model_info,
            validation=self._validation,
        )

        revision_metrics["iteration"] = self._revision_count
        self._revision_history.append(revision_metrics)
        self._conversation_history = updated_history
        self.recommendations_df = updated_df

        csv_path = _versioned_csv_path(self._csv_path or self._model_file, self._revision_count)
        updated_df.to_csv(csv_path, index=False)
        print(f"Revised recommendations (v{self._revision_count}) saved to {csv_path}")

        return self

    def feedback_loop(
        self,
        max_iterations: int = 10,
        get_feedback_fn: Callable[[pd.DataFrame, int], str] = None,
    ) -> "AnnotationResult":
        """Run an interactive feedback loop.

        Each round displays the current recommendations, collects feedback,
        and revises.  The loop ends when the user submits empty input or
        ``max_iterations`` is reached.

        Args:
            max_iterations: Safety cap on revision rounds.
            get_feedback_fn: ``(recommendations_df, iteration) -> str``.
                Return ``""`` or ``None`` to accept and stop.
                Defaults to a console ``input()`` prompt.

        Returns:
            ``self`` with the final ``recommendations_df``.
        """
        if get_feedback_fn is None:
            get_feedback_fn = _default_get_feedback

        for _ in range(max_iterations):
            iteration = self._revision_count + 1
            feedback = get_feedback_fn(self.recommendations_df, iteration)
            print("Feedback received: ", feedback)

            if not feedback or not feedback.strip():
                print("Feedback accepted – no further revisions.")
                break

            self.revise(feedback)

            if self._revision_history and self._revision_history[-1].get("error"):
                logger.warning(
                    f"Revision round {self._revision_count} encountered an error: "
                    f"{self._revision_history[-1]['error']}"
                )
        else:
            print(f"Reached maximum iterations ({max_iterations}).")

        return self

    @property
    def revision_history(self) -> List[Dict[str, Any]]:
        """Per-round metrics from all feedback revisions so far."""
        return list(self._revision_history)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _versioned_csv_path(csv_path: str, revision: int) -> str:
    """Generate ``<saved-prefix>_v<N>.csv`` from the last written file."""
    p = Path(csv_path)
    return str(p.with_name(f"{p.stem}_v{revision}{p.suffix}"))


def _build_feedback_prompt(feedback: str) -> str:
    return FEEDBACK_PROMPT_TEMPLATE.format(feedback=feedback)


def _format_recommendations_summary(
    recommendations_df: pd.DataFrame,
    entity_ids: Optional[List[str]] = None,
) -> str:
    """Concise text summary of current recommendations for the LLM."""
    if recommendations_df.empty:
        return "(no recommendations)"

    df = recommendations_df
    df = df[df["id"] != "Reason:"]
    if entity_ids:
        df = df[df["id"].isin(entity_ids)]

    lines = []
    for entity_id, group in df.groupby("id", sort=False):
        display = group["display_name"].iloc[0]
        candidates = []
        for _, row in group.iterrows():
            ann = row.get("annotation", "")
            label = row.get("annotation_label", "")
            score = row.get("match_score", "")
            if ann:
                candidates.append(f"{ann} ({label}, score={score})")
        if candidates:
            lines.append(f"{entity_id} ({display}): {'; '.join(candidates)}")
        else:
            lines.append(f"{entity_id} ({display}): no candidates")
    return "\n".join(lines)


def _revise_recommendations(
    model_file: str,
    previous_recommendations_df: pd.DataFrame,
    feedback: str,
    conversation_history: List[Dict[str, Any]],
    entities_to_evaluate: List[str],
    entity_type: str = "chemical",
    database: str = "chebi",
    method: str = "direct",
    llm_model: str = "gpt-4o-mini",
    top_k: int = 3,
    n_return: int = 3,
    tax_id: str = None,
    existing_annotations: Optional[Dict[str, List[str]]] = None,
    qualifier_annotations: Optional[Dict[str, List[str]]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    validation: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """Single feedback revision round (internal implementation)."""
    start_time = time.time()

    if model_info is None:
        model_info = extract_model_info(model_file, entities_to_evaluate, entity_type)
    if existing_annotations is None:
        existing_annotations = {}
    if qualifier_annotations is None:
        qualifier_annotations = {}

    rec_summary = _format_recommendations_summary(
        previous_recommendations_df, entities_to_evaluate
    )

    history = list(conversation_history)
    history.append({
        "role": "user",
        "content": (
            f"Here are the database-matched recommendations based on your previous response:\n"
            f"{rec_summary}\n\n"
            f"{_build_feedback_prompt(feedback)}"
        ),
    })

    logger.info("Querying LLM with feedback (revision round)...")
    llm_start = time.time()
    response_format = None
    if validation:
        from core.annotation_workflow import _normalization_response_format, _normalize_entity_type
        response_format = _normalization_response_format(_normalize_entity_type(entity_type))
    assistant_message = query_llm_message_with_history(
        history,
        model=llm_model,
        response_format=response_format,
    )
    llm_response = assistant_message.get("content") if assistant_message else ""
    llm_time = time.time() - llm_start

    if not llm_response:
        logger.error("No response from LLM during feedback revision")
        return previous_recommendations_df, {"error": "No LLM response"}, history

    history.append(assistant_message)

    if validation:
        from core.annotation_workflow import _normalize_entity_type, _parse_structured_normalization
        try:
            synonyms_dict, entity_type_dict, reason, component_dict = (
                _parse_structured_normalization(
                    llm_response,
                    entities_to_evaluate,
                    _normalize_entity_type(entity_type),
                )
            )
        except ValueError as exc:
            logger.warning("Structured feedback response failed validation: %s", exc)
            return previous_recommendations_df, {"error": str(exc)}, history
    else:
        synonyms_dict, entity_type_dict, reason, component_dict = parse_llm_response(
            llm_response, entity_type
        )

    if reason:
        print(f"LLM Reason: {reason}")

    if not synonyms_dict:
        logger.warning(
            "LLM revision produced no parseable synonyms; keeping previous recommendations"
        )
        return previous_recommendations_df, {"error": "Failed to parse revised response"}, history

    logger.info(f"Parsed revised synonyms for {len(synonyms_dict)} entities")

    search_start = time.time()
    from core.annotation_workflow import (
        _generate_recommendation_table,
        _normalize_databases,
        _normalize_entity_type,
        _search_databases,
        rank_species_annotations_with_llm,
    )

    entity_type = _normalize_entity_type(entity_type)
    databases = _normalize_databases(database)
    recommendations, species_database, candidate_databases = _search_databases(
        entities_to_evaluate,
        synonyms_dict,
        entity_type=entity_type,
        databases=databases,
        method=method,
        top_k=top_k,
        tax_id=tax_id,
        entity_type_dict=entity_type_dict,
        model_info=model_info,
        component_dict=component_dict,
        validation=validation,
        model_file=model_file,
    )
    search_time = time.time() - search_start

    updated_df = _generate_recommendation_table(
        model_file, recommendations, existing_annotations,
        model_info, entity_type.value, [db.value for db in databases], qualifier_annotations,
        synonyms_dict=synonyms_dict, reason=reason,
        entity_type_dict=entity_type_dict,
        species_database=species_database,
        candidate_databases=candidate_databases,
        component_dict=component_dict,
    )
    if top_k > n_return:
        ranked_df = rank_species_annotations_with_llm(
            model_file,
            updated_df,
            llm_model=llm_model,
            n_return=n_return,
            model_notes=(model_info or {}).get("model_notes", "") or "",
            validation=validation,
        )
        if not ranked_df.empty:
            updated_df = ranked_df

    total_time = time.time() - start_time
    metrics = {
        "feedback_round": True,
        "total_time": total_time,
        "llm_time": llm_time,
        "search_time": search_time,
        "entities_revised": len(synonyms_dict),
    }

    logger.info(
        f"Feedback revision completed in {total_time:.2f}s "
        f"({len(synonyms_dict)} entities revised)"
    )

    return updated_df, metrics, history


def _run_database_search(
    entities: List[str],
    synonyms_dict: Dict[str, List[str]],
    database: str,
    method: str,
    top_k: int,
    tax_id: str = None,
    entity_type: str = "chemical",
    entity_type_dict: Optional[Dict[str, str]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    component_dict: Optional[Dict] = None,
):
    """Thin dispatcher that mirrors the search logic in annotation_workflow."""
    from core.annotation_workflow import _normalize_databases, _normalize_entity_type, _search_databases

    databases = _normalize_databases(database)
    entity_type = _normalize_entity_type(entity_type)
    recommendations, _, _ = _search_databases(
        entities,
        synonyms_dict,
        entity_type=entity_type,
        databases=databases,
        method=method,
        top_k=top_k,
        tax_id=tax_id,
        entity_type_dict=entity_type_dict,
        model_info=model_info,
        component_dict=component_dict,
    )
    return recommendations


def build_initial_conversation(
    system_prompt: str,
    user_prompt: str,
    llm_response: str,
    assistant_message: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Construct the initial conversation history from the first annotation round."""
    if assistant_message is None:
        assistant_message = {"role": "assistant", "content": llm_response}
    else:
        assistant_message = dict(assistant_message)
        assistant_message.setdefault("role", "assistant")
        assistant_message.setdefault("content", llm_response)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        assistant_message,
    ]


def _default_get_feedback(recommendations_df: pd.DataFrame, iteration: int) -> str:
    """Console-based feedback collector."""
    print(f"\n{'='*60}")
    print(f"  Feedback Round {iteration}")
    print(f"{'='*60}")
    print("\nCurrent recommendations:\n")
    if recommendations_df.empty:
        print("  (no recommendations)")
    else:
        display_df = recommendations_df[recommendations_df['id'] != 'Reason:'] if 'id' in recommendations_df.columns else recommendations_df
        cols = ["id", "display_name", "curated_name", "annotation", "annotation_label",
                "match_score", "status", "update_annotation", "comment"]
        display_cols = [c for c in cols if c in display_df.columns]
        print(display_df[display_cols].to_string(index=False))

    print("\nProvide feedback to revise recommendations.")
    print("Press Enter with no input to accept and finish.\n")
    return input("Feedback> ")
