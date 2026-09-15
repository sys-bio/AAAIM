"""Manual smoke script for automatic entity-type detection.

The API-backed smoke run is guarded so importing this module during normal
pytest collection does not make a live request or write result artifacts.
"""

from utils.evaluation import evaluate_single_model


TEST_MODEL_FILE = "/Users/luna/Desktop/CRBM/AMAS_proj/Models/BioModels/BIOMD0000000039.xml"


def main() -> None:
    print("=" * 80)
    print("Testing Automatic Entity Type Detection")
    print("=" * 80)
    print(f"\nModel: {TEST_MODEL_FILE}")
    print("\nConfiguration:")
    print("  - entity_type: auto")
    print("  - database: ['chebi', 'uniprot']")
    print("  - method: direct")
    print("  - llm_model: meta-llama/llama-3.3-70b-instruct:free")
    print("  - top_k: 3")
    print("  - max_entities: 10")
    print("\n" + "=" * 80)

    result_df = evaluate_single_model(
        model_file=TEST_MODEL_FILE,
        llm_model="meta-llama/llama-3.3-70b-instruct:free",
        method="direct",
        top_k=3,
        max_entities=10,
        entity_type="auto",
        database=["chebi", "uniprot"],
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    if result_df is None or result_df.empty:
        print("\nNo results generated. Check the logs above for errors.")
        return

    print(f"\nTotal species evaluated: {len(result_df)}")
    print("\nDetected Entity Types:")
    for entity_type, count in result_df["detected_entity_type"].value_counts().items():
        print(f"  - {entity_type}: {count}")

    print("\nSample Results (first 5 species):")
    print("\n" + "-" * 80)
    for _, row in result_df.head(5).iterrows():
        print(f"\nSpecies ID: {row['species_id']}")
        print(f"  Display Name: {row['display_name']}")
        print(f"  Detected Type: {row['detected_entity_type']}")
        print(f"  LLM Synonyms: {row['synonyms_LLM']}")
        print(f"  Predictions: {row['predictions']}")
        print(f"  Prediction Names: {row['predictions_names']}")
        print(f"  Accuracy: {row['accuracy']}")

    output_file = "test_auto_detection_results.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
