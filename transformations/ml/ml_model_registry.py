import dlt
from pyspark.sql import functions as F

# Déclaration de la Streaming Table cible
dlt.create_streaming_table(
    name="ml_model_registry",
    comment="Registre historique comparatif des versions du modèle (v1.0, v2.0, ...)"
)

@dlt.append_flow(target="ml_model_registry")
def ml_model_registry_flow():
    """
    append_flow avec readStream :
    Chaque exécution AJOUTE une ligne sans écraser l'historique.
    Changer model_version ("v1.0" → "v2.0") entre deux runs pour comparer.
    """
    return (
        # ← Correction clé : readStream au lieu de dlt.read
        spark.readStream.table("taxi_mlops_prod.medoune_fall.ml_model_training")
        .withColumn("model_name",    F.lit("taxi_fare_prediction_model"))
        .withColumn("model_version", F.lit("v1.0"))   # ← changer ici entre les runs
        .withColumn("model_status",  F.lit("active"))
        .withColumn("catalog",       F.lit("taxi_mlops_prod"))
        .withColumn("schema",        F.lit("medoune_fall"))
        .withColumn("description",   F.lit("Baseline v1.0 : sans nouvelles features"))
        .withColumn("registered_at", F.current_timestamp())
        .select(
            "model_name",
            "model_version",
            "model_type",
            "model_status",
            "catalog",
            "schema",
            "rmse",
            "mae",
            "correlation",
            "description",
            "training_timestamp",
            "registered_at"
        )
    )