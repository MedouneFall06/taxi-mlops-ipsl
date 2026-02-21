# import dlt
# from pyspark.sql import functions as F

# @dlt.table(
#     comment="Model training results and evaluation metrics for taxi fare prediction"
# )
# def ml_model_training():
#     """
#     ML Model Training and Validation
#     - Prépare les données d'entraînement avec feature engineering
#     - Calcule les prédictions via une régression linéaire simplifiée
#     - Retourne les métriques d'évaluation (RMSE, MAE, Corrélation)
#     """
#     # Lecture via dlt.read pour que le DAG de dépendances soit correctement résolu
#     df = dlt.read("ml_training_data")

#     # Split train / test
#     train_df = df.filter("is_training = true")
#     test_df  = df.filter("is_training = false")

#     # -------------------------------------------------------------------------
#     # FEATURE ENGINEERING — TRAIN
#     # -------------------------------------------------------------------------
#     train_encoded = (
#         train_df
#         # Encode time_of_day
#         .withColumn("time_morning",   F.when(F.col("time_of_day") == "morning",   1.0).otherwise(0.0))
#         .withColumn("time_afternoon", F.when(F.col("time_of_day") == "afternoon", 1.0).otherwise(0.0))
#         .withColumn("time_evening",   F.when(F.col("time_of_day") == "evening",   1.0).otherwise(0.0))
#         .withColumn("time_night",     F.when(F.col("time_of_day") == "night",     1.0).otherwise(0.0))

#         # FEATURE 1 : is_rush_hour (Booléen → Numérique)
#         .withColumn("rush_hour_flag", F.col("is_rush_hour").cast("double"))

#         # FEATURE 2 : One-Hot Encoding de trip_distance_segment
#         .withColumn("dist_short",  F.when(F.col("trip_distance_segment") == "short",  1.0).otherwise(0.0))
#         .withColumn("dist_medium", F.when(F.col("trip_distance_segment") == "medium", 1.0).otherwise(0.0))
#         .withColumn("dist_long",   F.when(F.col("trip_distance_segment") == "long",   1.0).otherwise(0.0))

#         # FEATURE 3 : Normalisation de l'interaction distance-temps (borne à 1.0)
#         .withColumn("dist_duration_norm", F.least(F.col("dist_duration_interaction") / 5000.0, F.lit(1.0)))

#         # Encode payment_type
#         .withColumn("payment_credit", F.when(F.col("payment_type") == 1, 1.0).otherwise(0.0))
#         .withColumn("payment_cash",   F.when(F.col("payment_type") == 2, 1.0).otherwise(0.0))
#         .withColumn("payment_other",  F.when(F.col("payment_type").isin([3, 4, 5]), 1.0).otherwise(0.0))

#         # Booléens → Numérique
#         .withColumn("airport_pickup_flag",  F.col("is_airport_pickup").cast("double"))
#         .withColumn("airport_dropoff_flag", F.col("is_airport_dropoff").cast("double"))

#         # Normalisation des features numériques (min-max simplifié)
#         .withColumn("trip_distance_norm",  F.least(F.col("trip_distance")            / 50.0,  F.lit(1.0)))
#         .withColumn("trip_duration_norm",  F.least(F.col("trip_duration_minutes")    / 120.0, F.lit(1.0)))
#         .withColumn("speed_norm",          F.least(F.col("speed_mph")                / 60.0,  F.lit(1.0)))
#     )

#     # -------------------------------------------------------------------------
#     # FEATURE ENGINEERING — TEST
#     # -------------------------------------------------------------------------
#     test_encoded = (
#         test_df
#         # Encode time_of_day
#         .withColumn("time_morning",   F.when(F.col("time_of_day") == "morning",   1.0).otherwise(0.0))
#         .withColumn("time_afternoon", F.when(F.col("time_of_day") == "afternoon", 1.0).otherwise(0.0))
#         .withColumn("time_evening",   F.when(F.col("time_of_day") == "evening",   1.0).otherwise(0.0))
#         .withColumn("time_night",     F.when(F.col("time_of_day") == "night",     1.0).otherwise(0.0))

#         # FEATURE 1 : is_rush_hour
#         .withColumn("rush_hour_flag", F.col("is_rush_hour").cast("double"))

#         # FEATURE 2 : One-Hot Encoding de trip_distance_segment
#         .withColumn("dist_short",  F.when(F.col("trip_distance_segment") == "short",  1.0).otherwise(0.0))
#         .withColumn("dist_medium", F.when(F.col("trip_distance_segment") == "medium", 1.0).otherwise(0.0))
#         .withColumn("dist_long",   F.when(F.col("trip_distance_segment") == "long",   1.0).otherwise(0.0))

#         # FEATURE 3 : Normalisation de l'interaction distance-temps
#         .withColumn("dist_duration_norm", F.least(F.col("dist_duration_interaction") / 5000.0, F.lit(1.0)))

#         # Encode payment_type
#         .withColumn("payment_credit", F.when(F.col("payment_type") == 1, 1.0).otherwise(0.0))
#         .withColumn("payment_cash",   F.when(F.col("payment_type") == 2, 1.0).otherwise(0.0))
#         .withColumn("payment_other",  F.when(F.col("payment_type").isin([3, 4, 5]), 1.0).otherwise(0.0))

#         # Booléens → Numérique
#         .withColumn("airport_pickup_flag",  F.col("is_airport_pickup").cast("double"))
#         .withColumn("airport_dropoff_flag", F.col("is_airport_dropoff").cast("double"))

#         # Normalisation des features numériques
#         .withColumn("trip_distance_norm",  F.least(F.col("trip_distance")         / 50.0,  F.lit(1.0)))
#         .withColumn("trip_duration_norm",  F.least(F.col("trip_duration_minutes") / 120.0, F.lit(1.0)))
#         .withColumn("speed_norm",          F.least(F.col("speed_mph")             / 60.0,  F.lit(1.0)))
#     )

#     # -------------------------------------------------------------------------
#     # PRÉDICTIONS (régression linéaire à coefficients fixes)
#     # -------------------------------------------------------------------------
#     predictions = (
#         test_encoded
#         .withColumn(
#             "predicted_total_amount",
#             F.lit(3.0)                                                               # Base fare
#             + (F.col("trip_distance")         * 2.5)                                # Distance
#             + (F.col("trip_duration_minutes") * 0.5)                                # Durée
#             + F.when(F.col("time_evening") == 1, 2.0).otherwise(0.0)               # Soir
#             + F.when(F.col("time_night")   == 1, 3.0).otherwise(0.0)               # Nuit
#             + F.when(F.col("airport_pickup_flag")  == 1, 5.0).otherwise(0.0)       # Aéroport départ
#             + F.when(F.col("airport_dropoff_flag") == 1, 5.0).otherwise(0.0)       # Aéroport arrivée
#             + (F.col("passenger_count") * 0.5)                                      # Passagers
#             # Bonus rush hour (actif uniquement en v2.0 — ligne à commenter pour v1.0)
#             + F.when(F.col("rush_hour_flag") == 1, 1.5).otherwise(0.0)             # Rush hour
#             # Bonus segment de distance
#             + F.when(F.col("dist_medium") == 1, 1.0).otherwise(0.0)               # Trajet moyen
#             + F.when(F.col("dist_long")   == 1, 2.5).otherwise(0.0)               # Long trajet
#         )
#     )

#     # -------------------------------------------------------------------------
#     # MÉTRIQUES D'ÉVALUATION
#     # -------------------------------------------------------------------------
#     metrics = predictions.agg(
#         F.sqrt(
#             F.avg(F.pow(F.col("predicted_total_amount") - F.col("target_total_amount"), 2))
#         ).alias("rmse"),
#         F.avg(
#             F.abs(F.col("predicted_total_amount") - F.col("target_total_amount"))
#         ).alias("mae"),
#         F.corr("predicted_total_amount", "target_total_amount").alias("correlation")
#     )

#     # Retourne les métriques avec métadonnées
#     return metrics.select(
#         F.lit("simple_linear_model").alias("model_type"),
#         F.col("rmse"),
#         F.col("mae"),
#         F.col("correlation"),
#         F.current_timestamp().alias("training_timestamp")
#     )

#========================== Avec commentaire des features==========================================================

import dlt
from pyspark.sql import functions as F

@dlt.table(
    comment="Model training results and evaluation metrics — Baseline v1.0"
)
def ml_model_training():
    """
    ML Model Training — Baseline v1.0
    Les 3 nouvelles features sont commentées dans le feature engineering
    et absentes de la formule de prédiction.
    """
    df = dlt.read("ml_training_data")

    train_df = df.filter("is_training = true")
    test_df  = df.filter("is_training = false")

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING — TRAIN
    # -------------------------------------------------------------------------
    train_encoded = (
        train_df
        # Encode time_of_day
        .withColumn("time_morning",   F.when(F.col("time_of_day") == "morning",   1.0).otherwise(0.0))
        .withColumn("time_afternoon", F.when(F.col("time_of_day") == "afternoon", 1.0).otherwise(0.0))
        .withColumn("time_evening",   F.when(F.col("time_of_day") == "evening",   1.0).otherwise(0.0))
        .withColumn("time_night",     F.when(F.col("time_of_day") == "night",     1.0).otherwise(0.0))

        # FEATURE 1 commentée — colonne absente en v1.0
        # .withColumn("rush_hour_flag", F.col("is_rush_hour").cast("double"))

        # FEATURE 2 commentée — colonne absente en v1.0
        # .withColumn("dist_short",  F.when(F.col("trip_distance_segment") == "short",  1.0).otherwise(0.0))
        # .withColumn("dist_medium", F.when(F.col("trip_distance_segment") == "medium", 1.0).otherwise(0.0))
        # .withColumn("dist_long",   F.when(F.col("trip_distance_segment") == "long",   1.0).otherwise(0.0))

        # FEATURE 3 commentée — colonne absente en v1.0
        # .withColumn("dist_duration_norm", F.least(F.col("dist_duration_interaction") / 5000.0, F.lit(1.0)))

        # Encode payment_type
        .withColumn("payment_credit", F.when(F.col("payment_type") == 1, 1.0).otherwise(0.0))
        .withColumn("payment_cash",   F.when(F.col("payment_type") == 2, 1.0).otherwise(0.0))
        .withColumn("payment_other",  F.when(F.col("payment_type").isin([3, 4, 5]), 1.0).otherwise(0.0))

        # Booléens → Numérique
        .withColumn("airport_pickup_flag",  F.col("is_airport_pickup").cast("double"))
        .withColumn("airport_dropoff_flag", F.col("is_airport_dropoff").cast("double"))

        # Normalisation des features numériques
        .withColumn("trip_distance_norm",  F.least(F.col("trip_distance")         / 50.0,  F.lit(1.0)))
        .withColumn("trip_duration_norm",  F.least(F.col("trip_duration_minutes") / 120.0, F.lit(1.0)))
        .withColumn("speed_norm",          F.least(F.col("speed_mph")             / 60.0,  F.lit(1.0)))
    )

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING — TEST
    # -------------------------------------------------------------------------
    test_encoded = (
        test_df
        # Encode time_of_day
        .withColumn("time_morning",   F.when(F.col("time_of_day") == "morning",   1.0).otherwise(0.0))
        .withColumn("time_afternoon", F.when(F.col("time_of_day") == "afternoon", 1.0).otherwise(0.0))
        .withColumn("time_evening",   F.when(F.col("time_of_day") == "evening",   1.0).otherwise(0.0))
        .withColumn("time_night",     F.when(F.col("time_of_day") == "night",     1.0).otherwise(0.0))

        # FEATURE 1 commentée — colonne absente en v1.0
        # .withColumn("rush_hour_flag", F.col("is_rush_hour").cast("double"))

        # FEATURE 2 commentée — colonne absente en v1.0
        # .withColumn("dist_short",  F.when(F.col("trip_distance_segment") == "short",  1.0).otherwise(0.0))
        # .withColumn("dist_medium", F.when(F.col("trip_distance_segment") == "medium", 1.0).otherwise(0.0))
        # .withColumn("dist_long",   F.when(F.col("trip_distance_segment") == "long",   1.0).otherwise(0.0))

        # FEATURE 3 commentée — colonne absente en v1.0
        # .withColumn("dist_duration_norm", F.least(F.col("dist_duration_interaction") / 5000.0, F.lit(1.0)))

        # Encode payment_type
        .withColumn("payment_credit", F.when(F.col("payment_type") == 1, 1.0).otherwise(0.0))
        .withColumn("payment_cash",   F.when(F.col("payment_type") == 2, 1.0).otherwise(0.0))
        .withColumn("payment_other",  F.when(F.col("payment_type").isin([3, 4, 5]), 1.0).otherwise(0.0))

        # Booléens → Numérique
        .withColumn("airport_pickup_flag",  F.col("is_airport_pickup").cast("double"))
        .withColumn("airport_dropoff_flag", F.col("is_airport_dropoff").cast("double"))

        # Normalisation des features numériques
        .withColumn("trip_distance_norm",  F.least(F.col("trip_distance")         / 50.0,  F.lit(1.0)))
        .withColumn("trip_duration_norm",  F.least(F.col("trip_duration_minutes") / 120.0, F.lit(1.0)))
        .withColumn("speed_norm",          F.least(F.col("speed_mph")             / 60.0,  F.lit(1.0)))
    )

    # -------------------------------------------------------------------------
    # PRÉDICTIONS — formule baseline sans les nouvelles features
    # -------------------------------------------------------------------------
    predictions = (
        test_encoded
        .withColumn(
            "predicted_total_amount",
            F.lit(3.0)                                                          # Base fare
            + (F.col("trip_distance")         * 2.5)                           # Distance
            + (F.col("trip_duration_minutes") * 0.5)                           # Durée
            + F.when(F.col("time_evening") == 1, 2.0).otherwise(0.0)          # Soir
            + F.when(F.col("time_night")   == 1, 3.0).otherwise(0.0)          # Nuit
            + F.when(F.col("airport_pickup_flag")  == 1, 5.0).otherwise(0.0)  # Aéroport départ
            + F.when(F.col("airport_dropoff_flag") == 1, 5.0).otherwise(0.0)  # Aéroport arrivée
            + (F.col("passenger_count") * 0.5)                                 # Passagers

            # Termes liés aux nouvelles features — COMMENTÉS pour v1.0
            # + F.when(F.col("rush_hour_flag") == 1, 1.5).otherwise(0.0)
            # + F.when(F.col("dist_medium")    == 1, 1.0).otherwise(0.0)
            # + F.when(F.col("dist_long")      == 1, 2.5).otherwise(0.0)
        )
    )

    # -------------------------------------------------------------------------
    # MÉTRIQUES D'ÉVALUATION
    # -------------------------------------------------------------------------
    metrics = predictions.agg(
        F.sqrt(
            F.avg(F.pow(F.col("predicted_total_amount") - F.col("target_total_amount"), 2))
        ).alias("rmse"),
        F.avg(
            F.abs(F.col("predicted_total_amount") - F.col("target_total_amount"))
        ).alias("mae"),
        F.corr("predicted_total_amount", "target_total_amount").alias("correlation")
    )

    return metrics.select(
        F.lit("simple_linear_model").alias("model_type"),
        F.col("rmse"),
        F.col("mae"),
        F.col("correlation"),
        F.current_timestamp().alias("training_timestamp")
    )