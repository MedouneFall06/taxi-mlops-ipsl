#=========================== SANS COMMENTAIRE DES FEATURES=================================================
# import dlt
# from pyspark.sql import functions as F

# @dlt.table(
#     comment="ML training dataset enriched with Part 2 features for fare prediction"
# )
# def ml_training_data():
#     """
#     Préparation des données ML :
#     Sélectionne les variables cibles et les caractéristiques (features),
#     incluant les 3 nouvelles variables de la Partie 2 / 3.

#     Pour générer la baseline (v1.0) :
#       → Commenter le bloc "NOUVELLES FEATURES" ci-dessous avant de lancer le pipeline.
#     Pour générer la version améliorée (v2.0) :
#       → Décommenter le bloc "NOUVELLES FEATURES" et relancer.
#     """
#     return (
#         dlt.read("silver_taxi_features")

#         # Filtres qualité
#         .filter("""
#             trip_distance > 0          AND trip_distance < 100       AND
#             trip_duration_minutes > 0  AND trip_duration_minutes < 180 AND
#             total_amount > 0           AND total_amount < 500         AND
#             fare_amount > 0            AND
#             passenger_count > 0        AND passenger_count <= 6
#         """)
#         .select(
#             # ── Variable cible ────────────────────────────────────────────────
#             F.col("total_amount").alias("target_total_amount"),

#             # ── Features du trajet ───────────────────────────────────────────
#             "trip_distance",
#             "trip_duration_minutes",
#             "speed_mph",
#             "passenger_count",

#             # ── Features temporelles ─────────────────────────────────────────
#             "pickup_hour",
#             "pickup_day_of_week",
#             "time_of_day",

#             # ── NOUVELLES FEATURES (PARTIE 2 & 3) ───────────────────────────
#             # Pour la baseline v1.0 : commenter les 3 lignes suivantes
#             "is_rush_hour",                # FEATURE 1 : heure de pointe
#             "trip_distance_segment",       # FEATURE 2 : segment de distance
#             "dist_duration_interaction",   # FEATURE 3 : interaction distance × durée
#             # ─────────────────────────────────────────────────────────────────

#             # ── Features de localisation ─────────────────────────────────────
#             "PULocationID",
#             "DOLocationID",
#             "is_airport_pickup",
#             "is_airport_dropoff",

#             # ── Paiement et tarifs ───────────────────────────────────────────
#             "payment_type",
#             "RatecodeID",

#             # ── Date pour partitionnement ────────────────────────────────────
#             "pickup_date"
#         )

#         # Séparation train / test déterministe (80 / 20)
#         .withColumn(
#             "is_training",
#             (F.hash("pickup_date", "PULocationID") % 100) < 80
#         )
#     )
#=========================== SANS COMMENTAIRE DES FEATURES=================================================
import dlt
from pyspark.sql import functions as F

@dlt.table(
    comment="ML training dataset — Baseline v1.0 (sans nouvelles features)"
)
def ml_training_data():
    """
    Préparation des données ML — Baseline v1.0
    Les 3 nouvelles features de la Partie 2 sont commentées.
    """
    return (
        dlt.read("silver_taxi_features")

        # Filtres qualité
        .filter("""
            trip_distance > 0          AND trip_distance < 100        AND
            trip_duration_minutes > 0  AND trip_duration_minutes < 180 AND
            total_amount > 0           AND total_amount < 500          AND
            fare_amount > 0            AND
            passenger_count > 0        AND passenger_count <= 6
        """)
        .select(
            # ── Variable cible ────────────────────────────────────────────────
            F.col("total_amount").alias("target_total_amount"),

            # ── Features du trajet ───────────────────────────────────────────
            "trip_distance",
            "trip_duration_minutes",
            "speed_mph",
            "passenger_count",

            # ── Features temporelles ─────────────────────────────────────────
            "pickup_hour",
            "pickup_day_of_week",
            "time_of_day",

            # ── NOUVELLES FEATURES (PARTIE 2 & 3) — COMMENTÉES POUR v1.0 ────
            # "is_rush_hour",              # FEATURE 1 : heure de pointe
            # "trip_distance_segment",     # FEATURE 2 : segment de distance
            # "dist_duration_interaction", # FEATURE 3 : interaction distance × durée
            # ─────────────────────────────────────────────────────────────────

            # ── Features de localisation ─────────────────────────────────────
            "PULocationID",
            "DOLocationID",
            "is_airport_pickup",
            "is_airport_dropoff",

            # ── Paiement et tarifs ───────────────────────────────────────────
            "payment_type",
            "RatecodeID",

            # ── Date pour partitionnement ────────────────────────────────────
            "pickup_date"
        )

        # Séparation train / test déterministe (80 / 20)
        .withColumn(
            "is_training",
            (F.hash("pickup_date", "PULocationID") % 100) < 80
        )
    )