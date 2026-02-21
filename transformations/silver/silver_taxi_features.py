import dlt # Utilisation de la bibliothèque standard DLT pour la compatibilité [1, 2]
from pyspark.sql import functions as F

@dlt.table(
    comment="Nettoyage et enrichissement des données avec 3 nouvelles caractéristiques (Partie 2)"
)
# Ajout des nouvelles règles de qualité (Data Quality Expectations) [3]
@dlt.expect_all_or_drop({
    "valid_trip_distance": "trip_distance > 0 AND trip_distance < 100",
    "valid_fare": "fare_amount > 0 AND fare_amount < 500",
    "valid_passenger_count": "passenger_count > 0 AND passenger_count <= 6",
    "valid_timestamps": "tpep_pickup_datetime < tpep_dropoff_datetime",
    "valid_dist_duration": "dist_duration_interaction >= 0" # Règle pour la nouvelle feature
})
@dlt.expect_all({
    "reasonable_tip": "tip_amount >= 0 AND tip_amount < 100",
    "reasonable_total": "total_amount > 0 AND total_amount < 1000",
    "valid_distance_segment": "trip_distance_segment IN ('short', 'medium', 'long')" # Règle pour la nouvelle feature
})
def silver_taxi_features():
    """
    Couche Silver : Engineering des caractéristiques pour le modèle ML.
    - Nettoyage des données brutes
    - Calcul des métriques de base (vitesse, durée)
    - Ajout des 3 nouvelles features : is_rush_hour, trip_distance_segment, dist_duration_interaction [3]
    """
    return (
        spark.readStream.table("bronze_taxi_trips")
        .filter("VendorID IS NOT NULL")
        
        # 1. Calcul des métriques existantes
        .withColumn(
            "trip_duration_minutes",
            (F.unix_timestamp("tpep_dropoff_datetime") - 
             F.unix_timestamp("tpep_pickup_datetime")) / 60
        )
        .withColumn(
            "speed_mph",
            F.when(
                F.col("trip_duration_minutes") > 0,
                (F.col("trip_distance") / F.col("trip_duration_minutes")) * 60
            ).otherwise(0)
        )
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_day_of_week", F.dayofweek("tpep_pickup_datetime"))
        
        # 2. FEATURE N°1 : Indicateur d'Heure de Pointe (is_rush_hour) [3]
        # Justification : Le trafic intense augmente le temps de trajet et le coût.
        # Heures de pointe NY : 7h-9h et 16h-19h en semaine (Lundi=2 à Vendredi=6 en PySpark)
        .withColumn(
            "is_rush_hour",
            ((F.col("pickup_day_of_week").between(2, 6)) & 
             ((F.col("pickup_hour").between(7, 9)) | (F.col("pickup_hour").between(16, 19))))
        )
        
        # 3. FEATURE N°2 : Segment de Distance (trip_distance_segment) [3]
        # Justification : Les tarifs peuvent varier selon la catégorie de distance.
        .withColumn(
            "trip_distance_segment",
            F.when(F.col("trip_distance") < 2, "short")
            .when(F.col("trip_distance").between(2, 5), "medium")
            .otherwise("long")
        )
        
        # 4. FEATURE N°3 : Interaction Distance-Temps (dist_duration_interaction) [3, 4]
        # Justification : Aide le modèle à capturer les relations non linéaires (ex: trajet long et lent).
        .withColumn(
            "dist_duration_interaction",
            F.col("trip_distance") * F.col("trip_duration_minutes")
        )
        
        # 5. Extraction des autres caractéristiques temporelles et aéroports
        .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
        .withColumn(
            "time_of_day",
            F.when((F.col("pickup_hour") >= 6) & (F.col("pickup_hour") < 12), "morning")
            .when((F.col("pickup_hour") >= 12) & (F.col("pickup_hour") < 18), "afternoon")
            .when((F.col("pickup_hour") >= 18) & (F.col("pickup_hour") < 22), "evening")
            .otherwise("night")
        )
        .withColumn("is_airport_pickup", F.col("PULocationID").isin([5]))
        .withColumn("is_airport_dropoff", F.col("DOLocationID").isin([5]))
    )

