import dlt # Utilisation de la bibliothèque standard DLT

@dlt.table( # Décorateur standard
    comment="Raw taxi trip data ingested from cloud storage using Auto Loader"
)
def bronze_taxi_trips():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.inferColumnTypes", "true")
        .load("/Volumes/taxi_mlops_prod/taxi_analytics/yellowdata")
    )