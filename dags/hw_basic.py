"""Ручной запуск батч-инференса и публикация дневного снимка прогнозов."""

from datetime import timedelta

import pendulum
import requests
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import BaseHook, Param, dag, get_current_context, task

from batch_common import (
    business_date,
    digest,
    save,
    read,
    validate_rows,
    infer_batches,
    validate_predictions,
    replace_day,
)


@dag(
    dag_id="hw_basic_batch_inference",
    start_date=pendulum.datetime(2026, 9, 1, tz="Europe/Moscow"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["homework", "basic"],
    params={
        "business_date": Param(
            "2026-09-01", type="string", format="date"
        ),
        "batch_size": Param(
            32, type="integer", minimum=1, maximum=1000
        ),
    },
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=5),
        "execution_timeout": timedelta(minutes=2),
    },
)
def batch_inference():

    @task
    def extract():
        context = get_current_context()

        day = business_date(
            context["params"]["business_date"],
            context.get("data_interval_start"),
        )

        connection = BaseHook.get_connection("source_api")
        host = connection.host.rstrip("/")

        # Загружаем данные клиентов
        response = requests.get(
            f"{host}/basic/{day}.json",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if data["business_date"] != day:
            raise ValueError("Дата полученных данных не совпадает с запрошенной")

        # Загружаем модель
        model_response = requests.get(
            f"{host}/models/v1.json",
            timeout=10,
        )
        model_response.raise_for_status()

        return save({
            "business_date": day,
            "rows": data["rows"],
            "model": model_response.json(),
            "model_sha256": digest(model_response.content),
        })

    @task(retries=0)
    def infer(extracted_ref):
        context = get_current_context()
        batch_size = context["params"]["batch_size"]

        data = read(extracted_ref)

        rows = validate_rows(
            data["rows"],
            data["business_date"],
        )

        predictions = infer_batches(
            rows,
            data["model"],
            batch_size,
        )

        expected_ids = [row["customer_id"] for row in rows]
        validate_predictions(predictions, expected_ids)

        return save({
            "business_date": data["business_date"],
            "model_version": "v1",
            "predictions": predictions,
            "model_sha256": data["model_sha256"],
        })

    @task
    def publish(inferred_ref):
        data = read(inferred_ref)

        conn = PostgresHook(
            postgres_conn_id="warehouse"
        ).get_conn()

        try:
            replace_day(
                conn,
                data["business_date"],
                data["model_version"],
                data["predictions"],
            )
        finally:
            conn.close()

        return {
            "business_date": data["business_date"],
            "model_version": data["model_version"],
            "rows": len(data["predictions"]),
        }

    extracted_ref = extract()
    inferred_ref = infer(extracted_ref)
    publish(inferred_ref)


batch_inference()
