import polars as pl
from polars.exceptions import SchemaError, InvalidOperationError

def enforce_schema(df: pl.DataFrame, schema: dict) -> pl.DataFrame:
    """
    Enforces a given schema on a Polars DataFrame.

    Args:
        df: The input Polars DataFrame.
        schema: A dictionary where keys are column names and values are Polars data types.

    Returns:
        A new Polars DataFrame with the enforced schema.

    Raises:
        SchemaError: If a column in the DataFrame does not match the expected schema type.
    """
    for col, expected_type in schema.items():
        if col not in df.columns:
            # Add missing columns with null values and the expected type
            df = df.with_columns(pl.lit(None, dtype=expected_type).alias(col))
        elif df[col].dtype != expected_type:
            # Cast column to the expected type, raising an error if incompatible
            try:
                df = df.with_columns(df[col].cast(expected_type).alias(col))
            except (SchemaError, InvalidOperationError) as e:
                raise SchemaError(
                    f"Column '{col}' type mismatch. Expected {expected_type}, got {df[col].dtype}. "
                    f"Casting failed: {e}"
                )
    
    # Select columns in the order of the schema, ensuring all schema columns are present
    # and dropping any extra columns not in the schema.
    return df.select([col for col in schema.keys() if col in df.columns])
