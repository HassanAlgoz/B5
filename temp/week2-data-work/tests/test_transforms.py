import polars as pl
import pytest
from polars.exceptions import SchemaError
from bootcamp_data.transforms import enforce_schema

def test_enforce_schema_basic():
    # Test case 1: Basic schema enforcement
    data = {"col_a": [1, 2, 3], "col_b": ["a", "b", "c"]}
    df = pl.DataFrame(data)
    schema = {"col_a": pl.Int64, "col_b": pl.String}
    transformed_df = enforce_schema(df, schema)

    assert transformed_df.schema == schema
    assert transformed_df.rows() == df.rows()

def test_enforce_schema_missing_column():
    # Test case 2: Missing column in DataFrame, should be added with nulls
    data = {"col_a": [1, 2, 3]}
    df = pl.DataFrame(data)
    schema = {"col_a": pl.Int64, "col_b": pl.String}
    transformed_df = enforce_schema(df, schema)

    expected_data = {"col_a": [1, 2, 3], "col_b": [None, None, None]}
    expected_df = pl.DataFrame(expected_data, schema=schema)

    assert transformed_df.schema == schema
    assert transformed_df.equals(expected_df)

def test_enforce_schema_extra_column():
    # Test case 3: Extra column in DataFrame, should be dropped
    data = {"col_a": [1, 2, 3], "col_b": ["a", "b", "c"], "col_c": [True, False, True]}
    df = pl.DataFrame(data)
    schema = {"col_a": pl.Int64, "col_b": pl.String}
    transformed_df = enforce_schema(df, schema)

    expected_data = {"col_a": [1, 2, 3], "col_b": ["a", "b", "c"]}
    expected_df = pl.DataFrame(expected_data, schema=schema)

    assert transformed_df.schema == schema
    assert transformed_df.equals(expected_df)

def test_enforce_schema_type_mismatch_castable():
    # Test case 4: Type mismatch but castable
    data = {"col_a": ["1", "2", "3"], "col_b": ["a", "b", "c"]}
    df = pl.DataFrame(data)
    schema = {"col_a": pl.Int64, "col_b": pl.String}
    transformed_df = enforce_schema(df, schema)

    expected_data = {"col_a": [1, 2, 3], "col_b": ["a", "b", "c"]}
    expected_df = pl.DataFrame(expected_data, schema=schema)

    assert transformed_df.schema == schema
    assert transformed_df.equals(expected_df)

def test_enforce_schema_type_mismatch_uncastable():
    # Test case 5: Type mismatch and uncastable, should raise an error
    data = {"col_a": ["1", "b", "3"], "col_b": ["a", "b", "c"]}
    df = pl.DataFrame(data)
    schema = {"col_a": pl.Int64, "col_b": pl.String}
    
    with pytest.raises(SchemaError):
        enforce_schema(df, schema)

def test_enforce_schema_empty_dataframe():
    # Test case 6: Empty DataFrame
    df = pl.DataFrame({"col_a": pl.Series(dtype=pl.Int64)})
    schema = {"col_a": pl.Int64, "col_b": pl.String}
    transformed_df = enforce_schema(df, schema)
    
    expected_df = pl.DataFrame({"col_a": pl.Series(dtype=pl.Int64), "col_b": pl.Series(dtype=pl.String)})
    
    assert transformed_df.schema == schema
    assert transformed_df.equals(expected_df)

def test_enforce_schema_order_of_columns():
    # Test case 7: Ensure columns are in the order defined by the schema
    data = {"col_b": ["a", "b", "c"], "col_a": [1, 2, 3]}
    df = pl.DataFrame(data)
    schema = {"col_a": pl.Int64, "col_b": pl.String}
    transformed_df = enforce_schema(df, schema)

    expected_data = {"col_a": [1, 2, 3], "col_b": ["a", "b", "c"]}
    expected_df = pl.DataFrame(expected_data, schema=schema)

    assert transformed_df.schema == schema
    assert transformed_df.equals(expected_df)

