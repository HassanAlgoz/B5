"""Test functions for W1 notebooks.

This module contains all test functions used in the interactive notebooks.
Tests provide detailed feedback without raising exceptions.
"""

# ANSI color codes for red text
RED = '\033[91m'
RESET = '\033[0m'


def test_slugify():
    """Test for: slugify function."""
    errors = []
    
    # Test 1: Basic functionality
    test_input = "My Report 01"
    expected_output = "my-report-01"
    try:
        actual_output = slugify(test_input)
        if actual_output != expected_output:
            errors.append({
                'test': 'Basic functionality',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        errors.append({
            'test': 'Basic functionality',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 2: With extra whitespace
    test_input = "  Hello World  "
    expected_output = "hello-world"
    try:
        actual_output = slugify(test_input)
        if actual_output != expected_output:
            errors.append({
                'test': 'With extra whitespace',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        errors.append({
            'test': 'With extra whitespace',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 3: Single word
    test_input = "Python"
    expected_output = "python"
    try:
        actual_output = slugify(test_input)
        if actual_output != expected_output:
            errors.append({
                'test': 'Single word',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        errors.append({
            'test': 'Single word',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 4: Return type
    test_input = "Test"
    expected_output = "str"
    try:
        actual_output = slugify(test_input)
        if not isinstance(actual_output, str):
            errors.append({
                'test': 'Return type',
                'input': test_input,
                'expected': f"Type: {expected_output}",
                'actual': f"Type: {type(actual_output).__name__}",
                'error': None
            })
    except Exception as e:
        errors.append({
            'test': 'Return type',
            'input': test_input,
            'expected': f"Type: {expected_output}",
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    if errors:
        print("❌ Some tests failed. Here's what went wrong:\n")
        for i, err in enumerate(errors, 1):
            print(f"Test {i}: {err['test']}")
            print(f"  Input: {err['input']!r}")
            print(f"  Expected output: {err['expected']!r}")
            print(f"  Actual output: {err['actual']!r}")
            if err['error']:
                print(f"  {RED}Error: {err['error'][0]}: {err['error'][1]}{RESET}")
            print()
        print(f"\n{len(errors)} test(s) failed. Please review the errors above and fix your code.")
    else:
        print("✅ All tests passed! Great job!")


def test_person_class(Person):
    """Test for: Person class."""
    errors = []
    
    # Test 1: __repr__ works
    test_input = 'Person("Sara Ahmed", 23)'
    expected_output = 'repr string containing "Person", "Sara Ahmed", and "23"'
    try:
        p = Person("Sara Ahmed", 23)
        actual_output = repr(p)
        if "Person" not in actual_output or "Sara Ahmed" not in actual_output or "23" not in actual_output:
            errors.append({
                'test': '__repr__ method',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        errors.append({
            'test': '__repr__ method',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 2: first_name property
    test_input = 'Person("Sara Ahmed", 23).first_name'
    expected_output = "Sara"
    try:
        p = Person("Sara Ahmed", 23)
        actual_output = p.first_name
        if actual_output != expected_output:
            errors.append({
                'test': 'first_name property',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except AttributeError:
        errors.append({
            'test': 'first_name property',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': ('AttributeError', 'first_name property not implemented')
        })
    except Exception as e:
        errors.append({
            'test': 'first_name property',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 3: last_name property
    test_input = 'Person("Sara Ahmed", 23).last_name'
    expected_output = "Ahmed"
    try:
        p = Person("Sara Ahmed", 23)
        actual_output = p.last_name
        if actual_output != expected_output:
            errors.append({
                'test': 'last_name property',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except AttributeError:
        errors.append({
            'test': 'last_name property',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': ('AttributeError', 'last_name property not implemented')
        })
    except Exception as e:
        errors.append({
            'test': 'last_name property',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 4: Age validation (too high)
    test_input = 'Person("Test", 23); p.age = 300'
    expected_output = 'ValueError: age must be between 0 and 200'
    try:
        p = Person("Test", 23)
        p.age = 300
        errors.append({
            'test': 'Age validation (too high)',
            'input': test_input,
            'expected': expected_output,
            'actual': 'No error raised (validation failed)',
            'error': None
        })
    except ValueError as e:
        pass  # Expected
    except Exception as e:
        errors.append({
            'test': 'Age validation (too high)',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 5: Age validation (negative)
    test_input = 'Person("Test", 23); p.age = -5'
    expected_output = 'ValueError: age must be between 0 and 200'
    try:
        p = Person("Test", 23)
        p.age = -5
        errors.append({
            'test': 'Age validation (negative)',
            'input': test_input,
            'expected': expected_output,
            'actual': 'No error raised (validation failed)',
            'error': None
        })
    except ValueError as e:
        pass  # Expected
    except Exception as e:
        errors.append({
            'test': 'Age validation (negative)',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 6: Valid age works
    test_input = 'Person("Test", 50).age'
    expected_output = 50
    try:
        p = Person("Test", 50)
        actual_output = p.age
        if actual_output != expected_output:
            errors.append({
                'test': 'Valid age storage',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        errors.append({
            'test': 'Valid age storage',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    if errors:
        print("❌ Some tests failed. Here's what went wrong:\n")
        for i, err in enumerate(errors, 1):
            print(f"Test {i}: {err['test']}")
            print(f"  Input: {err['input']}")
            print(f"  Expected output: {err['expected']}")
            print(f"  Actual output: {err['actual']}")
            if err['error']:
                print(f"  {RED}Error: {err['error'][0]}: {err['error'][1]}{RESET}")
            print()
        print(f"\n{len(errors)} test(s) failed. Please review the errors above and fix your code.")
    else:
        print("✅ All tests passed! Great job!")


def test_column_profile_class(ColumnProfile):
    """Test for: ColumnProfile class."""
    errors = []
    
    # Test 1: Basic initialization
    test_input = 'ColumnProfile("age", "number", 100, 5, 95)'
    expected_output = 'name="age", total=100'
    try:
        col = ColumnProfile("age", "number", 100, 5, 95)
        actual_output = f'name="{col.name}", total={col.total}'
        if col.name != "age" or col.total != 100:
            errors.append({
                'test': 'Basic initialization',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        errors.append({
            'test': 'Basic initialization',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 2: missing_pct calculation
    test_input = 'ColumnProfile("test", "text", 100, 10, 90).missing_pct'
    expected_output = 10.0
    try:
        col = ColumnProfile("test", "text", 100, 10, 90)
        actual_output = col.missing_pct
        if abs(actual_output - expected_output) > 0.01:
            errors.append({
                'test': 'missing_pct calculation',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except AttributeError:
        errors.append({
            'test': 'missing_pct calculation',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': ('AttributeError', 'missing_pct property not implemented')
        })
    except Exception as e:
        errors.append({
            'test': 'missing_pct calculation',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 3: missing_pct with zero total (edge case)
    test_input = 'ColumnProfile("test", "text", 0, 0, 0).missing_pct'
    expected_output = 0.0
    try:
        col = ColumnProfile("test", "text", 0, 0, 0)
        actual_output = col.missing_pct
        if actual_output != expected_output:
            errors.append({
                'test': 'missing_pct with zero total',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        errors.append({
            'test': 'missing_pct with zero total',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 4: to_dict method
    test_input = 'ColumnProfile("age", "number", 100, 5, 95).to_dict()'
    expected_output = 'dict with "name" and "missing_pct" keys, missing_pct=5.0'
    try:
        col = ColumnProfile("age", "number", 100, 5, 95)
        actual_output = col.to_dict()
        if not isinstance(actual_output, dict):
            errors.append({
                'test': 'to_dict method',
                'input': test_input,
                'expected': 'dict type',
                'actual': f'{type(actual_output).__name__} type',
                'error': None
            })
        elif "name" not in actual_output or "missing_pct" not in actual_output:
            errors.append({
                'test': 'to_dict method',
                'input': test_input,
                'expected': expected_output,
                'actual': f'Missing keys. Got: {list(actual_output.keys())}',
                'error': None
            })
        elif actual_output["missing_pct"] != 5.0:
            errors.append({
                'test': 'to_dict method',
                'input': test_input,
                'expected': 'missing_pct=5.0',
                'actual': f'missing_pct={actual_output["missing_pct"]}',
                'error': None
            })
    except AttributeError:
        errors.append({
            'test': 'to_dict method',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': ('AttributeError', 'to_dict method not implemented')
        })
    except Exception as e:
        errors.append({
            'test': 'to_dict method',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 5: __repr__ method
    test_input = 'repr(ColumnProfile("age", "number", 100, 5, 95))'
    expected_output = 'repr string containing "ColumnProfile" and "age"'
    try:
        col = ColumnProfile("age", "number", 100, 5, 95)
        actual_output = repr(col)
        if "ColumnProfile" not in actual_output or "age" not in actual_output:
            errors.append({
                'test': '__repr__ method',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        errors.append({
            'test': '__repr__ method',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    if errors:
        print("❌ Some tests failed. Here's what went wrong:\n")
        for i, err in enumerate(errors, 1):
            print(f"Test {i}: {err['test']}")
            print(f"  Input: {err['input']}")
            print(f"  Expected output: {err['expected']}")
            print(f"  Actual output: {err['actual']}")
            if err['error']:
                print(f"  {RED}Error: {err['error'][0]}: {err['error'][1]}{RESET}")
            print()
        print(f"\n{len(errors)} test(s) failed. Please review the errors above and fix your code.")
    else:
        print("✅ All tests passed! Great job!")


def test_typer_cli():
    """Test for: Typer CLI profile command."""
    errors = []
    
    # Test 1: app is a Typer instance
    test_input = 'app variable'
    expected_output = 'typer.Typer instance with "command" attribute'
    try:
        actual_output = 'typer.Typer instance' if hasattr(app, "command") else 'not a typer.Typer instance'
        if not hasattr(app, "command"):
            errors.append({
                'test': 'app is a Typer instance',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except NameError as e:
        errors.append({
            'test': 'app is a Typer instance',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': ('NameError', 'app variable not defined')
        })
    except Exception as e:
        errors.append({
            'test': 'app is a Typer instance',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 2: profile function exists
    test_input = 'profile function'
    expected_output = 'callable function'
    try:
        actual_output = 'callable function' if callable(profile) else 'not callable'
        if not callable(profile):
            errors.append({
                'test': 'profile function exists',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except NameError as e:
        errors.append({
            'test': 'profile function exists',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': ('NameError', 'profile function not defined')
        })
    except Exception as e:
        errors.append({
            'test': 'profile function exists',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test 3: Check function signature (basic)
    test_input = 'inspect.signature(profile)'
    expected_output = 'function with parameters: input_path, out_dir, report_name'
    import inspect
    try:
        sig = inspect.signature(profile)
        params = list(sig.parameters.keys())
        actual_output = f'function with parameters: {params}'
        missing_params = []
        if "input_path" not in params:
            missing_params.append("input_path")
        if "out_dir" not in params:
            missing_params.append("out_dir")
        if "report_name" not in params:
            missing_params.append("report_name")
        
        if missing_params:
            errors.append({
                'test': 'Function signature',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': (None, f'Missing parameters: {", ".join(missing_params)}')
            })
    except Exception as e:
        errors.append({
            'test': 'Function signature',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    if errors:
        print("❌ Some tests failed. Here's what went wrong:\n")
        for i, err in enumerate(errors, 1):
            print(f"Test {i}: {err['test']}")
            print(f"  Input: {err['input']}")
            print(f"  Expected output: {err['expected']}")
            print(f"  Actual output: {err['actual']}")
            if err['error']:
                error_type, error_msg = err['error']
                if error_type:
                    print(f"  {RED}Error: {error_type}: {error_msg}{RESET}")
                else:
                    print(f"  {RED}Error: {error_msg}{RESET}")
            print()
        print(f"\n{len(errors)} test(s) failed. Please review the errors above and fix your code.")
    else:
        print("✅ All tests passed! Great job!")


def test_read_csv_rows():
    """Test for: read_csv_rows function."""
    from pathlib import Path
    import csv
    
    # Create a test CSV
    test_csv = Path("test_data.csv")
    test_csv.write_text("name,age\nSara,23\nAli,30", encoding="utf-8")
    
    errors = []
    
    # Import and test
    test_input = 'read_csv_rows(test_csv) where test_csv contains "name,age\\nSara,23\\nAli,30"'
    expected_output = 'list with 2 row dictionaries'
    try:
        from csv_profiler.io import read_csv_rows
        actual_output = read_csv_rows(test_csv)
        if not isinstance(actual_output, list) or len(actual_output) != 2:
            errors.append({
                'test': 'read_csv_rows basic functionality',
                'input': test_input,
                'expected': expected_output,
                'actual': f'list with {len(actual_output)} items' if isinstance(actual_output, list) else f'{type(actual_output).__name__}',
                'error': None
            })
        else:
            print(f"✅ Successfully read {len(actual_output)} rows")
            print(f"First row: {actual_output[0]}")
    except ImportError as e:
        errors.append({
            'test': 'read_csv_rows import',
            'input': test_input,
            'expected': 'successful import',
            'actual': None,
            'error': ('ImportError', str(e))
        })
        print(f"⚠️  Could not import: {e}")
        print("Make sure you're running from the project root with PYTHONPATH=src")
    except Exception as e:
        errors.append({
            'test': 'read_csv_rows basic functionality',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Test error handling
    test_input = 'read_csv_rows(Path("nonexistent.csv"))'
    expected_output = 'FileNotFoundError raised'
    try:
        from csv_profiler.io import read_csv_rows
        read_csv_rows(Path("nonexistent.csv"))
        errors.append({
            'test': 'read_csv_rows error handling',
            'input': test_input,
            'expected': expected_output,
            'actual': 'No error raised',
            'error': None
        })
    except FileNotFoundError:
        print("✅ FileNotFoundError raised correctly")
    except Exception as e:
        errors.append({
            'test': 'read_csv_rows error handling',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    # Cleanup
    test_csv.unlink()
    
    if errors:
        print(f"\n{RED}❌ Some tests failed. Here's what went wrong:{RESET}\n")
        for i, err in enumerate(errors, 1):
            print(f"Test {i}: {err['test']}")
            print(f"  Input: {err['input']}")
            print(f"  Expected output: {err['expected']}")
            print(f"  Actual output: {err['actual']}")
            if err['error']:
                print(f"  {RED}Error: {err['error'][0]}: {err['error'][1]}{RESET}")
            print()
        print(f"\n{len(errors)} test(s) failed. Please review the errors above and fix your code.")


def test_profiling_functions():
    """Test for: profiling functions."""
    errors = []
    
    # Test is_missing
    test_input = 'is_missing("")'
    expected_output = True
    try:
        from csv_profiler.profiling import is_missing
        actual_output = is_missing("")
        if actual_output != expected_output:
            errors.append({
                'test': 'is_missing with empty string',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except ImportError:
        errors.append({
            'test': 'is_missing import',
            'input': test_input,
            'expected': 'successful import',
            'actual': None,
            'error': ('ImportError', 'Could not import is_missing. Check your module path.')
        })
    except Exception as e:
        errors.append({
            'test': 'is_missing with empty string',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    test_input = 'is_missing("NA")'
    expected_output = True
    try:
        from csv_profiler.profiling import is_missing
        actual_output = is_missing("NA")
        if actual_output != expected_output:
            errors.append({
                'test': 'is_missing with "NA" (case-insensitive)',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        if not any(err.get('test') == 'is_missing import' for err in errors):
            errors.append({
                'test': 'is_missing with "NA"',
                'input': test_input,
                'expected': expected_output,
                'actual': None,
                'error': (type(e).__name__, str(e))
            })
    
    test_input = 'is_missing("valid")'
    expected_output = False
    try:
        from csv_profiler.profiling import is_missing
        actual_output = is_missing("valid")
        if actual_output != expected_output:
            errors.append({
                'test': 'is_missing with valid text',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        if not any(err.get('test') == 'is_missing import' for err in errors):
            errors.append({
                'test': 'is_missing with valid text',
                'input': test_input,
                'expected': expected_output,
                'actual': None,
                'error': (type(e).__name__, str(e))
            })
    
    # Test try_float
    test_input = 'try_float("123")'
    expected_output = 123.0
    try:
        from csv_profiler.profiling import try_float
        actual_output = try_float("123")
        if actual_output != expected_output:
            errors.append({
                'test': 'try_float with valid number',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except ImportError:
        errors.append({
            'test': 'try_float import',
            'input': test_input,
            'expected': 'successful import',
            'actual': None,
            'error': ('ImportError', 'Could not import try_float')
        })
    except Exception as e:
        errors.append({
            'test': 'try_float with valid number',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    test_input = 'try_float("abc")'
    expected_output = None
    try:
        from csv_profiler.profiling import try_float
        actual_output = try_float("abc")
        if actual_output != expected_output:
            errors.append({
                'test': 'try_float with non-numeric',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        if not any(err.get('test') == 'try_float import' for err in errors):
            errors.append({
                'test': 'try_float with non-numeric',
                'input': test_input,
                'expected': expected_output,
                'actual': None,
                'error': (type(e).__name__, str(e))
            })
    
    # Test infer_type
    test_input = 'infer_type(["1", "2", "3"])'
    expected_output = "number"
    try:
        from csv_profiler.profiling import infer_type
        actual_output = infer_type(["1", "2", "3"])
        if actual_output != expected_output:
            errors.append({
                'test': 'infer_type with numeric values',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except ImportError:
        errors.append({
            'test': 'infer_type import',
            'input': test_input,
            'expected': 'successful import',
            'actual': None,
            'error': ('ImportError', 'Could not import infer_type')
        })
    except Exception as e:
        errors.append({
            'test': 'infer_type with numeric values',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    test_input = 'infer_type(["a", "b", "c"])'
    expected_output = "text"
    try:
        from csv_profiler.profiling import infer_type
        actual_output = infer_type(["a", "b", "c"])
        if actual_output != expected_output:
            errors.append({
                'test': 'infer_type with non-numeric values',
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'error': None
            })
    except Exception as e:
        if not any(err.get('test') == 'infer_type import' for err in errors):
            errors.append({
                'test': 'infer_type with non-numeric values',
                'input': test_input,
                'expected': expected_output,
                'actual': None,
                'error': (type(e).__name__, str(e))
            })
    
    # Test profile_rows (basic structure)
    test_input = 'profile_rows([{"name": "Sara", "age": "23"}, {"name": "Ali", "age": "30"}])'
    expected_output = 'dict with keys: n_rows, n_cols, columns; n_rows=2'
    try:
        from csv_profiler.profiling import profile_rows
        test_rows = [{"name": "Sara", "age": "23"}, {"name": "Ali", "age": "30"}]
        actual_output = profile_rows(test_rows)
        missing_keys = []
        if "n_rows" not in actual_output:
            missing_keys.append("n_rows")
        if "n_cols" not in actual_output:
            missing_keys.append("n_cols")
        if "columns" not in actual_output:
            missing_keys.append("columns")
        if missing_keys:
            errors.append({
                'test': 'profile_rows structure',
                'input': test_input,
                'expected': expected_output,
                'actual': f'dict missing keys: {missing_keys}',
                'error': None
            })
        elif actual_output["n_rows"] != 2:
            errors.append({
                'test': 'profile_rows n_rows',
                'input': test_input,
                'expected': 'n_rows=2',
                'actual': f'n_rows={actual_output["n_rows"]}',
                'error': None
            })
    except ImportError:
        errors.append({
            'test': 'profile_rows import',
            'input': test_input,
            'expected': 'successful import',
            'actual': None,
            'error': ('ImportError', 'Could not import profile_rows')
        })
    except Exception as e:
        errors.append({
            'test': 'profile_rows structure',
            'input': test_input,
            'expected': expected_output,
            'actual': None,
            'error': (type(e).__name__, str(e))
        })
    
    if errors:
        print("❌ Some tests failed. Here's what went wrong:\n")
        for i, err in enumerate(errors, 1):
            print(f"Test {i}: {err['test']}")
            print(f"  Input: {err['input']}")
            print(f"  Expected output: {err['expected']}")
            print(f"  Actual output: {err['actual']}")
            if err['error']:
                print(f"  {RED}Error: {err['error'][0]}: {err['error'][1]}{RESET}")
            print()
        print(f"\n{len(errors)} test(s) failed. Please review the errors above and fix your code.")
    else:
        print("✅ All tests passed! Great job!")

