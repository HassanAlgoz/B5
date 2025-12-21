## Convert from `.qmd` slides to interactive `.ipynb` notebooks with blanks, hidden solutions, and test assertions.

Task: Transform the previously identified concepts into individual, interactive Jupyter Notebooks (.ipynb) without omitting any of the concepts or details (or than formatting changes).

Whenever we have tasks/lab acitivites / exercises, the required has to be converted to be a code blocks. Work Cells: Provide boilerplate code where students must implement logic. Use strictly: ### CODE START HERE ### and ### CODE END HERE ###.

After each exercise step that a student fills in. You add a call to a test function from the `test_utils.py` module. This test function will assert that the student's code is correct. If the student's code is incorrect, the test function will raise an AssertionError. The AssertionError will provide feedback to the student, including the input, expected output, actual output, and the actual error (the raised exception type and its error message) in red color.

Following both the exercise and the test function call, add a solution cell. Solutions must have comments to explain the steps (the non-obvious ones). Solutions cells must be collapsed by default.
