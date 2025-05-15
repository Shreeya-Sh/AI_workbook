from approvedimports import *

def exhaustive_search_4tumblers(puzzle: CombinationProblem) -> list:
    """simple brute-force search method that tries every combination until
    it finds the answer to a 4-digit combination lock puzzle.
    """

    # check that the lock has the expected number of digits
    assert puzzle.numdecisions == 4, "this code only works for 4 digits"

    # create an empty candidate solution
    my_attempt = CandidateSolution()
    
    # ====> insert your code below here

    # loops over all possible values for each digit
    for digit1 in puzzle.value_set:
        for digit2 in puzzle.value_set:
            for digit3 in puzzle.value_set:
                for digit4 in puzzle.value_set:
                    my_attempt.variable_values = [digit1, digit2, digit3, digit4] # assign the values to attempt
                    try:
                        result = puzzle.evaluate(my_attempt.variable_values) # check if the combination is correct

                        # if the combination is correct return it
                        if result == 1:
                            return my_attempt.variable_values 
                    except:
                        pass  # if the combination is incorrect, skip it and continue with the next one

    # <==== insert your code above here
    
    # should never get here
    return [-1, -1, -1, -1]

def get_names(namearray: np.ndarray) -> list:
    family_names = []
    # ====> insert your code below here

    # loop through each name in the array
    for i in range(namearray.shape[0]):
        surname_chars = namearray[i, -6:]  # get the last 6 characters in the name array i.e. surname
        family_names.append("".join(surname_chars)) # join the surname characters into a string and append to list
    
    # <==== insert your code above here
    return family_names

def check_sudoku_array(attempt: np.ndarray) -> int:
    tests_passed = 0
    slices = []  # this will be a list of numpy arrays
    
    # ====> insert your code below here

    # use assertions to check that the array has 2 dimensions each of size 9
    assert attempt.shape == (9, 9), "Array must be 9x9"
    
    ## Remember all the examples of indexing above
    ## and use the append() method to add something to a list

    # loops through each row in the array
    for i in range(9):
        row = attempt[i,:]  # get the current row
        slices.append(row)  # add the row to the list of slices

    # loops through each column in the array
    for j in range(9):
        col = attempt[:,j]  # get the current column
        slices.append(col)  # add the column to the list of slices
    
    # loops through each 3x3 sub-grid in the array
    for i in range(0,9,3):
        for j in range(0,9,3):
            box = attempt[i:i+3, j:j+3]  # get the current 3x3 sub-grid
            slices.append(box.flatten())  # add the flattened sub-grid to the list of slices

    # loops through each slice in the list of slices
    for slice in slices:  
        if len(np.unique(slice)) == 9:  # check if the slice has 9 unique values
            tests_passed += 1  # if there are no duplicates, increment the test counter
        
    # <==== insert your code above here

    # return count of tests passed
    return tests_passed
