#!/bin/env python
""" A program to solve some simple Sudoku puzzles."""

#---------------------------------------------------------------
# Sudoku Solver
# Author : Srivathsan Srinivasagopalan
# Email: srivathsan.srinivas@gmail.com
# Date: 21st Oct 2014
# Algorithm used are from the following study materials:
# A Pencil-and-PaperAlgorithm for SolvingSudoku Puzzles - J. F. Crook

# Comments: Has crazy high runtime complexity (and space complexity).
# This code works only certain, easy puzzles.

#----------------------------------------------------------------

import pandas as pd
from itertools import chain, combinations


def get_empty_cells_in_block(block_id, grid_df):
    """Given a block_id, return coords of all empty cells in it."""
    all_cells_dict = get_blocks_cells()
    blk_cells = all_cells_dict[block_id]
    empty_cells = []
    for cell in blk_cells:
        if grid_df.T[cell[0]][cell[1]] == 0:
            empty_cells.append(cell)

    return empty_cells


def get_col_values(col_num, input_df):
    """Returns a list of col values."""
    values = list(input_df.T.ix[col_num])
    return values


def get_row_values(row_num, input_df):
    """Returns a list of row values."""
    values = list(input_df.ix[row_num, ])
    return values

def powerset(iterable):
    """Returns a powerset of a given iterable."""
    iterlist = list(iterable)
    return chain.from_iterable(combinations(iterlist, r) for \
                               r in range(len(iterlist)+1))

def get_block_id(pair):
    """Given a pair of numbers, returns the candidate block numbers."""
    if 0 <= max(pair) <= 2:
        candidate_block_ids = range(1, 4)
    elif 3 <= max(pair) <= 5:
        candidate_block_ids = range(4, 7)
    else:
        candidate_block_ids = range(7, 10)
    return candidate_block_ids


def flatten_list(somelist):
    """Flattens a given list."""
    flattened = [val for sublist in somelist for val in sublist]
    return flattened


def get_all_nums_in_block(block_id, input_df):
    """Returns a list of numbers for a given block_id."""
    blocks_dict = get_blocks_cells()
    values = []
    cells = blocks_dict[block_id]
    for cell in cells:
        num = input_df.T[cell[0]][cell[1]]
        values.append(num)

    return values


def get_non_zero_nums_in_block(block_id, input_df):
    """Returns a list of non-zero numbers for a given block_id."""
    values = get_all_nums_in_block(block_id, input_df)
    val_set = set(values)
    if 0 in val_set:
        val_set.remove(0)
    return list(val_set)


def get_block_nums(num, block_list, input_df):
    """Returns a list of blocks (in block_list) where 'num' is present."""
    block_nums = []
    for block_num in block_list:
        numbers = set(get_non_zero_nums_in_block(block_num, input_df))
        if num in numbers:
            block_nums.append(block_num)
    return block_nums


def get_block_list_col(col_pair):
    """Given a col-pair, find the list of possible blocks that it could have."""
    block_list = []
    if 0 <= max(col_pair) < 3:
        block_list = [1, 4, 7]
    elif 3 <= max(col_pair) < 6:
        block_list = [2, 5, 8]
    else:
        block_list = [3, 6, 9]
    return block_list


def get_block_list_row(row_pair):
    """Given a row-pair, find the list of possible blocks that it could have."""
    block_list = []
    if 0 <= max(row_pair) < 3:
        block_list = range(1, 4)
    elif 3 <= max(row_pair) < 6:
        block_list = range(4, 7)
    else:
        block_list = range(7, 10)
    return block_list

def get_col_nums(target_block):
    """Returns a list of columns for a given block."""
    if target_block in [1, 4, 7]:
        col_nums = [0, 1, 2]
    elif target_block in [2, 5, 8]:
        col_nums = [3, 4, 5]
    else:
        col_nums = [6, 7, 8]
    return col_nums

def get_row_nums(target_block):
    """Returns a list of rows for a given block."""
    target_block = target_block.pop()
    if target_block in [0, 1, 2]:
        row_nums = [0, 1, 2]
    elif target_block in [3, 4, 5]:
        row_nums = [3, 4, 5]
    else:
        row_nums = [6, 7, 8]
    return row_nums


def get_candidate_cols(target_block):
    """Returns candidate columns for a given block."""
    col_nums = get_col_nums(target_block)
    return col_nums

def get_pairs(col_list):
    """Returns pairs of columns."""
    pairs = []
    for cols in powerset(col_list):
        if len(cols) == 2:
            pairs.append(cols)
    return pairs


def force_number(num, coord, grid_df):
    """Place a given number into a given cell-coord in the given grid."""
    grid_df.T[coord[0]][coord[1]] = num
    return grid_df


def get_coords(target_block, target_row):
    """Find the coordinates of the target block row."""
    cols = get_col_nums(target_block)
    coords = []
    for col in cols:
        coords.append([target_row, col])
    return coords


def get_row_pairs(input_df, row_array):
    """Returns a pair of rows and a row dict."""
    row_a = set()
    row_b = set()
    row_c = set()
    vals = get_row_values(row_array[0], input_df)
    row_a.update(vals)
    if 0 in row_a:
        row_a.remove(0)

    vals = get_row_values(row_array[1], input_df)
    row_b.update(vals)
    if 0 in row_b:
        row_b.remove(0)

    vals = get_row_values(row_array[2], input_df)
    row_c.update(vals)
    if 0 in row_c:
        row_c.remove(0)

    # {0: row_a values, 1:  row_b values, 2: row_c values}
    row_dict = {row_array[0]: row_a, row_array[1]: row_b, row_array[2]: row_c}
    row_pairs = []
    for result in powerset(row_dict.keys()):
        if len(result) == 2:
            row_pairs.append(result)

    return (row_pairs, row_dict)


def get_grid_row_pairs(input_df):
    """Returns a pair of rows from their values."""
    row_list = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    row_pairs = []
    row_dict = []
    row_pair_val = [] # list of row-pairs and their row-values

    for row_arr in row_list:
        row_pairs, row_dict = get_row_pairs(input_df, row_arr)
        row_pair_val.append((row_pairs, row_dict))

    return row_pair_val


def get_target_blocks(num, block_list, input_df):
    """Find all blocks (in block_list only) where num is present."""
    target_block_set = set()
    num_present_blks = get_block_nums(num, block_list, input_df)
    target_block_set = set(block_list) - set(num_present_blks)

    return target_block_set


def get_target_block_row_coords(target_block_set, target_row):
    """Gets the target cells in the target row, block."""
    while target_block_set:
        blk_id = target_block_set.pop()
        target_block_row_coords = get_coords(blk_id, target_row)
    return target_block_row_coords


def get_potential_cells(num, target_block_row_coords, input_df):
    """Returns potentials cells (dictionary)."""
    potential_cells = {}
    coord_list = []
    for coord in target_block_row_coords:
        elem = input_df.T[coord[0]][coord[1]]
        if elem == 0:
            coord_list.append(coord)
            potential_cells[num] = coord_list
    return potential_cells


def get_tgt_cols(num, potential_cells, input_df):
    """Returns a dict of potential columns for each number."""
    potential_target_cols = {}
    col_list = []
    for num, pcells in potential_cells.iteritems():
        for cell in pcells:
            col = cell[1]
            col_values = set(get_col_values(col, input_df))
            if num not in col_values:
                col_list.append(col)
            else:
                continue
        potential_target_cols[num] = col_list
    return potential_target_cols

def get_common_nums(row_pairs, row_dict):
    """Returns a list of common numbers from the given rows."""
    common_nums = {}
    for item in row_pairs:
        common_list = []
        set1 = row_dict[item[0]]
        set2 = row_dict[item[1]]
        common = set1.intersection(set2)
        common_list.append(common)
        common_nums[item] = common_list

    return common_nums


def get_master_dict(row_tuples, row_pair_val):
    """returns a dictionary of block_nums and a dict of \
    row-tuples and values."""
    master_dict = {}
    for i in range(0, 3):
        key = row_tuples[i]
        value = row_pair_val[i]
        master_dict[key] = value
    return master_dict

def get_common_nums_dict(value):
    """Returns a dictionary of common numbers among two rows."""
    row_pairs = value[0]
    row_dict = value[1]
    common_nums_dict = get_common_nums(row_pairs, row_dict)
    return common_nums_dict


def is_row_eligible(number, row, grid_df):
    """Returns a flag indicating if the row is eligible for \
    a number.0: Not eligible."""
    flag = 1
    r_values = get_row_values(row, grid_df)
    if number in r_values:
        flag = 0
    return flag


def is_col_eligible(number, col, grid_df):
    """Returns a flag indicating if the column is eligible for a \
    number. 0: Not eligible."""
    flag = 1
    c_values = get_col_values(col, grid_df)
    if number in c_values:
        flag = 0
    return flag


def get_cell_eligibilty(num, coord, grid_df):
    """Returns a flag indicating if the cell-coord is eligible for \
    a number. 0: Not eligible."""
    row = coord[0]
    col = coord[1]
    values_in_row = get_row_values(row, grid_df)
    if num in values_in_row:
        flag = 0
    else:
        values_in_col = get_col_values(col, grid_df)
        if num in values_in_col:
            flag = 0
        else:
            flag = 1

    return flag



def get_num_eligible_cells(empty_cells, number, grid_df):
    """Returns number of eligible cells for a given number."""
    # There must be ONLY one eligible cell in the block.
    num_eligible_cells = 0
    coord_list = []
    for cell in empty_cells:
        row = cell[0]
        col = cell[1]
        coord = [row, col]
        flag = get_cell_eligibilty(number, coord, grid_df)
        if flag == 1:
            coord_list.append(coord)
            num_eligible_cells += 1
        else:
            continue

    return (num_eligible_cells, coord_list)


def force2(grid_df):
    """For each block, see if any number [1..9] can fit in."""
    for block_id in range(1, 10):
        empty_cells = get_empty_cells_in_block(block_id, grid_df)
        nums_in_block = get_non_zero_nums_in_block(block_id, grid_df)
        for number in range(1, 10):
            if number in nums_in_block:
                continue
            else:
                num_eligible_cells, coord_list = \
                            get_num_eligible_cells(empty_cells, number, grid_df)
                # There must be ONLY one eligible cell in the block.
                if num_eligible_cells == 1:
                    # We found ONLY ONE eligible Cell. So fill it in.
                    force2_df = force_number(number, coord_list[0], grid_df)
                else:
                    continue

    return force2_df

def find_push_num(num, blk_list, t_row, input_df):
    """Find target col and push the num."""
    forced_df = input_df.copy()
    t_blk_set = get_target_blocks(num, blk_list, input_df)
    # Now, we know target-row and target-block-set
    # (potentially more than 1 block). Have to find target-col.
    if len(t_blk_set) != 0:
        t_blk_row_coords = get_target_block_row_coords(t_blk_set, t_row)
        potential_cells = get_potential_cells(num, t_blk_row_coords, input_df)
        potential_tgt_cols = get_tgt_cols(num, potential_cells, input_df)
        for num, col_list in potential_tgt_cols.iteritems():
            if len(col_list) == 1:
                for col in col_list:
                    coord = [t_row, col]
                    forced_df = force_number(num, coord, input_df)

    return forced_df


def populate_force2(input_df):
    """Populate a given grid with a computed number."""
    forced_df = input_df.copy()
    row_tuples = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    row_pair_val = get_grid_row_pairs(input_df)
    master_dict = get_master_dict(row_tuples, row_pair_val)

    for row_tup, value in master_dict.iteritems():
        common_nums_dict = get_common_nums_dict(value)
        for rpair in common_nums_dict.iterkeys():
            trow = set(row_tup) - set(rpair)
            t_row = trow.pop()
            common_numbers = common_nums_dict[rpair]
            common_numbers = flatten_list(common_numbers)
            blk_list = get_block_list_row(rpair)
            # for each num, find a suitable block and col it could fit in.
            for num in common_numbers:
                # check if target row has num
                trow_values = get_row_values(t_row, input_df)
                if num not in trow_values:
                    forced_df = find_push_num(num, blk_list, t_row, input_df)

    return forced_df


def get_box_tl_br(cell):
    """Get the top-left and bottom-right coords of the block."""
    block_side_len = 3
    row = cell[0]
    col = cell[1]
    top_left = [((row)/block_side_len)*block_side_len, \
                ((col)/block_side_len)*block_side_len]
    bottom_right = [(((row)/block_side_len)*block_side_len)+2, \
                    (((col)/block_side_len)*block_side_len)+2]
    box_tl_br = [top_left, bottom_right]
    return box_tl_br


def get_box_value_set(input_df, box_tl_br):
    """Get all unique values in a 3x3 box."""
    box_value_set = set()
    row1 = box_tl_br[0][0]
    row2 = box_tl_br[1][0]
    col1 = box_tl_br[0][1]
    col2 = box_tl_br[1][1]
    box_df = input_df.ix[row1:row2, col1:col2]

    for row in box_df.iterrows()[1]:
        box_value_set.update(row.values)

    return box_value_set


def get_candidate_num_list(cell, block_id, grid_df):
    """Get a list of markup numbers for a given cell in a given block."""
    candidate_num_set = set()
    full_set = set()
    full_set.update([1, 2, 3, 4, 5, 6, 7, 8, 9])
    row = cell[0]
    col = cell[1]
    row_set = set()
    col_set = set()
    row_vals = get_row_values(row, grid_df)
    col_vals = get_col_values(col, grid_df)
    row_set.update(row_vals)
    col_set.update(col_vals)
    row_union_col_set = row_set | col_set
    candidate_num_set = full_set - row_union_col_set
    # a - if a number is in block, remove from candidate set
    nums_already_present = get_non_zero_nums_in_block(block_id, grid_df)
    candidate_num_set = candidate_num_set - set(nums_already_present)
    if 0 in candidate_num_set:
        candidate_num_set.remove(0)

    return list(candidate_num_set)


def do_block_markup(block_id, grid_df):
    """# returns a dict of cells and associalted numbers as tuples."""
    empty_cells = get_empty_cells_in_block(block_id, grid_df)
    cell_markup_dict = {}
    for cell in empty_cells:
        candidate_num_list = get_candidate_num_list(cell, block_id, grid_df)
        cell_markup_dict[cell] = candidate_num_list

    return cell_markup_dict


def do_markups(grid_df):
    """Returns a dict of block-ids and dict of markups of cells."""

    #{1: {(0,0): (3,4,6), (0,2): (3,9)}, 2: {..}, ..9: {..}}
    blocks = list(range(1, 10))
    df_markup_dict = {}
    for block_id in blocks:
        df_markup_dict[block_id] = do_block_markup(block_id, grid_df)

    return df_markup_dict


def get_blocks_cells():
    """return a block num -> set of cells mapping."""
    block = {}
    cell_list = []
    for row in range(0, 3):
        for col in range(0, 3):
            cell = (row, col)
            cell_list.append(cell)
        block[1] = cell_list

    cell_list = []
    for row in range(0, 3):
        for col in range(3, 6):
            cell = (row, col)
            cell_list.append(cell)
        block[2] = cell_list

    cell_list = []
    for row in range(0, 3):
        for col in range(6, 9):
            cell = (row, col)
            cell_list.append(cell)
        block[3] = cell_list

    cell_list = []
    for row in range(3, 6):
        for col in range(0, 3):
            cell = (row, col)
            cell_list.append(cell)
        block[4] = cell_list

    cell_list = []
    for row in range(3, 6):
        for col in range(3, 6):
            cell = (row, col)
            cell_list.append(cell)
        block[5] = cell_list

    cell_list = []
    for row in range(3, 6):
        for col in range(6, 9):
            cell = (row, col)
            cell_list.append(cell)
        block[6] = cell_list


    cell_list = []
    for row in range(6, 9):
        for col in range(0, 3):
            cell = (row, col)
            cell_list.append(cell)
        block[7] = cell_list

    cell_list = []
    for row in range(6, 9):
        for col in range(3, 6):
            cell = (row, col)
            cell_list.append(cell)
        block[8] = cell_list

    cell_list = []
    for row in range(6, 9):
        for col in range(6, 9):
            cell = (row, col)
            cell_list.append(cell)
        block[9] = cell_list

    return block


def get_markup_list_for_block(block_id, df_markup_dict):
    """Returns markups for a block."""
    candidate_sets_dict = df_markup_dict[block_id]
    markup_list = []
    for markups in candidate_sets_dict.itervalues():
        markup_list.append(markups)
    return markup_list


def process_single_mkups(df_markup_dict, forced2_df):
    """If a markup is singleton for any cell, place that value into the grid."""
    for cell_mkup_combo in df_markup_dict.itervalues():
        for cell, mkup in cell_mkup_combo.iteritems():
            if len(mkup) == 1:
                num = mkup[0]
                force_number(num, cell, forced2_df)
    return forced2_df


def process_blocks(processed_df):
    """Find coords for any number that can be placed into grid."""
    df_markup_dict = do_markups(processed_df)
    for cell_mkup_combo in df_markup_dict.itervalues():
        for cell, markup in cell_mkup_combo.iteritems():
            if len(markup) == 1:
                num = markup[0]
                processed_df = force_number(num, cell, processed_df)
    return processed_df


def validate_block(block_id, grid_df):
    """Returns a flag indicating if the block is valid. 0: invalid"""
    flag = 1
    numbers = get_all_nums_in_block(block_id, grid_df)
    if 0 in numbers:
        flag = 0
    else:
        full_block_len = len(numbers)
        set_block_len = len(list(set(numbers)))
        if full_block_len != set_block_len:
            flag = 0

    return flag


def validate_row(row_num, grid_df):
    """Returns a flag indicating if a row is valid. 0: invalid"""
    flag = 1
    numbers = get_row_values(row_num, grid_df)
    if 0 in numbers:
        flag = 0
    else:
        full_row_len = len(numbers)
        set_row_len = len(list(set(numbers)))
        if full_row_len != set_row_len:
            flag = 0

    return flag


def validate_col(col_num, grid_df):
    """Returns a flag indicating if the column is valid. 0: invalid"""
    flag = 1
    numbers = get_col_values(col_num, grid_df)
    if 0 in numbers:
        flag = 0
    else:
        full_col_len = len(numbers)
        set_col_len = len(list(set(numbers)))
        if full_col_len != set_col_len:
            print "invalid col"
            flag = 0

    return flag


def validate(grid_df):
    """Returns a flag indicating if a grid is valid. 0: invalid"""
    flag_count = 0
    for block_id in range(1, 10):
        flag = validate_block(block_id, grid_df)
        if flag:
            flag_count += 1
    return flag_count


def validate_input_rc(input_df):
    """Returns a flag indicating if a given row and column \
    is valid. 0: invalid"""
    flag = 1
    all_nums = []
    for val in range(0, 9):
        row_nums = get_row_values(val, input_df)
        all_nums = all_nums + row_nums
        col_nums = get_col_values(val, input_df)
        all_nums = all_nums + col_nums
    for num in all_nums:
        if num not in range(0, 10):
            flag = 0

    return flag

def validate_input_grid(input_df):
    """Returns a flag indicating if the input puzzle is valid. 0: invalid"""
    flag = 1
    dfshape = input_df.shape
    if dfshape[0] != dfshape[1]:
        flag = 0
    else:
        flag = validate_input_rc(input_df)

    return flag


def main():
    """The Main driver function."""
    input_df = pd.read_csv('insight.csv', header=None)
    print "\n\nInput grid: \n"
    print input_df
    flag = validate_input_grid(input_df)
    if flag == 0:
        print "\nThe given Sudoku puzzle is invalid."
        exit()
    else:
        print "\nThe given Sudoku puzzle is valid."

    # Force values into cells
    forced_df = populate_force2(input_df)
    forced_df1 = populate_force2(forced_df)
    forced_df2 = populate_force2(forced_df1)
    forced2_df = force2(forced_df2)
    df_markup_dict = do_markups(forced2_df)

    # process each block's markups
    processed_df = process_single_mkups(df_markup_dict, forced2_df)

    final_df = process_blocks(processed_df)
    flag_count = validate(final_df)
    if flag_count == 9:
        print "\nSolved Sudoku: \n", final_df
        print "\nThe given Sudoku puzzle is solved and the solution is valid.\n"
        final_df.to_csv('solution.csv', sep=',', index=False, index_label=False)
        print "The solution is present in 'solution.csv' in pwd.\n"
    else:
        print "\nThe given Sudoku puzzle is not yet solved.\n"

if __name__ == "__main__":
    main()
