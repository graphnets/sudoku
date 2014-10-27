#!/bin/env python

#-------------------------------------
# Sudoku Solver
# Author : Srivathsan Srinivasagopalan [srivathsan.srinivas@gmail.com]
# Date: 21st Oct 2014
# Algorithm used are from the following study materials:
#  (i)  A Pencil-and-PaperAlgorithm for SolvingSudoku Puzzles - J. F. Crook
#  (ii) http://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1015&context=mathmidexppap

# Comments: This code is NOT clean. Has crazy high runtime complexity (and space complexity).
# This code is incomplete. Partially solves Sudoku.
#-------------------------------------

import csv
import pandas as pd
import numpy as np
from collections import OrderedDict
from itertools import chain, combinations


#---------------------------------------------
# input: input dataframe
# output: dict of row # and array of zero elem
#---------------------------------------------
def get_empty_cells(input_df):
    zero_dict = {}
    zero_list = []
    cell = []
    for index, row in input_df.iterrows():
        zero_columns = np.where(row.values == 0)[0]
        zero_dict[index] = zero_columns    
        for item in zero_columns:
            cell = [index, item]
            zero_list.append(cell)

    return zero_list


# Given a block_id, return coords of all empty cells in it
def get_empty_cells_in_block(block_id, df):
    all_cells_dict = get_blocks_cells();
    blk_cells = all_cells_dict[block_id];
    empty_cells = []
    for cell in blk_cells:
        if (df.T[cell[0]][cell[1]] == 0):
            empty_cells.append(cell)

    return empty_cells


# return a list of col values
def get_col_values(col_num, input_df):
    values = list(input_df.T.ix[col_num])
    return values

# returns a list of row values
def get_row_values(row_num, input_df):
    values = list(input_df.ix[row_num, ])
    return values

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_block_id(pair):
    if 0 <= max(pair) <= 2:
        candidate_block_ids = range(1,4)
    elif 3 <= max(pair) <= 5:
        candidate_block_ids = range(4,7)
    else:
        candidate_block_ids = range(7,10)
    return candidate_block_ids


def flatten_list(l):
    flattened = [val for sublist in l for val in sublist]
    return flattened


# returns a list of numbers for a given block_id
def get_non_zero_nums_in_block(block_id, input_df):
    dfT = input_df.T
    blocks_dict = get_blocks_cells(); 
    values = []
    cells = blocks_dict[block_id]
    for cell in cells:
        num = dfT[cell[0]][cell[1]]
        values.append(num)

    val_set = set(values); val_set.remove(0)
    return list(val_set)


# returns a list of blocks (in block_list) where 'num' is present
def get_block_nums(num, block_list, input_df):
    block_nums = []
    for b in block_list:
        numbers = set(get_non_zero_nums_in_block(b, input_df)); 
        if (num in numbers):
            block_nums.append(b)
    return block_nums


# given a col-pair, find the list of possible blocks that it could have
def get_block_list(col_pair):
    block_list = []
    if 0 <= max(col_pair) < 3:
        block_list = [1, 4, 7]
    elif 3 <= max(row_pair) < 6:
        block_list = [2, 5, 8]
    else:
        block_list = [3, 6, 9]
    return block_list

# given a row-pair, find the list of possible blocks that it could have
def get_block_list(row_pair):
    block_list = []
    if 0 <= max(row_pair) < 3:
        block_list = range(1,4)
    elif 3 <= max(row_pair) < 6:
        block_list = range(4,7)
    else:
        block_list = range(7,10)
    return block_list

def get_col_nums(target_block):
    if target_block in [1, 4, 7]:
        col_nums = [0,1,2]
    elif target_block in [2, 5, 8]:
        col_nums = [3,4,5]
    else:
        col_nums = [6,7,8]
    return col_nums

def get_row_nums(target_block):
    target_block = target_block.pop()
    if target_block in [0,1,2]:
        row_nums = [0,1,2]
    elif target_block in [3,4,5]:
        row_nums = [3,4,5]
    else:
        row_nums = [6,7,8]
    return row_nums


def get_candidate_cols(target_block, num):
    col_nums = get_col_nums(target_block)
    return col_nums

def get_pairs(l):
    pairs = []
    for cols in powerset(l):
        if len(cols) == 2:
            pairs.append(cols)
    return pairs


def force_number(num, coord, df):
    df.T[coord[0]][coord[1]] = num
    return df

# Find the coordinates of the target block row.
def get_coords(target_block, target_row):
    cols = get_col_nums(target_block)
    coords = []
    for c in cols:
        coords.append([target_row, c])
    return coords


def get_row_pairs(input_df, row_array):
    row_a = set(); row_b = set(); row_c = set()
    
    # for now, considering only rows 0, 1 and 2
    v = get_row_values(row_array[0], input_df)
    row_a.update(v); row_a.remove(0)

    v = get_row_values(row_array[1], input_df)
    row_b.update(v); row_b.remove(0)

    v = get_row_values(row_array[2], input_df)
    row_c.update(v); row_c.remove(0)

    # {0: row_a values, 1:  row_b values, 2: row_c values}
    row_dict = {row_array[0]: row_a, row_array[1]: row_b, row_array[2]: row_c}
    row_pairs = []
    for result in powerset(row_dict.keys()):
        if (len(result) == 2):
            row_pairs.append(result)

    return (row_pairs, row_dict)


def get_grid_row_pairs(input_df):
    row_list = [[0,1,2], [3,4,5], [6,7,8]]
    row_pairs = []; row_dict = []; 
    row_pair_val = [] # list of row-pairs and their row-values

    for row_arr in row_list:
        row_pairs, row_dict = get_row_pairs(input_df, row_arr)
        row_pair_val.append((row_pairs, row_dict))

    return row_pair_val


# find all blocks (in block_list only) where num is present.
def get_target_blocks(num, block_list, input_df):
    target_block_set = set()
    num_present_blks = get_block_nums(num, block_list, input_df)
    target_block_set = set(block_list) - set(num_present_blks)

    return target_block_set


# Gets the target cells in the target row, block    
def get_target_block_row_coords(num, target_block_set, target_row):
    while target_block_set:
        blk_id = target_block_set.pop()
        candidate_cols = get_candidate_cols(blk_id, num)
        col_pairs = get_pairs(candidate_cols)
        target_col = set()
        target_block_row_coords = get_coords(blk_id, target_row)
    return target_block_row_coords

# returns potentials cells (dictionary)
def get_potential_cells(num, target_block_row_coords, input_df):
    potential_cells = {}; coord_list = []
    for coord in target_block_row_coords:
        elem = input_df.T[coord[0]][coord[1]]
        if (elem == 0):
            coord_list.append(coord)
            potential_cells[num] = coord_list

    return potential_cells

# returns a dict of potential columns for each number
def get_potential_target_cols(num, potential_cells, input_df):
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
    common_nums = {}
    for item in row_pairs:
        common_list = []
        set1 = row_dict[item[0]];
        set2 = row_dict[item[1]];
        common = set1.intersection(set2)
        common_list.append(common)
        common_nums[item] = common_list

    return common_nums


def get_master_dict(row_tuples, row_pair_val):
    master_dict = {}
    for i in range(0, 3):
        k = row_tuples[i]
        v = row_pair_val[i]
        master_dict[k] = v

    return master_dict

def get_common_nums_dict(v):
    row_pairs = v[0]
    row_dict = v[1]
    common_nums_dict = get_common_nums(row_pairs, row_dict)

    return common_nums_dict


def other_2_stuff(k):
    if 0 <= k < 3:
        other_2_stuff = set([0,1,2]) - set(k)
    elif 3 <= k < 6:
        other_2_stuff = set([3,4,5]) - set(k)
    else:
        other_2_stuff = set([6,7,8]) - set(k)

    return other_2_stuff

# flag = 0: row is NOT eligible
def is_row_eligible(number, row, df):
    flag = 1
    r_values = get_row_values(row, df)
    if number in r_values:
        flag = 0
    
    return flag


# flag = 0: col is NOT eligible
def is_col_eligible(number, col, df):
    flag = 1
    c_values = get_col_values(row, df)
    if number in c_values:
        flag = 0
    
    return flag

# 0 = Not Eligbile
def get_cell_eligibilty(num, coord, block_id, df):
    r = coord[0]
    c = coord[1]
    values_in_row = get_row_values(r, df)
    if num in values_in_row:
        flag = 0
    else:
        values_in_col = get_col_values(c, df)
        if num in values_in_col:
            flag = 0
        else:
            flag = 1
            
    return flag


# returns number of eligible cells for a given number and block_id
def get_num_eligible_cells(empty_cells, number, block_id, df):
    # There must be ONLY one eligible cell in the block.
    num_eligible_cells = 0
    coord_list = []
    for cell in empty_cells:
        r = cell[0]; c = cell[1]
        coord = [r,c]
        # cell_eligibility: 0 = Not eligible. 1 = Eligible.
        flag = get_cell_eligibilty(number,coord, block_id, df)
        if flag == 1:
            coord_list.append(coord)
            num_eligible_cells += 1
        else:
            continue
    

    return (num_eligible_cells, coord_list)


# For each block, see if any number [1..9] can fit in.
# use Sudoku criteria
def force2(df):
    for block_id in range(1, 10):
        empty_cells = get_empty_cells_in_block(block_id, df)
        nums_in_block = get_non_zero_nums_in_block(block_id, df)
        for number in range(1, 10):
            if number in nums_in_block:
                continue
            else:
                num_eligible_cells, coord_list = get_num_eligible_cells(empty_cells, number, block_id, df)
                # There must be ONLY one eligible cell in the block.
                if num_eligible_cells == 1:
                    # We found ONLY ONE eligible Cell. So fill it in.
                    df = force_number(number, coord_list[0], df)
                else:
                    continue

    return df


def populate_force2(input_df):
    df = input_df.copy()
    row_tuples = [(0,1,2), (3,4,5), (6,7,8)]
    row_pair_val = get_grid_row_pairs(input_df)
    master_dict = get_master_dict(row_tuples, row_pair_val)

    for row_tup, v in master_dict.iteritems():
        common_nums_dict = get_common_nums_dict(v)

        for rpair, c_elem_set in common_nums_dict.iteritems():
            trow = set(row_tup) - set(rpair)
            target_row = trow.pop()
            common_numbers = common_nums_dict[rpair]
            common_numbers = flatten_list(common_numbers)
            block_list = get_block_list(rpair)
            
            # for each num in common nums, find a suitable block and col it could fit in.
            for num in common_numbers:                
                # check if target row has num
                trow_values = get_row_values(target_row, input_df)
                if num in trow_values:
                    continue
                else:
                    target_block_set = get_target_blocks(num, block_list, input_df)
                    # Now, we know target-row, target-block-set (potentially more than 1 block).
                    # Have to find target-col.
                    if len(target_block_set) == 0:
                        continue
                    else:
                        target_block_row_coords = get_target_block_row_coords(num, target_block_set, target_row)
                        potential_cells = get_potential_cells(num, target_block_row_coords, input_df)
                        potential_target_cols = get_potential_target_cols(num, potential_cells, input_df)
                        
                        for num, col_list in potential_target_cols.iteritems():
                            if (len(col_list) == 1):
                                for col in col_list:
                                    coord = [target_row, col]
                                    df = force_number(num, coord, input_df)
    
    return df


# Get the top-left and bottom-right coords of the block where the given cell resides
def get_box_tl_br(cell):
    block_side_len = 3
    r = cell[0]
    c = cell[1]
    top_left = [((r)/block_side_len)*block_side_len, ((c)/block_side_len)*block_side_len]
    bottom_right = [(((r)/block_side_len)*block_side_len)+2, (((c)/block_side_len)*block_side_len)+2]
    box_tl_br = [top_left, bottom_right]

    return box_tl_br
    
# Get all unique values in a 3x3 box
def get_box_value_set(input_df, box_tl_br):
    box_value_set = set()
    r1 = box_tl_br[0][0]; r2 = box_tl_br[1][0]
    c1 = box_tl_br[0][1]; c2 = box_tl_br[1][1]
    box_df = input_df.ix[r1:r2, c1:c2]

    for index, row in box_df.iterrows():
        box_value_set.update(row.values)

    return box_value_set

# -- This is stage 1 and 2 - creating markups.
# -- also force single values into the table.
def gen_markups(input_df, zero_list):
    #print zero_list
    dfT = input_df.T
    candidate_num_set = set()
    cell_markup_dict = {}
    full_set = set()
    full_set.update([1,2,3,4,5,6,7,8,9])
    for cell in zero_list:
        #print cell
        r = cell[0]; c = cell[1]
        row_set = set(); col_set = set()
        #print dfT[r]; print dfT[c]
        row_set.update(dfT[r]); #print row_set
        col_set.update(input_df[c]); #print col_set
        row_U_col_set = row_set | col_set; #print row_U_col_set
        candidate_num_set = full_set - row_U_col_set
        # a - if a number is in box, remove from candidate set
        box_tl_br = get_box_tl_br(cell)
        box_value_set = get_box_value_set(input_df, box_tl_br)
        #print box_value_set
        candidate_num_set = candidate_num_set - box_value_set
        
        #print candidate_num_set
        if len(candidate_num_set) == 1:
            input_df[r][c] = canidate_num_set.pop()
        else:
            cell_markup_dict[tuple(cell)] = candidate_num_set
        #print cell_markup_dict
        
    #print cell_markup_dict
    return (input_df, cell_markup_dict)

def populate_singleton_values(input_df, cell_markup_dict):
    a_df = input_df.copy(deep = True)
    #key_list = cell_markup_dict.keys()

    for k, v in cell_markup_dict:
        print k; print v; exit()
        if len(v) == 1:
            print k;
            a_df[k[0]][k[1]] = v.pop()
            cell_markup_dict.pop(k)

    return (a_df, cell_markup_dict)

def gen_markups1(input_df, zero_dict):
    dfT = input_df.T
    for key, value in zero_dict.iteritems():
        print key, value
        #check if they are indeed zeros in original input
        for item in value:
            print dfT[key][item]

# revisit this
def get_blk_id_density_sorted(df):
    blocks = list(range(1,10))
    return blocks







# get a list of markup numbers for a given cell in a given block
def get_candidate_num_list(cell, block_id, df):
    candidate_num_set = set()
    full_set = set()
    full_set.update([1,2,3,4,5,6,7,8,9])
    r = cell[0]; c = cell[1]
    row_set = set(); col_set = set()
    row_vals = get_row_values(r, df)
    col_vals = get_col_values(c, df)
    row_set.update(row_vals)
    col_set.update(col_vals)
    row_U_col_set = row_set | col_set
    candidate_num_set = full_set - row_U_col_set
    # a - if a number is in block, remove from candidate set
    nums_already_present = get_non_zero_nums_in_block(block_id, df)
    candidate_num_set = candidate_num_set - set(nums_already_present)
    if 0 in candidate_num_set:
        candidate_num_set.remove(0)

    return list(candidate_num_set)


# returns a dict of cells and associalted numbers as tuples. {(x,y): (2,8,4), ...}
def do_block_markup(block_id, df):
    empty_cells = get_empty_cells_in_block(block_id, df)
    cell_markup_dict = {}
    for cell in empty_cells:
        candidate_num_list = get_candidate_num_list(cell, block_id, df)
        cell_markup_dict[cell] = candidate_num_list

    return cell_markup_dict



# returns a dict of block-ids and dict of markups of cells  
# {1: {(0,0): (3,4,6), (0,2): (3,9)}, 2: {..}, ..9: {..}}
def do_markups(df):
    sorted_blk_id_list = get_blk_id_density_sorted(df)
    df_markup_dict = {}
    for block_id in sorted_blk_id_list:
        df_markup_dict[block_id] = do_block_markup(block_id, df)

    return df_markup_dict



# block num -> set of cells mapping.
# for now, only for a 3x3 table
def get_blocks_cells():
    block = {}
    num = 0  # block number [starts from 1 to 9]
    
    cell_list = []
    for r in range(0, 3):
        for c in range(0, 3):
            cell = (r, c);
            cell_list.append(cell) 
        block[1] = cell_list

    cell_list = []
    for r in range(0, 3):
        for c in range(3, 6):
            cell = (r, c)
            cell_list.append(cell)
        block[2] = cell_list

    cell_list = []
    for r in range(0, 3):
        for c in range(6, 9):
            cell = (r, c)
            cell_list.append(cell)
        block[3] = cell_list


    cell_list = []
    for r in range(3, 6):
        for c in range(0, 3):
            cell = (r, c);
            cell_list.append(cell) 
        block[4] = cell_list

    cell_list = []
    for r in range(3, 6):
        for c in range(3, 6):
            cell = (r, c)
            cell_list.append(cell)
        block[5] = cell_list

    cell_list = []
    for r in range(3, 6):
        for c in range(6, 9):
            cell = (r, c)
            cell_list.append(cell)
        block[6] = cell_list


    cell_list = []
    for r in range(6, 9):
        for c in range(0, 3):
            cell = (r, c);
            cell_list.append(cell) 
        block[7] = cell_list

    cell_list = []
    for r in range(6, 9):
        for c in range(3, 6):
            cell = (r, c)
            cell_list.append(cell)
        block[8] = cell_list

    cell_list = []
    for r in range(6, 9):
        for c in range(6, 9):
            cell = (r, c)
            cell_list.append(cell)
        block[9] = cell_list

    return block


def get_preemptive_sets(input_df, cell_markup_dict):
    # consider each block (total of 9 blocks - 1 - 9)
    for block_id in range(1,10):
        print block_id
        pset[block_id] = get_block_psets(block_id, input_df, cell_markup_dict)

# returns markups for a block
# eg: block-1 = [[8, 9, 7], [8, 9, 3, 5, 6], [8, 5, 6], [8, 9, 5, 7], [8, 9, 3, 5, 6], [8, 9, 3, 6]]
def get_markup_list_for_block(block_id, df_markup_dict):
    candidate_sets_dict = df_markup_dict[block_id]
    markup_list = []
    for cell, markups in candidate_sets_dict.iteritems():
        markup_list.append(markups)
    
    print "\nMarkups for block %s: %s" %(block_id, markup_list)
    return markup_list



# returns all markups that are of size 2 and their corresponding cells
# Dict {cell: markup, cell: markup, ...}
def find_sizeNsets(n, block_id, df_markup_dict):
    cell_mkup_dict = df_markup_dict[block_id]
    markup_list = []
    for cell, markups in cell_mkup_dict.iteritems():
        if len(markups) == n:
            markup_list.append(markups)

    return markup_list

# cross out numbers in a block given a preempive set
def cross_numbers(pset, bdict):
    for cell, mkups in bdict.iteritems():  # only one entry in dict
        common_nums = 



# get subset markups of len N or smaller.
def get_subsets(markups, bdict):
    pset = {}
    lm = len(markups)
    for cell, mkup in bdict.iteritems():
        if (set(mkup) == set(markups)) or (set(mkup) <= set(markups)):
            pset[cell] = mkup
        if (lm == len(mkup)) and (len(set(mkup) & set(markups)) > 0):
            pset[cell] = mkup

        # now, check if numbers can be crossed-out
        crossed_dict = cross_numbers(pset, bdict)
    
    print "\n\nPreemptive Set:", pset; 
    return pset
            

    
# returns preemptive_set_dict {(2,3,5,9): [(6,0), (6,0), (7,2), (8,2)], [(1,2,6,7,8): [(3,6), (3,8)]}
# {tuple: list of cells}
def find_preemptive_sets(block_id, df_markup_dict, df):
    markup_list = get_markup_list_for_block(block_id, df_markup_dict)
    # find 3 cells of size 3 that has same numbers or some subset.
    # similarly, find 4 cells of size 4 that has same numbers or some subset.

    block_cell_markups_dict = df_markup_dict[block_id]
    print "block cell markups-dict:", block_cell_markups_dict; 

    for cell, markups in block_cell_markups_dict.iteritems():
        print "Markup:", markups
        subsets = get_subsets(markups, block_cell_markups_dict)
        #print "Subsets:", subsets


    exit()

    #doubles = find_sizeNsets(2, block_id, df_markup_dict); print "Doubles:", doubles
    #triples = find_sizeNsets(3, block_id, df_markup_dict); print "Triples:", triples
    #quads = find_sizeNsets(4, block_id, df_markup_dict); print "Quads:", quads
    #pents = find_sizeNsets(5, block_id, df_markup_dict); print "Pents:", pents
    
    #[89356] = [89356, 8957, 8936, 897, 856]

    #if size2sets:
    #    process_size2sets(block_id, df)

    #if triples:
    #    process_triples(block_id,  triples, df_markup_dict, df)


    exit()
    

def find_preemptive_sets_grid(df_markup_dict, df):
    preemptive_sets_grid = {}
    for block_id in range(1, 10):
        preemptive_sets_grid[block_id] = find_preemptive_sets(block_id, df_markup_dict, df)

    return preemptive_sets_grid



def validate_block(block_id, df):
    numbers = get_non_zero_nums_in_block(block_id, df)
    a = len(numbers)
    b = len(list(set(numbers)))
    flag = 1  # valid
    if (a != b):
        print "invalid block"
        flag = 0
    
    return flag


def validate_row(row_num, df):
    numbers = get_row_values(row_num, df)
    a = len(numbers)
    b = len(list(set(numbers)))
    flag = 1  # valid
    if (a != b):
        print "invalid row"
        flag = 0
    
    return flag


def validate_col(col_num):
    numbers = get_col_values(col_num, df)
    a = len(numbers)
    b = len(list(set(numbers)))
    flag = 1  # valid
    if (a != b):
        print "invalid col"
        flag = 0
    
    return flag

# flag = 0: Invalid
def validate(df):
    for block_id in range(1, 10):
        flag = validate_block_id(block_id, df)
        if flag:
            print "\n df is complete and valid."
        else:
            print "\n Grid is invalid."

    return flag

def main():
    block_dict = get_blocks_cells(); #print block_dict;
    input_df = pd.read_csv('unl.csv', header = None)
    #input_df = pd.read_csv('input.csv', header = None)
    print "\n\nInput grid: \n"
    print input_df

    #nums = get_non_zero_nums_in_block(2, input_df); print nums; exit()


    zero_list = get_empty_cells(input_df)

    # Force values into cells
    forced_df = populate_force2(input_df); 
    forced_df1 = populate_force2(forced_df); 
    #grid_diff = grid_diff(forced_df, forced_df1)
    #print "grid_diff:", grid_diff

    forced_df2 = populate_force2(forced_df1)
    print "\nForced DF:\n", forced_df2; 

    df = force2(forced_df2)
    print "\nLatest DF:\n", df
    
    # Check if successive runs of populate force returns newer df.
    # If not, do a different populate_force.
    #new_df = force2(forced_df2)
    #print "\n New DF: \n", new_df


    # step-1&2
    df_markup_dict = do_markups(df)
    print "\n\nInitial Markups are: \n"
    print df_markup_dict


    # For each block, find pre-emptive sets
    preemptive_set_dict_grid = find_preemptive_sets_grid(df_markup_dict, df)
    print "\n\nPreemptive sets:\n", preemptive_set_dict_grid

    # From the markups, if xy, xy and xyz are candidates in three cells of a block,
    # then the cell marked xyz must hold z.
    

if __name__ == "__main__":
    main()
