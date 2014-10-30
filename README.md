sudoku
======
This Python program attempts to solve any Sudoku puzzle of size 9x9. The approach followed is adopted from "A Pencil-and-Paper
Algorithm for Solving Sudoku Puzzles" - J. F. Crook. The paper can be downloaded from:
http://www.ams.org/notices/200904/tx090400460p.pdf

to run the program, run it as below:
$python sudoku_solver.py

Input puzzle is:
0,3,5,2,9,0,8,6,4
0,8,2,4,1,0,7,0,3
7,6,4,3,8,0,0,9,0
2,1,8,7,3,9,0,4,0
0,0,0,8,0,4,2,3,0
0,4,3,0,5,2,9,7,0
4,0,6,5,7,1,0,0,9
3,5,9,0,2,8,4,1,7
8,0,0,9,0,0,5,2,6

The output is (as printed out by this program): The first row and col are indices.

   0  1  2  3  4  5  6  7  8
0  1  3  5  2  9  7  8  6  4
1  9  8  2  4  1  6  7  5  3
2  7  6  4  3  8  5  1  9  2
3  2  1  8  7  3  9  6  4  5
4  5  9  7  8  6  4  2  3  1
5  6  4  3  1  5  2  9  7  8
6  4  2  6  5  7  1  3  8  9
7  3  5  9  6  2  8  4  1  7
8  8  7  1  9  4  3  5  2  6

Some TODO stuff:

TODO: Proper variable naming (PEP-8)
TODO: Create classes and objects for cells and blocks.
TODO: Implement unittest features.
TODO: process all markups for each block, row and column. [do cross-outs]
TODO: find an efficient way to compare 2 grids.
TODO: find an efficient way to validate a grid if it satisfies Sudoku criteria:
      The solution of a Sudoku puzzle requires that every row, column, and box contain all
      the numbers in the set [1, 2, . . . , 9] and that every cell be occupied by one
      and only one number.