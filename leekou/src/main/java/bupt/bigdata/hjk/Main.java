package bupt.bigdata.hjk;

public class Main {

        public static void main(String[] args) {
            solution test = new solution();
            char[][] sudokuBoard = [[".",".","4",".",".",".","6","3","."],[".",".",".",".",".",".",".",".","."],["5",".",".",".",".",".",".","9","."],[".",".",".","5","6",".",".",".","."],["4",".","3",".",".",".",".",".","1"],[".",".",".","7",".",".",".",".","."],[".",".",".","5",".",".",".",".","."],[".",".",".",".",".",".",".",".","."],[".",".",".",".",".",".",".",".","."]];
            boolean isValid = test.isValidSudoku(sudokuBoard);
            System.out.println("Is valid Sudoku? " + isValid);
        }





}

