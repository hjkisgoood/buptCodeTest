package bupt.bigdata.hjk;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;

class solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int totalLength = nums1.length + nums2.length;
        if (totalLength % 2 == 0) {
            return (findKth(nums1, nums2, totalLength / 2) + findKth(nums1, nums2, totalLength / 2 + 1)) / 2.0;
        } else {
            return findKth(nums1, nums2, totalLength / 2 + 1);
        }
    }

    private double findKth(int[] nums1, int[] nums2, int k) {
        int index1 = 0, index2 = 0;
        while (true) {
            // 处理边界情况
            if (index1 == nums1.length) {
                return nums2[index2 + k - 1];
            }
            if (index2 == nums2.length) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return Math.min(nums1[index1], nums2[index2]);
            }

            // 正常情况下，每次排除掉一半的元素
            int newIndex1 = Math.min(index1 + k / 2 - 1, nums1.length - 1);
            int newIndex2 = Math.min(index2 + k / 2 - 1, nums2.length - 1);
            if (nums1[newIndex1] <= nums2[newIndex2]) {
                k -= newIndex1 - index1 + 1;
                index1 = newIndex1 + 1;
            } else {
                k -= newIndex2 - index2 + 1;
                index2 = newIndex2 + 1;
            }
        }
    }


    public boolean isMatch(String s, String p) {//


        return isMatchHelper(s, p, 0, 0);


    }

    private boolean isMatchHelper(String s, String p, int sIndex, int pIndex) {

        if (p.length() == pIndex) {
            return sIndex == s.length();
        }
        if (pIndex + 1 < p.length() && p.charAt(pIndex + 1) == '*') {
            if (sIndex < s.length() && (p.charAt(pIndex) == '.' || p.charAt(pIndex) == s.charAt(sIndex))) {
                return isMatchHelper(s, p, sIndex + 1, pIndex) || isMatchHelper(s, p, sIndex, pIndex + 2);
            } else {
                return isMatchHelper(s, p, sIndex, pIndex + 2);
            }
        } else {
            if (sIndex < s.length() && (p.charAt(pIndex) == '.' || p.charAt(pIndex) == s.charAt(sIndex))) {
                return isMatchHelper(s, p, sIndex + 1, pIndex + 1);
            } else {
                return false;
            }


        }


    }

    public int maxArea(int[] height) {
        int l = 0;
        int r = height.length-1;
        int res = 0;
        while (l != r){
            int sum = 0;
            if (height[l] < height[r]){
                sum = height[l]*(r-l);
                l++;
            }else {
                sum = height[r]*(r-l);
                r--;
            }
            if (sum > res){
                res = sum;
            }
        }
        return res;
    }


    public String intToRoman(int num) {
        StringBuilder roman = new StringBuilder();

        // 数字对应的罗马数字字符
        String[] romanDigits = {"I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"};
        // 对应的数字值
        int[] values = {1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000};

        // 从最大的值开始判断
        for (int i = values.length - 1; i >= 0; i--) {
            // 每次取得最大值的罗马数字字符
            String romanDigit = romanDigits[i];
            // 检查当前数字是否大于等于该最大值
            while (num >= values[i]) {
                // 如果是，则将对应的罗马数字字符添加到结果中，并更新数字值
                roman.append(romanDigit);
                num -= values[i];
            }
        }

        return roman.toString();
    }



    public int romanToInt(String s) {
        int romashu = 0;
        // 数字对应的罗马数字字符
        String[] romanDigits = {"I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"};
        // 对应的数字值
        int[] values = {1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000};
        int vLen = values.length;
        int slen = s.length();
        for(int i =0,j=vLen;i < slen && j > 0;){
            if(i+1<slen&&j-2>0&&s.substring(i,i+2).equals(romanDigits[j - 2])){
                romashu+=values[j-2];
                i+=2;
                j-=2;
            } else if (s.substring(i,i+1).equals(romanDigits[j - 1])) {
                romashu+=values[j-1];
                i+=1;
            }else{
                j--;
            }
        }
        return romashu;


    }

    public String longestCommonPrefix(String[] strs) {
        // 如果输入为空或者字符串数组长度为0，则返回空字符串
        if (strs == null || strs.length == 0) {
            return "";
        }

        // 将第一个字符串设为基准
        String prefix = strs[0];

        // 遍历字符串数组中的每个字符串
        for (int i = 1; i < strs.length; i++) {
            // 当前字符串与基准字符串的最长公共前缀长度
            int j = 0;
            // 比较当前字符串与基准字符串的字符，直到遇到不同的字符或者其中一个字符串遍历结束
            while (j < prefix.length() && j < strs[i].length() && prefix.charAt(j) == strs[i].charAt(j)) {
                j++;
            }
            // 更新基准字符串为最长公共前缀
            prefix = prefix.substring(0, j);

            // 如果最长公共前缀为空，则直接返回空字符串
            if (prefix.isEmpty()) {
                return "";
            }
        }

        return prefix;
    }


    public List<List<Integer>> threeSum(int[] nums) {//15. 三数之和

        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);//排序
        for(int i=0 ; i < nums.length-2;i++){
            if(i > 0 && nums[i] == nums[i-1]){
                continue;
            }
            for(int left = i+1,right = nums.length-1;left<right;){
                int sum = nums[i] + nums[left] + nums[right];
                if(sum == 0){
                    result.add(Arrays.asList(nums[i],nums[left],nums[right]));
                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;
                }else if(sum < 0){
                    left++;
                }else {
                    right--;
                }
            }

        }
        return result;
    }

    public int threeSumClosest(int[] nums, int target) {
        int result = nums[0] + nums[1] + nums[2];
        Arrays.sort(nums);
        for(int i=0 ; i < nums.length-2;i++){
            for(int left = i+1,right = nums.length-1;left < right;){
                int sum = nums[i] + nums[left] + nums[right] - target;
                if(sum == 0){
                    return target;
                }else if(sum < 0){
                    if(Math.abs(sum) < Math.abs(result)){
                        result = sum;
                    }
                    left++;
                }else if(sum > 0){
                    if(sum < result){
                        result = sum;
                    }
                    right--;
                }
            }
        }
        return result+target;
    }


    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        String[] keys = {"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
        int digitLen = digits.length();
        if(digitLen == 0) return result;
        letterCombinationsHelp(keys, digits, 0, new StringBuilder(), result);
        return result;


    }
    private void letterCombinationsHelp(String[] keys,String digits, int index, StringBuilder current, List<String> result ){
        if(index == digits.length()){
            result.add(current.toString());
            return ;
        }
        String key = keys[digits.charAt(index) - '2'];
        for(int i=0;i<key.length();i++ ){
            current.append(key.charAt(i));
            letterCombinationsHelp(keys, digits, index + 1, current, result);
            current.deleteCharAt(current.length() - 1);
        }
    }




    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;

        if(n< 4) return result;
        if ((target < 0 && nums[0] > target / 4) || (target > 0 && nums[n - 1] < target/4)) {//
            return result;
        }

        for (int i = 0; i < n - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue; // 跳过重复元素
            for (int j = i + 1; j < n - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) continue; // 跳过重复元素
                int left = j + 1;
                int right = n - 1;

                while (left < right) {
                    int sum = nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        result.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) left++; // 跳过重复元素
                        while (left < right && nums[right] == nums[right - 1]) right--; // 跳过重复元素
                        left++;
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }

        return result;

    }



     //Definition for singly-linked list.
      public static class ListNode {
         int val;
         ListNode next;
        ListNode() {}
       ListNode(int val) { this.val = val; }
          ListNode(int val, ListNode next) { this.val = val; this.next = next; }
      }

    public ListNode removeNthFromEnd(ListNode head, int n) {//删除链表的倒数第n节点
        ListNode headHead = new ListNode(0);
        headHead.next = head;
        ListNode left = headHead;
        ListNode right = headHead;
        int count =0;
        for(count = 0;right.next != null; count++){//如果n比链表长直接返回null
            right = right.next;
        }
        int index = 0;
        while(index < count - n){
            left = left.next;
            index++;
        }
        left.next = left.next.next;
        return headHead.next;
    }

    public boolean isValid(String s) {//有效括号
    Stack<Character> stack = new Stack<>();
    for(char c : s.toCharArray()){
        if(c == '(' || c == '[' || c =='{'){
            stack.push(c);
        }else {
            if(stack.isEmpty()) return false;

            char top = stack.pop();
            if((c == 'c' && top == ')') ||(c == '[' && top == ']') || (c == '{' && top == '}')){
                continue;
            }else return false;
        }
    }
    return stack.isEmpty();

    }

    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {//将两个升序链表合并为一个新的 升序 链表并返回
        ListNode mergeList = new ListNode();
        ListNode head = mergeList;
        while(list1 != null || list2 != null){
            int l1 = (list1 != null)? list1.val : Integer.MAX_VALUE;
            int l2 = (list2 != null)? list2.val : Integer.MAX_VALUE;
            if(l1<l2) {
                head.next = new ListNode(l1);
                list1 = list1.next;
            }else{
                head.next = new ListNode(l2);
                list2= list2.next;
            }
            head = head.next;
        }
        return mergeList.next;

    }
    public List<String> generateParenthesis(int n) {
        List<String> result= new ArrayList<>();
        generateParenthesisHelp(result, "",0,0,n);
        return result;
    }
    void generateParenthesisHelp(List<String> result,String str, int l, int r, int n){
        if(l < r) return;
        if(l > n) return;
        if(r > n) return;
        if(l == n && r == n) {
            result.add(str);
            return;
        }
        generateParenthesisHelp(result,str+"(",l+1,r,n);
        generateParenthesisHelp(result,str+")",l,r+1,n);
    }


    public ListNode mergeKLists(ListNode[] lists) {//给你一个链表数组，每个链表都已经按升序排列。


        int totalNodes = 0;
        for (ListNode node : lists) {
            totalNodes += countNodes(node);
        }

        // 存储所有节点值的数组
        int[] nodeValues = new int[totalNodes];
        int index = 0;
        for (ListNode node : lists) {
            while (node != null) {
                nodeValues[index++] = node.val;
                node = node.next;
            }
        }

        // 对节点值数组排序
        Arrays.sort(nodeValues);

        // 根据排序后的数组构建链表
        ListNode dummy = new ListNode(-1); // 哑节点
        ListNode current = dummy;
        for (int value : nodeValues) {
            current.next = new ListNode(value);
            current = current.next;
        }

        return dummy.next;
    }  //请你将所有链表合并到一个升序链表中，返回合并后的链表。// 统计所有节点数量

        // 辅助函数，用于统计链表中的节点数量
        private int countNodes(ListNode node) {
            int count = 0;
            while (node != null) {
                count++;
                node = node.next;
            }
            return count;
        }
    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode();
        dummy.next = head;
        ListNode tempHead = new ListNode(-1,dummy);
        ListNode p1 = new ListNode();
        ListNode p2 = new ListNode();
        ListNode p3 = new ListNode();
        while(dummy.next != null){
            if(dummy.next.next == null) return tempHead.next.next;
            p1.next = dummy.next;
            p2.next = p1.next.next;
            p3.next = p2.next.next;

            dummy.next = p2.next;
            p2.next.next = p1.next;
            p1.next.next = p3.next;

            dummy = dummy.next.next;
        }
        return tempHead.next.next;

    }

    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode prevGroupEnd = dummy;

        while (true) {
            ListNode cur = prevGroupEnd.next;
            ListNode[] p = new ListNode[k + 1];
            boolean canReverse = true;
            for (int i = 0; i < k; i++) {
                if (cur == null) {
                    canReverse = false;
                    break;
                }
                p[i] = cur;
                cur = cur.next;
            }
            if (!canReverse) break;

            ListNode nextGroupStart = p[0].next;
            for (int i = k - 1; i > 0; i--) {
                p[i].next = p[i - 1];
            }
            prevGroupEnd.next = p[k - 1];
            p[0].next = cur;
            prevGroupEnd = p[0];
        }

        return dummy.next;



    }//给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。

    public int removeDuplicates(int[] nums) {
            int k = nums.length;
            int len = 1;
            for(int i = 1;i < nums.length;i++){
                if(nums[i] != nums[i-1] ){
                    nums[len] = nums[i];
                    len++;
                }
            }
            return len;

    }//给你一个 非严格递增排列 的数组 nums
    // ，请你 原地 删除重复出现的元素，使每个元素 只出现一次
    // ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致
    // 。然后返回 nums 中唯一元素的个数。
    public int removeElement(int[] nums, int val) {
        int len = 0;
        for(int i =0;i < nums.length;i++){
            if(nums[i] != val){
                nums[len] = nums[i];
                len++;
            }
        }
        return len;

    }//给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

    public int strStr(String haystack, String needle) {
        if(needle.length() > haystack.length()){
            return -1;
        }
        int len = needle.length();
        for(int i =0;i <= haystack.length()-len;i++){
            String str = haystack.substring(i,i+len);
            if(str.equals(needle)) return i;
        }
        return -1;
    }//找出字符串第一个匹配项的下标

    public int divide(int dividend, int divisor) {
        int flag1 = 0,flag2 = 0;
        int answer = 0;
        if(dividend < 0 && dividend != Integer.MIN_VALUE){
            flag1 = 1;
            dividend = Math.abs(dividend);
        }else if(dividend == Integer.MIN_VALUE){
            if (divisor == Integer.MIN_VALUE) return 1;
            if(divisor == -1) return Integer.MAX_VALUE;
            else {
                flag1 = 1;
                answer ++;
                if(divisor > 0) dividend =Math.abs(dividend+divisor);
                else if(divisor < 0) dividend = Math.abs(dividend-divisor);
            }
        }

        if(divisor < 0 && divisor != Integer.MIN_VALUE){
            flag2 = 1;
            divisor = Math.abs(divisor);
        } else if (divisor == Integer.MIN_VALUE) {
            return 0;
        }

    while(dividend >= divisor){
        long temp =divisor;
        long multiple = 1;
        while(dividend >= (temp << 1)){
            temp <<= 1;
            multiple <<= 1;
        }
        dividend = (int) (dividend - temp);
        answer += multiple;

    }
        if(flag1 == flag2) return answer;
        else   return -answer;

    }//定义整数除法



//public List<Integer> findSubstring(String s, String[] words) {//30题不会

    private ArrayList<Integer> findSubstringHelp(String s, String str){
        ArrayList<Integer> result = new ArrayList<>();
        int len1 = s.length();
        int len2 = str.length();
        int index = 0;
        while(len1 >= len2){
            int answer = s.substring(index).indexOf(str);
            if(answer == -1) return result;//找不到直接返回

            result.add(answer+index);
            index = index+answer + len2;
            len1 = len1 - index;
        }
        return result;
    }

    public void nextPermutation(int[] nums) {
        int numsLen = nums.length;
        if(numsLen == 1) return ;
        int flag = 0;//未发现字典
        int point  = numsLen-1;

        for(int i = numsLen -1;i > 0 ; i--){
            if (nums[i-1] < nums[i]) {
                flag = 1;
                point = i-1;
                break;
            }
        }
        if(flag == 0)  {
            nextPermutationreverse( nums,0);
            return ;
        }
        for(int i = numsLen -1;i > point ;i--){
            if(nums[i] > nums[point]){
                nextPermutationswap(nums,point,i);
                nextPermutationreverse( nums,point+1);
                return ;
            }
        }



    }//下一个排列
    private void nextPermutationswap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    private void nextPermutationreverse(int[] nums, int start) {
        int left = start, right = nums.length - 1;
        while (left < right) {
            nextPermutationswap(nums, left, right);
            left++;
            right--;
        }
    }

    public int longestValidParentheses(String s) {
        if (s.isEmpty() || s.length() == 1) return 0;

        int maxLen = 0;
        int n = s.length();
        int[] dp = new int[n];

        for(int i =1 ;i<n;i++){
            if(s.charAt(i) == ')'){
                if(s.charAt(i-1) == '('){
                    dp[i] = (i > 2? dp[i-2] : 0) + 2;
                } else if (i - dp[i-1] > 0 && s.charAt(i - dp[i-1] - 1) == '(') {
                    dp[i] = dp[i-1] +(i - dp[i-1] >= 2 ? dp[i -dp[i-1] - 2] : 0)+2;
                }
                maxLen = Math.max(maxLen, dp[i]);
            }
        }


        return maxLen;
    }//最长有效括号

    public int search(int[] nums, int target) {
        int n = nums.length;
        if(n == 1&& target == nums[0]) return 0;
        else if (n == 1) return -1;

        int left =0,right = n -1;
        while(left <= right){
            int mid = (left + right) /2;
            if(nums[mid] == target) return mid;
            if(nums[mid] < nums[right]){//右边有序
                if(nums[mid] < target && nums[right] >= target){
                    left = mid +1;
                } else right = mid -1;
            } else  {//左边有序
                if(nums[mid] > target && target >= nums[left]){
                    right = mid -1;
                }else left = mid+1;
            }
        }
        return -1;


    }//33搜索旋转数组

    public int[] searchRange(int[] nums, int target) {
        int[] wrong = {-1,-1};
        int[] result = new int[2];
        int n = nums.length;
        int mid_answer = 0;
        if(n == 0) return wrong;
        
        int left = 0,right= n -1;
        while(left  <= right){
            if(nums[left]<=target && target <= nums[right] ){
                int mid = (left+right)/2;
                if(nums[mid] == target){
                    mid_answer = mid;
                    break;
                } else if (nums[mid] < target) {
                    left = mid + 1;
                }else right = mid - 1;

            }else return wrong;
        }
        int l_mid = mid_answer, r_mid = mid_answer;
        while(l_mid>=0 ){
            if( nums[l_mid] == target) {
                result[0] = l_mid;
                l_mid--;
            }else break;
        }
        while (r_mid < n){
            if(nums[r_mid] == target){
                result[1] = r_mid;
                r_mid++;
            }else break;
        }
        return result;
    }//34在排序数组中查找元素的第一个和最后一个元素

    public int searchInsert(int[] nums, int target) {  int n = nums.length;
        int left = 0, right = n - 1, ans = n;
        while (left <= right) {
            int mid = ((right - left) >> 1) + left;
            if (target <= nums[mid]) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;

    }//35搜索插入位置
    public boolean isValidSudoku(char[][] board) {
        int [][] row=new int [9][9];
        int [][] col=new int [9][9];
        int [][][] box = new int [3][3][9];
        for(int i = 0;i<9;i++){
            for(int j=0;j<9;j++){
                char c = board[i][j];
                if('0' <= c && c <= '9'){
                    int index = c - '1';
                    row[i][index]++;
                    col[j][index]++;
                    box[i/3][j/3][index]++;
                    if(row[i][index]>1 || col[j][index]>1 || box[i/3][j/3][index]>1) return false;
                }
            }
        }
        return  true;
    }//36数独是否合法


    public void solveSudoku(char[][] board) {
        solveSudokuBackTracking(board);

    }//37回溯法解数独
    private boolean solveSudokuBackTracking(char[][] board){
        for(int i = 0;i<board.length;i++){
            for(int j =0 ;j<board.length;j++){
                if(board[i][j] == '.'){
                    for(char k = '1';k<='9';k++){
                        if(solveSudokuIsValid(i,j,k,board)){
                            board[i][j] = k;
                            if(solveSudokuBackTracking(board)) return true;//找到合适的
                            board[i][j] = '.';//回溯回点
                        }
                    }
                    return false;//9个数字试完了
                }
            }
        }
        return true;//没返回flase说明找到了
    }//37.help1
    private boolean solveSudokuIsValid(int row,int col, char val,char[][] board){
        for(int i =0;i<9;i++){
            if(board[row][i] == val) return false;//遍历横
        }
        for(int i =0;i<9;i++){//遍历竖
            if(board[i][col] == val) return false;
        }
        int rowStart = (row/3) * 3;
        int colStart = (col/3) * 3;
        for(int i=rowStart;i<rowStart+3;i++){//遍历小正方形
            for(int j=colStart;j<colStart+3;j++){
                if(board[i][j] == val) return false;
            }
        }
        return true;
    }//37.help2


}
