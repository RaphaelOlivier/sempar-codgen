Did you receive any help whatsoever from anyone in solving this assignment?   No
Did you give any help whatsoever to anyone in solving this assignment?  No.
Did you find or come across code that implements any part of this assignment?  yes, I did,
I came across the follwing websites which have a part of the code or the pseudo code:
https://en.wikipedia.org/wiki/Levenshtein_distance
https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
Geeks for geeks to understand how to build a trie
https://murilo.wordpress.com/2011/02/01/fast-and-easy-levenshtein-distance-using-a-trie-in-c/

TASK 2
NOTE: I have coded the Damerau-Levenshtein algorithm.

BONUS TASK 1

We have observed that both Task 1 and Task 2 take a really long time. This is because, for every word in the raw.text, we need to calculate its Levenshtein distance with every word in the dictinonary to find the least Levenshtein distance. There will be so many words in the dictionary which can never be the original word of a mis-spelt word and calculating the Levenshtein distance for that doesnt make sense. Hence, we needed a mechanism to search faster for the best words which could be the original words of the Levenshtein distance. Also we observe that when we use a dynamic programming to construct the matrix, we don't need the whole matrix to predict the operations for the next character in the target string. Let the rows represent the characters in the source string and the columns represent the columns in the target string.

We can use a trie which is a tree where each path of the tree holds a string. All words which have a similar prefix, will fall into the same path. Just that the path keeps increasing till the longest word in the dictionary with that prefix if inserted into the trie. Each node in the trie stores a partial word or a complete word. Now when we process the raw.txt, since there is an ordering in the trie, we need to do less number of computations. We can use the previous row for our present row calculations because we will process all words with the same prefix at the same time. Then after this we will process the next branch of the trie. Bascially we will never have to recompute the a row for a similar prefix. Thus, only one row is created for each node in the tree, and this row is built using the previous node's row. This algorithm would be much faster than the first one since we are searching through the branch of a trie, which will at max take O(no_of node_in_trie * max_word_length) which is much less the previous task's time. 


