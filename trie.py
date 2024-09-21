import heapq
from trieNode import TrieNode

class Trie:
    def __init__(self):
        self.head = TrieNode()
    
    def addNodes(self, sentence, times):
        for word, t in zip(sentence, times):
            tmp = self.head
            for w in word:
                if w in tmp.children:
                    tmp = tmp.children[w]
                else:
                    tmp.children[w] = TrieNode()
                    tmp = tmp.children[w]
            
                if word in tmp.cl:
                    tmp.cl[word] = t + tmp.cl[word]
                else:
                    tmp.cl[word] = t


                arr = []
                for i in tmp.heap:
                    if i[1] != word:
                        arr.append(i)
                
                tmp.heap = arr
                heapq.heapify(tmp.heap)


                heapq.heappush(tmp.heap, (-tmp.cl[word], word))
                
                print(tmp.heap)
                # ans = []
                # if len(tmp.heap) > 3:
                #     cnt = 0
                #     while cnt != 3:
                #         ans.append(heapq.heappop(tmp.heap))
                #         cnt += 1
                    
                #     tmp.heap = ans
            
    
    def search(self, word):
        tmp = self.head
        for w in word:
            if w in tmp.children:
                tmp = tmp.children[w]
            else:
                return []
        
        ans = []
        for i in sorted(tmp.heap):
            ans.append(i[1])
        
        return ans