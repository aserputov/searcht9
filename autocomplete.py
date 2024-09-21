from trie import Trie

class AutocompleteSystem:

    def __init__(self, sentences, times):
        self.inst = Trie()
        self.inst.addNodes(sentences, times) 
        self.state = ""

    def input(self, c: str):
        if c == "#":
            self.inst.addNodes([self.state], [1])
            self.state = ""
            return []
        self.state += c
        return self.inst.search(self.state)