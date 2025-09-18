import collections


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    """
    In fact, this Trie is a letter tree.
    root is a fake node, its function is only the begin of a word, same as <bow>
    the the first layer is all the word's possible first letter, for example, '中国'
        its first letter is '中'
    the second the layer is all the word's possible second letter.
    and so on
    """

    def __init__(self, use_single=True):
        self.root = TrieNode()
        self.max_depth = 0
        if use_single:
            self.min_len = 0
        else:
            self.min_len = 1

    def insert(self, word):
        current = self.root
        deep = 0
        for letter in word:
            current = current.children[letter]
            deep += 1
        current.is_word = True
        if deep > self.max_depth:
            self.max_depth = deep

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)

            if current is None:
                return False
        return current.is_word

    def enumerateMatch(self, str, space=""):
        """
        Args:
            str: 需要匹配的词
        Return:
            返回匹配的词, 如果存在多字词，则会筛去单字词
        """
        matched = []
        while len(str) > self.min_len:
            if self.search(str):
                matched.insert(0, space.join(str[:]))  # 短的词总是在最前面
            del str[-1]

        if len(matched) > 1 and len(matched[0]) == 1:  # filter single character word
            matched = matched[1:]

        return matched

    def get_lexicon(self, sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_word:
                    result.append([i, j, sentence[i:j+1]])

        return result
