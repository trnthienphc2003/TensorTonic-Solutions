from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        # YOUR CODE HERE
        n = len(word)
        if n > self.max_word_len:
            return [self.unk_token]
        st, en = 0, n
        ans = []
        while st < n:
            en = n
            token = word[st:en]
            if st > 0:
                token = "##" + token
            while en > st and not (token in self.vocab.keys()):
                en -= 1
                token = token[:-1]
            if st >= en:
                # assert False, f'{st} -> {en}'
                # assert False, ans
                return [self.unk_token]
            ans.append(token)
            st = en

        return ans
