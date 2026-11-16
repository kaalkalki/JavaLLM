package com.javallm.tokenizer;

import java.util.List;

/**
 * Converts text to model vocabulary IDs and converts IDs back to text.
 *
 * <p>Implementations expose vocabulary and boundary-token metadata so inference
 * code can prepare prompts and recognize end-of-sequence tokens without
 * depending on a particular tokenization algorithm.</p>
 */
public interface Tokenizer {

    int vocabularySize();

    int bosTokenId();

    int eosTokenId();

    boolean addsBosToken();

    List<Integer> encode(String text);

    String decode(List<Integer> tokenSequence);

    String token(int tokenId);
}
