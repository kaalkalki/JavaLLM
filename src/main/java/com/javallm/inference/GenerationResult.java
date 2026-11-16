package com.javallm.inference;

import java.util.List;
import java.util.Objects;

/**
 * Generated continuation text and the token IDs that produced it.
 *
 * <p>The token list is copied on construction so callers can inspect the exact
 * generated sequence without changing the stored result.</p>
 */
public final class GenerationResult {

    private final String text;
    private final List<Integer> tokenIds;

    public GenerationResult(String text, List<Integer> tokenIds) {
        this.text = Objects.requireNonNull(text, "Text cannot be null");
        this.tokenIds = List.copyOf(Objects.requireNonNull(tokenIds, "Token IDs cannot be null"));
    }

    public String text() {
        return text;
    }

    public List<Integer> tokenIds() {
        return tokenIds;
    }
}
