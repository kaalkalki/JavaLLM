package com.javallm.model;

import com.javallm.tokenizer.Tokenizer;

import java.util.Objects;

/**
 * A decoder model and tokenizer loaded together from one GGUF file.
 *
 * <p>The context length is carried with the model so generation can validate the
 * prompt and size its KV cache consistently with the source model metadata.</p>
 */
public final class LoadedModel {

    private final LanguageModel model;
    private final Tokenizer tokenizer;
    private final int contextLength;

    public LoadedModel(LanguageModel model, Tokenizer tokenizer, int contextLength) {
        this.model = Objects.requireNonNull(model, "Model cannot be null");
        this.tokenizer = Objects.requireNonNull(tokenizer, "Tokenizer cannot be null");
        if (contextLength <= 0) {
            throw new IllegalArgumentException("Context length must be positive");
        }
        this.contextLength = contextLength;
    }

    public LanguageModel model() {
        return model;
    }

    public Tokenizer tokenizer() {
        return tokenizer;
    }

    public int contextLength() {
        return contextLength;
    }
}
