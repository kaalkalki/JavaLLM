package com.javallm.inference;

/**
 * Generates a continuation for a text prompt using a loaded language model.
 *
 * <p>Implementations apply a generation configuration and return both decoded
 * text and the generated token IDs.</p>
 */
public interface TextGenerator {

    GenerationResult generate(String prompt, GenerationConfig config);
}
