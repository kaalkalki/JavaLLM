package com.javallm.inference;

/**
 * Sampling controls for autoregressive generation.
 *
 * <p>Temperature zero selects the highest-scoring token. Positive temperature
 * samples from scaled scores; top-k limits the candidate count and top-p keeps
 * the smallest high-probability candidate prefix meeting the requested mass.</p>
 */
public final class GenerationConfig {

    private final int maximumNewTokens;
    private final double temperature;
    private final int topK;
    private final double topP;

    public GenerationConfig(int maximumNewTokens, double temperature, int topK, double topP) {
        if (maximumNewTokens < 0) {
            throw new IllegalArgumentException("Maximum new tokens must be non-negative");
        }
        if (!Double.isFinite(temperature) || temperature < 0.0) {
            throw new IllegalArgumentException("Temperature must be finite and non-negative");
        }
        if (topK < 0) {
            throw new IllegalArgumentException("Top-k must be non-negative");
        }
        if (!Double.isFinite(topP) || topP <= 0.0 || topP > 1.0) {
            throw new IllegalArgumentException("Top-p must be in the interval (0, 1]");
        }
        this.maximumNewTokens = maximumNewTokens;
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
    }

    public static GenerationConfig greedy(int maximumNewTokens) {
        return new GenerationConfig(maximumNewTokens, 0.0, 1, 1.0);
    }

    public int maximumNewTokens() {
        return maximumNewTokens;
    }

    public double temperature() {
        return temperature;
    }

    public int topK() {
        return topK;
    }

    public double topP() {
        return topP;
    }
}
