
package com.javallm.nn;

import com.javallm.tensor.Tensor;

import java.util.Objects;

/**
 * Looks up token vectors from a vocabulary-by-feature weight tensor.
 *
 * <p>Each input token ID selects one row from the weight matrix. The result has
 * shape {@code [numberOfTokens, embeddingSize]}.</p>
 */
public final class Embedding {

    private final Tensor weights;
    private final int vocabularySize;
    private final int embeddingSize;

    public Embedding(Tensor weights) {

        Objects.requireNonNull(weights, "Weights cannot be null");

        if (weights.rank() != 2) {
            throw new IllegalArgumentException("Embedding weights must be a rank-2 tensor");
        }

        this.weights = weights;
        this.vocabularySize = weights.dimension(0);
        this.embeddingSize = weights.dimension(1);
    }

    public Tensor forward(int... tokenIds) {

        Objects.requireNonNull(tokenIds, "Token IDs cannot be null");
        int outputSize;
        try {
            outputSize = Math.multiplyExact(tokenIds.length, embeddingSize);
        } catch (ArithmeticException exception) {
            throw new IllegalArgumentException("Embedding result exceeds supported integer range", exception);
        }
        float[] embeddingData = new float[outputSize];
        for (int tokenIndex = 0; tokenIndex < tokenIds.length; tokenIndex++) {
            int tokenId = tokenIds[tokenIndex];
            if (tokenId < 0 || tokenId >= vocabularySize) {
                throw new IllegalArgumentException("Token ID out of vocabulary range: " + tokenId);
            }

            for (int dimension = 0; dimension < embeddingSize; dimension++) {
                embeddingData[tokenIndex * embeddingSize + dimension] = weights.get(tokenId, dimension);
            }
        }

        return Tensor.of(embeddingData, tokenIds.length, embeddingSize);
    }

    public Tensor weights() {
        return weights;
    }

    public int vocabularySize() {
        return vocabularySize;
    }

    public int embeddingSize() {
        return embeddingSize;
    }
}
