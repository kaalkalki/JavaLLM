
package com.javallm.nn;

import com.javallm.tensor.Tensor;

import java.util.Objects;

/**
 * Normalizes each final-dimension feature vector by its root mean square.
 *
 * <p>The learned rank-1 weights scale normalized features, and the configured
 * positive epsilon stabilizes the inverse root-mean-square calculation.</p>
 */
public final class RMSNorm implements Normalization {

    private final Tensor weights;
    private final float epsilon;

    public RMSNorm(Tensor weights, float epsilon) {

        Objects.requireNonNull(weights, "Weights cannot be null");

        if (weights.rank() != 1) {
            throw new IllegalArgumentException(
                    "RMSNorm weights must be a rank-1 tensor"
            );
        }

        if (weights.dimension(0) == 0) {
            throw new IllegalArgumentException("RMSNorm feature dimension must be positive");
        }

        if (!Float.isFinite(epsilon) || epsilon <= 0.0f) {
            throw new IllegalArgumentException(
                    "Epsilon must be positive"
            );
        }

        this.weights = weights;
        this.epsilon = epsilon;
    }

    public Tensor weights() {
        return weights;
    }

    @Override
    public int featureSize() {
        return weights.dimension(0);
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "Input cannot be null");
        if (input.rank() < 1) {
            throw new IllegalArgumentException("RMSNorm input must have rank >= 1");
        }
        int features = input.dimension(input.rank() - 1);
        if (features == 0) {
            throw new IllegalArgumentException("RMSNorm feature dimension must be positive");
        }
        if (weights.dimension(0) != features) {
            throw new IllegalArgumentException("Input features do not match RMSNorm weights");
        }
        int[] inputShape = input.shape();
        int sequenceSize = features;
        int totalElements = input.size();
        float[] outputData = new float[totalElements];
        float[] inputData = input.toArray();
        float[] weightData = weights.toArray();
        for (int row = 0; row < totalElements / sequenceSize; row++) {
            int rowOffset = row * sequenceSize;
            float sumOfSquares = 0.0f;
            for (int column = 0; column < sequenceSize; column++) {
                float value = inputData[rowOffset + column];
                sumOfSquares += value * value;
            }

            float meanSquare = sumOfSquares / sequenceSize;
            float inverseRootMeanSquare = 1.0f / (float) Math.sqrt(meanSquare + epsilon);

            for (int column = 0; column < sequenceSize; column++) {
                float normalizedValue = inputData[rowOffset + column] * inverseRootMeanSquare;
                outputData[rowOffset + column] = normalizedValue * weightData[column];
            }
        }

        return Tensor.of(outputData, inputShape);
    }
}
