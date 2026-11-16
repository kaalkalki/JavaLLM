package com.javallm.nn;

import com.javallm.tensor.Tensor;

import java.util.Objects;

/**
 * Applies rotary positional embeddings to adjacent pairs in each attention head.
 *
 * <p>The sequence axis is the second-last axis and the head feature axis must be
 * positive and even. {@link #forward(Tensor, int)} supports nonzero starting
 * positions for incremental decoding.</p>
 */
public final class RoPE implements PositionalEncoding {

    private static final double DEFAULT_THETA = 10000.0;

    private final double theta;

    public RoPE() {
        this(DEFAULT_THETA);
    }

    public RoPE(double theta) {
        if (!Double.isFinite(theta) || theta <= 0.0) {
            throw new IllegalArgumentException("Theta must be positive and finite");
        }
        this.theta = theta;
    }

    public double theta() {
        return theta;
    }

    @Override
    public Tensor forward(Tensor input) {
        return forward(input, 0);
    }

    @Override
    public Tensor forward(Tensor input, int startPosition) {
        Objects.requireNonNull(input, "Input cannot be null");
        if (input.rank() < 2) {
            throw new IllegalArgumentException("RoPE input must have rank >= 2");
        }
        if (startPosition < 0) {
            throw new IllegalArgumentException("Start position must be non-negative");
        }

        int[] shape = input.shape();
        int sequenceAxis = shape.length - 2;
        int sequenceLength = shape[sequenceAxis];
        int headDimension = shape[shape.length - 1];
        if (headDimension <= 0 || headDimension % 2 != 0) {
            throw new IllegalArgumentException("RoPE head dimension must be positive and even");
        }

        if (input.size() == 0) {
            return Tensor.of(new float[0], shape);
        }

        int rowsPerSequence = 1;
        for (int axis = 0; axis < sequenceAxis; axis++) {
            rowsPerSequence = Math.multiplyExact(rowsPerSequence, shape[axis]);
        }

        float[] inputData = input.toArray();
        float[] outputData = new float[inputData.length];

        for (int row = 0; row < rowsPerSequence; row++) {
            for (int sequenceIndex = 0; sequenceIndex < sequenceLength; sequenceIndex++) {
                long position = (long) startPosition + sequenceIndex;
                int rowOffset = (row * sequenceLength + sequenceIndex) * headDimension;

                for (int pair = 0; pair < headDimension / 2; pair++) {
                    double frequency = 1.0 / Math.pow(theta, (2.0 * pair) / headDimension);
                    double angle = position * frequency;
                    float cosine = (float) Math.cos(angle);
                    float sine = (float) Math.sin(angle);

                    int evenIndex = rowOffset + pair * 2;
                    int oddIndex = evenIndex + 1;
                    float evenValue = inputData[evenIndex];
                    float oddValue = inputData[oddIndex];

                    outputData[evenIndex] = evenValue * cosine - oddValue * sine;
                    outputData[oddIndex] = evenValue * sine + oddValue * cosine;
                }
            }
        }

        return Tensor.of(outputData, shape);
    }
}
