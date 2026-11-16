
package com.javallm.nn;

import com.javallm.tensor.Tensor;

import java.util.Objects;

/**
 * Converts values along a selected tensor axis into normalized probabilities.
 *
 * <p>The default axis is the final dimension. Negative axes count backward from
 * the final dimension, and the implementation subtracts each slice maximum for
 * numerical stability.</p>
 */
public final class Softmax implements Layer {

    private final int axis;

    public Softmax(int axis) {
        this.axis = axis;
    }

    public Softmax() {
        this(-1);
    }

    public int axis() {
        return axis;
    }

    @Override
    public Tensor forward(Tensor input) {

        Objects.requireNonNull(input, "Input cannot be null");

        if (input.rank() < 1) {
            throw new IllegalArgumentException("Softmax input must have rank >= 1");
        }

        int normalizedAxis = axis < 0 ? input.rank() + axis : axis;

        if (normalizedAxis < 0 || normalizedAxis >= input.rank()) {
            throw new IllegalArgumentException("Invalid softmax axis: " + axis);
        }

        int[] shape = input.shape();
        float[] inputData = input.toArray();
        float[] outputData = new float[input.size()];

        int outerSize = 1;
        int innerSize = 1;
        int axisSize = shape[normalizedAxis];

        for (int dimension = 0; dimension < normalizedAxis; dimension++) {
            outerSize *= shape[dimension];
        }

        for (int dimension = normalizedAxis + 1; dimension < shape.length; dimension++) {
            innerSize *= shape[dimension];
        }

        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                float maximum = Float.NEGATIVE_INFINITY;
                for (int axisIndex = 0; axisIndex < axisSize; axisIndex++) {
                    int index = outer * axisSize * innerSize + axisIndex * innerSize + inner;
                    maximum = Math.max(maximum, inputData[index]);
                }

                int positiveInfinityCount = 0;
                for (int axisIndex = 0; axisIndex < axisSize; axisIndex++) {
                    int index = outer * axisSize * innerSize + axisIndex * innerSize + inner;
                    if (inputData[index] == Float.POSITIVE_INFINITY) {
                        positiveInfinityCount++;
                    }
                }

                if (positiveInfinityCount > 0 || maximum == Float.NEGATIVE_INFINITY) {
                    int selectedCount = positiveInfinityCount > 0 ? positiveInfinityCount : axisSize;
                    float probability = 1.0f / selectedCount;
                    for (int axisIndex = 0; axisIndex < axisSize; axisIndex++) {
                        int index = outer * axisSize * innerSize + axisIndex * innerSize + inner;
                        boolean selected = positiveInfinityCount == 0
                                || inputData[index] == Float.POSITIVE_INFINITY;
                        outputData[index] = selected ? probability : 0.0f;
                    }
                    continue;
                }

                float sum = 0.0f;
                for (int axisIndex = 0; axisIndex < axisSize; axisIndex++) {
                    int index = outer * axisSize * innerSize + axisIndex * innerSize + inner;
                    float exponential = (float) Math.exp(inputData[index] - maximum);
                    outputData[index] = exponential;
                    sum += exponential;
                }

                for (int axisIndex = 0; axisIndex < axisSize; axisIndex++) {
                    int index = outer * axisSize * innerSize + axisIndex * innerSize + inner;
                    outputData[index] /= sum;
                }
            }
        }

        return Tensor.of(outputData, shape);
    }
}
